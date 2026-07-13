//! W3-3 (attribution) gate: `GpuLiteralSet::scan_presence_by_region_timed`
//! returns backend-owned timing WITHOUT changing the presence result.
//!
//! The timed path exists so a consumer/benchmark can attribute per-scan cost
//! (kernel vs staging/readback) on the hot literal region-presence path, not
//! only the resident path. The contract this locks: the timed bitmap is
//! byte-for-byte identical to the untimed `scan_presence_by_region` (same
//! program, same inputs, only `dispatch_borrowed_timed` vs `dispatch_borrowed`
//! differs), and the returned `TimedDispatchResult` carries the raw presence
//! bytes it decoded plus honest timing (`device_ns` is `None`: a loud absence,
//! not a fabricated zero, on the CPU reference backend that has no device
//! timer).

use vyre_driver_reference::CpuRefBackend;
use vyre_libs::scan::GpuLiteralSet;

/// A haystack with two regions; region 0 contains `alpha`+`kilo`, region 1
/// contains `tango` only (so the two region rows differ (a meaningful bitmap)).
fn fixture() -> (GpuLiteralSet, Vec<u8>, Vec<u32>) {
    let patterns: [&[u8]; 3] = [b"alpha", b"kilo", b"tango"];
    let set = GpuLiteralSet::compile(&patterns);
    let region0 = b"__alpha__kilo__".to_vec();
    let region1 = b"..tango..".to_vec();
    let region_starts = vec![0u32, region0.len() as u32];
    let mut haystack = region0;
    haystack.extend_from_slice(&region1);
    (set, haystack, region_starts)
}

#[test]
fn timed_presence_equals_untimed_and_reports_timing() {
    let backend = CpuRefBackend;
    let (set, haystack, region_starts) = fixture();

    let plain = set
        .scan_presence_by_region(&backend, &haystack, &region_starts)
        .expect("untimed region-presence scan");
    let (timed_bitmap, timed) = set
        .scan_presence_by_region_timed(&backend, &haystack, &region_starts, 0)
        .expect("timed region-presence scan");

    // Correctness: the timed path must not change the result by a single bit.
    assert_eq!(
        timed_bitmap, plain,
        "timed region-presence bitmap must equal the untimed scan_presence_by_region bitmap"
    );

    // The bitmap must actually distinguish the two regions (guards a vacuous pass
    // where both rows are all-zero and equality is trivially true).
    assert!(
        timed_bitmap.iter().any(|&w| w != 0),
        "fixture must produce at least one present (pattern, region) bit"
    );

    // Attribution: the result carries the raw presence bytes it decoded from, and
    // honest timing, the reference backend has no device timer, so device_ns is
    // a loud None, never a fabricated 0.
    assert!(
        !timed.outputs.is_empty() && !timed.outputs[0].is_empty(),
        "TimedDispatchResult must carry the raw presence output bytes"
    );
    assert!(
        timed.device_ns.is_none(),
        "CPU reference backend has no device timer; device_ns must be None, not a fabricated zero"
    );
}

#[test]
fn timed_scan_is_stable_across_repeated_calls() {
    // The prepare path allocates fresh owned buffers each call; two timed scans
    // of the same input must return identical bitmaps (no cross-call state leak).
    let backend = CpuRefBackend;
    let (set, haystack, region_starts) = fixture();
    let (first, _) = set
        .scan_presence_by_region_timed(&backend, &haystack, &region_starts, 0)
        .expect("first timed scan");
    let (second, _) = set
        .scan_presence_by_region_timed(&backend, &haystack, &region_starts, 0)
        .expect("second timed scan");
    assert_eq!(first, second, "repeated timed scans must be deterministic");
}
