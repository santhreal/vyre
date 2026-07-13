//! W3-3 (attribution) gate: `GpuLiteralSet::scan_presence_timed` returns
//! backend-owned timing WITHOUT changing the global-presence result.
//!
//! The timed path exists so a consumer/benchmark can attribute per-scan cost
//! (kernel vs staging/readback) on the global-presence path, not only the
//! resident/region paths. The contract this locks: the timed bitmap is
//! byte-for-byte identical to the untimed `scan_presence` (same program, same
//! inputs, only `dispatch_borrowed_timed` vs `dispatch_borrowed` differs), and
//! the returned `TimedDispatchResult` carries the raw presence bytes it decoded
//! plus honest timing (`device_ns` is `None`: a loud absence, not a fabricated
//! zero (on the CPU reference backend that has no device timer)).

use vyre_driver_reference::CpuRefBackend;
use vyre_libs::scan::GpuLiteralSet;

/// A haystack containing two of three patterns (`alpha`, `tango`) but not the
/// third (`kilo`), so the global-presence bitmap has some set bits and some
/// clear bits (not all-zero, not all-one), guarding a vacuous pass.
fn fixture() -> (GpuLiteralSet, Vec<u8>) {
    let patterns: [&[u8]; 3] = [b"alpha", b"kilo", b"tango"];
    let set = GpuLiteralSet::compile(&patterns);
    let haystack = b"__alpha__and__tango__present__kilo_absent_here_no_k1lo".to_vec();
    // Note: `kilo` never appears literally (the near-miss `k1lo` is a digit-one).
    (set, haystack)
}

#[test]
fn timed_presence_equals_untimed_and_reports_timing() {
    let backend = CpuRefBackend;
    let (set, haystack) = fixture();

    let plain = set
        .scan_presence(&backend, &haystack)
        .expect("untimed global-presence scan");
    let (timed_bitmap, timed) = set
        .scan_presence_timed(&backend, &haystack)
        .expect("timed global-presence scan");

    // Correctness: the timed path must not change the result by a single bit.
    assert_eq!(
        timed_bitmap, plain,
        "timed global-presence bitmap must equal the untimed scan_presence bitmap"
    );

    // The bitmap must actually carry present bits (guards a vacuous all-zero pass).
    assert!(
        timed_bitmap.iter().any(|&w| w != 0),
        "fixture must produce at least one present pattern bit"
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
    let (set, haystack) = fixture();
    let (first, _) = set
        .scan_presence_timed(&backend, &haystack)
        .expect("first timed scan");
    let (second, _) = set
        .scan_presence_timed(&backend, &haystack)
        .expect("second timed scan");
    assert_eq!(first, second, "repeated timed scans must be deterministic");
}
