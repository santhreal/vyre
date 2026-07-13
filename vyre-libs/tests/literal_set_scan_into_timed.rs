//! W3-3 (attribution) gate: `GpuLiteralSet::scan_into_timed` returns
//! backend-owned timing WITHOUT changing the position-scan result.
//!
//! Contract: the decoded `(pattern_id, start, end)` triples are identical to the
//! untimed `scan_into` (same program, same inputs, only `dispatch_borrowed_timed`
//! vs `dispatch_borrowed` differs), and the returned `TimedDispatchResult` carries
//! honest timing (`device_ns` is `None`: a loud absence, not a fabricated zero 
//! on the CPU reference backend that has no device timer).

use vyre_driver_reference::CpuRefBackend;
use vyre_libs::scan::{GpuLiteralSet, LiteralMatch};

fn sorted_triples(matches: &[LiteralMatch]) -> Vec<(u32, u32, u32)> {
    let mut v: Vec<(u32, u32, u32)> = matches
        .iter()
        .map(|m| (m.pattern_id, m.start, m.end))
        .collect();
    v.sort_unstable();
    v
}

#[test]
fn timed_scan_into_equals_untimed_and_reports_timing() {
    let backend = CpuRefBackend;
    let patterns: [&[u8]; 3] = [b"alpha", b"kilo", b"tango"];
    let set = GpuLiteralSet::compile(&patterns);
    // Multiple occurrences so the match set is non-trivial.
    let haystack = b"__alpha__kilo__tango__alpha__tango__";
    let max_matches = 64;

    let mut plain = Vec::new();
    set.scan_into(&backend, haystack, max_matches, &mut plain)
        .expect("untimed scan_into");

    let mut timed_matches = Vec::new();
    let timed = set
        .scan_into_timed(&backend, haystack, max_matches, &mut timed_matches)
        .expect("timed scan_into");

    // Correctness: the timed path must not change the decoded matches.
    assert_eq!(
        sorted_triples(&timed_matches),
        sorted_triples(&plain),
        "timed scan_into matches must equal the untimed scan_into matches"
    );
    // Non-vacuous: the fixture actually produces matches (guards a trivial pass).
    assert!(
        !timed_matches.is_empty(),
        "fixture must produce at least one match"
    );

    // Attribution: honest timing (the reference backend has no device timer).
    assert!(
        !timed.outputs.is_empty(),
        "TimedDispatchResult must carry the raw scan output bytes"
    );
    assert!(
        timed.device_ns.is_none(),
        "CPU reference backend has no device timer; device_ns must be None, not a fabricated zero"
    );
}

#[test]
fn timed_scan_into_clears_stale_matches() {
    // scan_into_timed must clear the caller's buffer first, so a reused Vec does
    // not accumulate matches across calls.
    let backend = CpuRefBackend;
    let set = GpuLiteralSet::compile(&[b"kilo".as_slice()]);
    let mut matches = vec![]; // will be reused
    set.scan_into_timed(&backend, b"kilo kilo", 64, &mut matches)
        .expect("first timed scan");
    let first_len = matches.len();
    set.scan_into_timed(&backend, b"kilo", 64, &mut matches)
        .expect("second timed scan");
    assert_eq!(
        matches.len(),
        1,
        "second scan (one occurrence) must not accumulate the first scan's {first_len} matches"
    );
}
