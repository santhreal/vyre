//! W3-3 (attribution) gate: `GpuLiteralSet::scan_all_timed` returns
//! backend-owned timing WITHOUT changing the complete match set, and reports
//! WHICH auto-resize dispatch the timing describes.
//!
//! `scan_all` may dispatch twice (default capacity, then a resize to the exact
//! device count). The timed twin locks that the returned matches are identical to
//! the untimed `scan_all`, that `resized` honestly reports whether a re-dispatch
//! happened, and that `device_ns` is `None` (a loud absence, not a fabricated
//! zero) on the CPU reference backend.

use vyre_driver_reference::CpuRefBackend;
use vyre_foundation::match_result::Match;
use vyre_libs::scan::GpuLiteralSet;

#[test]
fn timed_scan_all_equals_untimed_and_reports_timing() {
    let backend = CpuRefBackend;
    let patterns: [&[u8]; 3] = [b"alpha", b"kilo", b"tango"];
    let set = GpuLiteralSet::compile(&patterns);
    // Several matches, including repeats, so the match set is non-trivial.
    let haystack = b"alpha__kilo__tango__alpha__tango__kilo__alpha".to_vec();

    let plain = set.scan_all(&backend, &haystack).expect("untimed scan_all");

    let mut timed_matches: Vec<Match> = Vec::new();
    let result = set
        .scan_all_timed(&backend, &haystack, &mut timed_matches)
        .expect("timed scan_all");

    // Correctness: the timed match set must equal the untimed one exactly.
    assert_eq!(
        timed_matches, plain,
        "timed scan_all matches must equal the untimed scan_all matches"
    );

    // Non-vacuous: the fixture must actually produce matches.
    assert!(
        !timed_matches.is_empty(),
        "fixture must produce at least one match"
    );

    // This fixture fits the default capacity, so no resize is expected.
    assert!(
        !result.resized,
        "small fixture fits default capacity; no auto-resize should occur"
    );

    // Attribution: raw outputs carried, honest device_ns.
    assert!(
        result.timed.outputs.len() >= 2,
        "TimedDispatchResult must carry the count + matches output buffers"
    );
    assert!(
        result.timed.device_ns.is_none(),
        "CPU reference backend has no device timer; device_ns must be None, not a fabricated zero"
    );
}

#[test]
fn timed_scan_all_clears_stale_matches() {
    let backend = CpuRefBackend;
    let patterns: [&[u8]; 1] = [b"needle"];
    let set = GpuLiteralSet::compile(&patterns);
    let haystack = b"..needle..needle..".to_vec();

    // Pre-seed with stale entries; the timed scan must clear them.
    let mut matches: Vec<Match> = vec![Match::new(42, 1, 2); 3];
    let result = set
        .scan_all_timed(&backend, &haystack, &mut matches)
        .expect("timed scan_all");
    assert!(
        !matches.iter().any(|m| m.pattern_id == 42),
        "stale pre-seeded matches must be cleared, not accumulated"
    );
    assert_eq!(matches.len(), 2, "exactly the two `needle` matches remain");
    assert!(!result.resized);

    // Determinism across repeats.
    let mut again: Vec<Match> = Vec::new();
    set.scan_all_timed(&backend, &haystack, &mut again)
        .expect("repeat timed scan_all");
    assert_eq!(
        matches, again,
        "repeated timed scan_all must be deterministic"
    );
}

/// The `resized == true` branch: a corpus with more matches than the default
/// initial capacity (10_000) forces the auto-resize re-dispatch, and
/// `ScanAllTimed.resized` must honestly report it so the timing is not
/// misattributed as a single launch. The matches must still be complete and
/// exact (auto-resize never truncates).
#[test]
fn timed_scan_all_reports_resize_and_stays_exact() {
    let backend = CpuRefBackend;
    let set = GpuLiteralSet::compile(&[b"a" as &[u8]]);
    // 25_000 'a' bytes => 25_000 single-byte matches, far past the 10_000 cap.
    let n = 25_000usize;
    let haystack = vec![b'a'; n];

    let plain = set.scan_all(&backend, &haystack).expect("untimed scan_all");

    let mut timed_matches: Vec<Match> = Vec::new();
    let result = set
        .scan_all_timed(&backend, &haystack, &mut timed_matches)
        .expect("timed scan_all with resize");

    assert!(
        result.resized,
        "a corpus of {n} matches exceeds the 10_000 initial cap; resized must be true"
    );
    assert_eq!(
        timed_matches.len(),
        n,
        "auto-resize must return every match past the cap, got {}",
        timed_matches.len()
    );
    assert_eq!(
        timed_matches, plain,
        "timed (resized) scan_all matches must equal the untimed scan_all matches"
    );
    // The reported timing is the resize RE-dispatch (the one that produced these
    // matches); on CpuRefBackend it carries the honest None device time.
    assert!(
        result.timed.device_ns.is_none(),
        "CPU reference backend has no device timer; device_ns must be None"
    );
}
