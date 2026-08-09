//! CUDA parity for the authenticated resident NFA scan route.
//!
//! The test keeps every resident resource and the executable in one materializer
//! generation, reuses the authenticated artifact across submissions, and checks
//! every result against the backend-neutral reference scan.

use vyre_driver_cuda as _;
use vyre_foundation::match_result::Match;
use vyre_scan::build_scan_session;

/// Sort matches into a deterministic order so the expected-set assertion is
/// independent of the kernel's per-workgroup emission order.
fn sorted(mut matches: Vec<Match>) -> Vec<Match> {
    matches.sort_by_key(|m| (m.start, m.end, m.pattern_id));
    matches
}

#[test]
fn resident_rule_pipeline_matches_reference_on_cuda() {

    // Patterns ab=0, cd=1, xyz=2. Haystack plants known, overlap-free hits:
    // "ab"@[2,4), "cd"@[6,8), "xyz"@[8,11), "ab"@[11,13). The NFA program declares a
    // STATIC input buffer of `input_len` bytes (the CUDA backend enforces it, unlike
    // wgpu), so the haystack length, `build`'s input_len, and `prepare_resident`'s
    // capacity must all agree: 16 here (a multiple of 4 so the packed length equals
    // the raw length). The trailing "www" adds no match.
    const HAYSTACK_LEN: u32 = 16;
    let pipeline = build_scan_session(&["ab", "cd", "xyz"], "input", "hits", HAYSTACK_LEN);
    let haystack = b"zzabqqcdxyzabwww";
    assert_eq!(haystack.len() as u32, HAYSTACK_LEN);

    // The NFA program statically declares the hits buffer for nfa::NUM_HIT_SLOTS
    // (10000 matches); CUDA enforces that static size, so scan + prepare_resident
    // must use this exact match cap (wgpu would accept any).
    const HIT_CAP: u32 = 10_000;

    let expected = sorted(pipeline.reference_scan(haystack));
    // The expected scan must find exactly the planted hits.
    assert_eq!(
        expected,
        vec![
            Match::new(0, 2, 4),
            Match::new(1, 6, 8),
            Match::new(2, 8, 11),
            Match::new(0, 11, 13),
        ],
        "CUDA reference scan must find exactly the planted ab/cd/xyz/ab hits"
    );

    let session = pipeline
        .prepare_resident("cuda", HAYSTACK_LEN as usize, HIT_CAP)
        .expect("prepare authenticated resident scan session on CUDA");

    // Re-dispatch several times: the authenticated artifact and NFA tables stay resident.
    let mut matches = Vec::new();
    let mut scratch = Vec::new();
    for iter in 0..4 {
        session
            .scan_into(haystack, &mut matches, &mut scratch)
            .expect("resident CUDA artifact submission");
        assert_eq!(
            sorted(matches.clone()),
            expected,
            "iteration {iter}: CUDA resident match set must equal the reference scan"
        );
    }

    session
        .free()
        .expect("free resident scan artifact resources on CUDA");
}
