//! Property + differential coverage for `GpuLiteralSet::scan_all`.
//!
//! The existing `literal_set_scan_all` tests are hand-picked corpora (all-`a`,
//! tail-past-cap, …). This adds the missing PROPERTY layer the Testing Contract
//! requires: the real dispatch path (`scan_all` on `CpuRefBackend`, the full
//! auto-resize + reference-eval control flow) must agree. EXACT `(pattern_id,
//! start, end)` triple set, not cardinality, with the independent `reference_scan`
//! plain-Rust DFA oracle across thousands of randomly generated literal sets and
//! haystacks, plus a by-construction case where the placed needle offsets are known
//! a priori. A dense 3-symbol alphabet (`a`/`b`/`c`) maximizes overlap density so
//! prefix/suffix aliasing, adjacent hits, and empty-match edges are all exercised.

use proptest::prelude::*;
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

/// One byte from the dense 3-symbol alphabet, so short literals collide often.
fn sym() -> impl Strategy<Value = u8> {
    (0usize..3).prop_map(|i| b"abc"[i])
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    /// The dispatched scan and the reference DFA must produce the identical match
    /// set for ANY literal set over ANY haystack (the definitive differential).
    #[test]
    fn scan_all_agrees_with_reference_over_random_literal_sets(
        literals in prop::collection::vec(prop::collection::vec(sym(), 1..6), 1..4),
        haystack in prop::collection::vec(sym(), 0..200),
    ) {
        let lit_refs: Vec<&[u8]> = literals.iter().map(Vec::as_slice).collect();
        let matcher = GpuLiteralSet::compile(&lit_refs);
        let backend = CpuRefBackend;

        let dispatched = matcher
            .scan_all(&backend, &haystack)
            .expect("scan_all auto-resizes and completes");
        let oracle = matcher.reference_scan(&haystack);

        prop_assert_eq!(
            sorted_triples(&dispatched),
            sorted_triples(&oracle),
            "scan_all must equal reference_scan; literals={:?} haystack={:?}",
            literals,
            haystack
        );
    }
}

/// By-construction rigor independent of the reference oracle: distinct
/// upper-alphabet needles planted in lowercase filler that shares no byte with any
/// needle, so the ONLY matches are the planted ones at their known offsets.
#[test]
fn planted_distinct_needles_report_exact_offsets() {
    // needle -> (its bytes). Distinct alphabets so no needle is a substring of the
    // filler or of another needle.
    let literals: &[&[u8]] = &[b"XY", b"QRS", b"ZZ"];
    let matcher = GpuLiteralSet::compile(literals);
    let backend = CpuRefBackend;

    // "..XY...QRS..ZZ." (filler is dots (absent from every needle)).
    //  0123456789012345
    let haystack = b"..XY...QRS..ZZ.";
    let got = sorted_triples(
        &matcher
            .scan_all(&backend, haystack)
            .expect("scan_all completes"),
    );

    // XY (pid 0) at 2..4; QRS (pid 1) at 7..10; ZZ (pid 2) at 12..14.
    let expected = vec![(0u32, 2u32, 4u32), (1, 7, 10), (2, 12, 14)];
    assert_eq!(
        got, expected,
        "planted needles must be found at exactly their offsets: got {got:?}"
    );

    // Cross-check against the reference oracle too.
    assert_eq!(
        got,
        sorted_triples(&matcher.reference_scan(haystack)),
        "planted-needle result must also equal the reference oracle"
    );
}
