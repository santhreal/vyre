//! Leftmost-longest ("maximal munch") extraction for BOUNDED-REPEAT `{n,m}`
//! patterns (the fix for vyre BACKLOG items 18/27).
//!
//! Root cause (now fixed in `regex_compile::build_repetition`): a `{n,m}`
//! fragment recorded `match_len = n` (the MINIMUM, since `total_len` only
//! accumulated the mandatory copies), so `max_pattern_len`: the replay-window
//! cap, was the minimum. The anchored windowed walk therefore capped at `n`
//! and never visited the longer accepts (`a{2,4}` surfaced only length-2), while
//! the uncapped unanchored single-pass over-reported every admissible end. The
//! lowering now records the MAX length, so the window covers `n..=m`, and
//! [`AnchoredWindowValidator::validate_candidate_leftmost_longest`] collapses the
//! per-origin accept run to the single longest match, one finding covering the
//! whole token, which is exactly the scanner-correct semantics.
//!
//! Every assertion is an EXACT `(pattern_id, start, end)` set (never
//! `!is_empty()`), with a by-construction oracle (the token is planted, so its
//! start and maximal end are known a priori) plus a direct contrast against the
//! raw all-ends fan-out (`validate_candidate`). A NEW non-colliding file that
//! touches no foreign-dirty source.
#![cfg(feature = "matching-regex")]

use proptest::prelude::*;
use vyre_foundation::match_result::Match;
use vyre_libs::scan::{build_regex_dfa_pipeline, AnchoredWindowValidator};

/// Build the anchored DFA for one pattern and return its validator-ready DFA.
fn dfa_for(pattern: &str) -> vyre_libs::scan::regex_dfa::RegexDfaPipeline {
    build_regex_dfa_pipeline(&[pattern], 4096, 16_384)
        .unwrap_or_else(|e| panic!("pattern {pattern:?} must compile to an anchored DFA: {e:?}"))
}

/// `(pattern_id, start, end)` triples of a match slice, in canonical order.
fn triples(matches: &[Match]) -> Vec<(u32, u32, u32)> {
    let mut v: Vec<(u32, u32, u32)> = matches
        .iter()
        .map(|m| (m.pattern_id, m.start, m.end))
        .collect();
    v.sort_unstable();
    v
}

#[test]
fn literal_repeat_collapses_to_single_longest_match() {
    // `a{2,4}` over "aaaa" seeded at origin 0.
    let pipeline = dfa_for("a{2,4}");
    let validator = AnchoredWindowValidator::new(&pipeline.dfa);
    let haystack = b"aaaa";

    // Raw fan-out: one hit per admissible length 2, 3, 4.
    let raw = validator.validate_candidates(haystack, &[0]);
    assert_eq!(
        triples(&raw),
        vec![(0, 0, 2), (0, 0, 3), (0, 0, 4)],
        "raw fan-out must surface every admissible {{2,4}} length at origin 0"
    );

    // Leftmost-longest: exactly the longest match (the whole 4-'a' run).
    let ll = validator.validate_candidates_leftmost_longest(haystack, &[0]);
    assert_eq!(
        triples(&ll),
        vec![(0, 0, 4)],
        "leftmost-longest must collapse the run to its single maximal match"
    );
}

#[test]
fn class_repeat_takes_maximal_body_and_stops_at_terminator() {
    // `k[0-9]{2,4}`: prefix `k`, then 2..4 digits. max_pattern_len == 1 + 4 == 5.
    let pipeline = dfa_for("k[0-9]{2,4}");
    assert_eq!(
        pipeline.dfa.max_pattern_len, 5,
        "window must size to the MAX repetition (k + 4 digits)"
    );
    let validator = AnchoredWindowValidator::new(&pipeline.dfa);

    // Exactly 2 digits then a non-digit: single match `k12` (end 3).
    assert_eq!(
        triples(&validator.validate_candidates_leftmost_longest(b"k12x", &[0])),
        vec![(0, 0, 3)],
        "a 2-digit body terminated by a non-digit is the whole (minimal-length) token"
    );

    // 4 digits (== max): single match `k1234` (end 5).
    assert_eq!(
        triples(&validator.validate_candidates_leftmost_longest(b"k1234", &[0])),
        vec![(0, 0, 5)],
        "a 4-digit body is consumed whole"
    );

    // 6 digits (> max): maximal munch takes only 4 → `k1234` (end 5), NOT the
    // whole 6-digit run. The trailing digits are not part of a `{2,4}` match.
    assert_eq!(
        triples(&validator.validate_candidates_leftmost_longest(b"k123456", &[0])),
        vec![(0, 0, 5)],
        "maximal munch caps the body at m == 4 digits even when more digits follow"
    );

    // 1 digit (< min): no match at all.
    assert!(
        validator
            .validate_candidates_leftmost_longest(b"k1x", &[0])
            .is_empty(),
        "a 1-digit body is below the {{2,4}} minimum and must not match"
    );
}

#[test]
fn fixed_repeat_is_unchanged_by_leftmost_longest() {
    // A fixed `{n}` pattern accepts at exactly one length, so leftmost-longest
    // and the raw fan-out must agree (the fix must not perturb fixed patterns).
    let pipeline = dfa_for("ghp_[A-Za-z0-9]{4}");
    let validator = AnchoredWindowValidator::new(&pipeline.dfa);
    let haystack = b"ghp_aB3d";

    let raw = triples(&validator.validate_candidates(haystack, &[0]));
    let ll = triples(&validator.validate_candidates_leftmost_longest(haystack, &[0]));
    assert_eq!(
        raw,
        vec![(0, 0, 8)],
        "fixed token accepts once at its full length"
    );
    assert_eq!(
        ll, raw,
        "leftmost-longest must equal the fan-out for fixed patterns"
    );
}

#[test]
fn two_variable_tokens_each_collapse_at_their_own_origin() {
    // Two `{2,4}` tokens; feed each token's start origin. Each yields exactly one
    // whole-token match (no cross-token bleed, no per-length duplicates).
    let pipeline = dfa_for("v[0-9]{2,4}");
    let validator = AnchoredWindowValidator::new(&pipeline.dfa);
    //                0123456789012
    let haystack = b"v123 xx v4567";
    // Token A: origin 0, `v123` (3 digits) -> end 4.
    // Token B: origin 8, `v4567` (4 digits) -> end 13.
    assert_eq!(
        triples(&validator.validate_candidates_leftmost_longest(haystack, &[0, 8])),
        vec![(0, 0, 4), (0, 8, 13)],
        "each variable token collapses to one maximal match at its own origin"
    );
}

#[test]
fn open_ended_repeat_window_is_documented_limitation() {
    // Open-ended repeats (`+`, `*`, `{n,}`, i.e. `max = None`) have NO finite
    // maximum length, so the windowed-replay architecture (which caps each
    // origin's walk at `max_pattern_len`) fundamentally cannot cover them 
    // `build_repetition` records only the MIN length for the open case. This
    // test PINS that limitation so a future open-ended fix (route unbounded
    // repeats through the uncapped single-pass path, or reject them at
    // DFA-window compile (see BACKLOG) trips here and updates the expectation).
    // `k[0-9]+` -> min length 1 (prefix) + 1 (one mandatory digit) == 2.
    let plus = dfa_for("k[0-9]+");
    assert_eq!(
        plus.dfa.max_pattern_len, 2,
        "open-ended `+` records only the MIN window (prefix + one repeat); \
         longer matches under-scan on the windowed path, a documented limitation"
    );
    // `{n,}` (lower bound only) is the same open case: `k[0-9]{3,}` -> 1 + 3 == 4.
    let lower_bounded = dfa_for("k[0-9]{3,}");
    assert_eq!(
        lower_bounded.dfa.max_pattern_len, 4,
        "open-ended `{{3,}}` records only the MIN window (prefix + 3 repeats)"
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(800))]

    /// Plant `q_[a-z]{k}` (k in the pattern's `[2,6]` range) at a chosen origin in
    /// filler that shares no byte with the token (digits + spaces cannot start the
    /// letter prefix or extend the lowercase body), terminated so the run cannot
    /// grow. The leftmost-longest match at the token origin is therefore known by
    /// construction: exactly one `(0, origin, origin + 2 + k)`.
    #[test]
    fn planted_bounded_token_yields_single_maximal_match(
        k in 2usize..=6,
        pre in 0usize..6,
        body_seed in "[a-z]{6}",
    ) {
        let pattern = "q_[a-z]{2,6}";
        let pipeline = dfa_for(pattern);
        let validator = AnchoredWindowValidator::new(&pipeline.dfa);

        let body: String = body_seed.chars().take(k).collect();
        let mut haystack = String::new();
        haystack.push_str(&" ".repeat(pre)); // filler: cannot start `q`
        let origin = haystack.len() as u32;
        haystack.push_str("q_");
        haystack.push_str(&body);
        haystack.push('9'); // digit terminator: not in [a-z], cannot extend body

        let expected_end = origin + 2 + k as u32; // q_ (2) + k body bytes
        prop_assert_eq!(
            triples(&validator.validate_candidates_leftmost_longest(haystack.as_bytes(), &[origin])),
            vec![(0, origin, expected_end)],
            "planted token {:?} at origin {} must yield one maximal match ending at {}",
            haystack, origin, expected_end
        );
    }

    /// Negative twin: a body of length 1 (below the `{2,6}` minimum) at the token
    /// origin yields NO match, regardless of filler.
    #[test]
    fn planted_below_minimum_body_yields_no_match(
        pre in 0usize..6,
        c in "[a-z]",
    ) {
        let pipeline = dfa_for("q_[a-z]{2,6}");
        let validator = AnchoredWindowValidator::new(&pipeline.dfa);

        let mut haystack = String::new();
        haystack.push_str(&" ".repeat(pre));
        let origin = haystack.len() as u32;
        haystack.push_str("q_");
        haystack.push_str(&c);
        haystack.push('9'); // terminator so only 1 body byte is available

        prop_assert!(
            validator
                .validate_candidates_leftmost_longest(haystack.as_bytes(), &[origin])
                .is_empty(),
            "a single-byte body is below the {{2,6}} minimum; token {:?} must not match",
            haystack
        );
    }
}
