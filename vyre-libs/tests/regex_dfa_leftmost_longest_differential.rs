//! Differential + property coverage for the unanchored regex DFA's single-pass
//! accept semantics on FIXED-structure token patterns (`<prefix>[class]{n}`: the
//! shape of every real secret detector: `ghp_[A-Za-z0-9]{36}`, `AKIA[A-Z0-9]{16}`,
//! …).
//!
//! The existing `regex_dfa` inline tests are hand-picked single cases. This suite
//! adds the missing PROPERTY + DIFFERENTIAL layer the Testing Contract requires:
//! a by-construction oracle (the token is planted, so its exact end offset is known
//! a priori) driven over thousands of proptest-generated pattern/haystack pairs,
//! plus a data-driven table of exact-value regressions. It asserts the EXACT end
//! set (leftmost-longest, one end per occurrence), never `!is_empty()`, so it fails
//! on both a missed hit (false negative) and a spurious/duplicated hit.
//!
//! Scope note (deliberate): fixed-count `{n}` repeats are exercised because they are
//! recall-correct today (verified: the 4 `ghp_…{36}` parity cases pass). BOUNDED
//! RANGE `{n,m}` repeats are intentionally EXCLUDED, they are a known, tracked bug
//! (the single-pass DFA over-reports every admissible end `n..=m` instead of the one
//! leftmost-longest end; vyre `BACKLOG.md` item 18/27). Locking the range contract
//! belongs with that fix, not here, so this suite stays green and pins the
//! correct-today semantics as a regression floor beneath the range work.
#![cfg(feature = "matching-regex")]

use proptest::prelude::*;
use vyre_libs::scan::regex_dfa::build_regex_dfa_unanchored;

/// Build the unanchored DFA and run the production single-pass scan (mirrors
/// `regex_dfa.rs`): follow one transition per byte off the public transition table;
/// every byte whose landed state accepts closes a match ending at `i + 1`. Reusing
/// the exact production walk (not a re-derivation) over the public `CompiledDfa`
/// fields makes this a faithful check of the shipped DFA. The `CompiledDfa` type is
/// never named, field access through the inferred `pipeline.dfa` keeps this file
/// free of any `vyre-primitives` feature-visibility coupling.
fn ends_for(pattern: &str, haystack: &[u8]) -> Vec<usize> {
    let pipeline = build_regex_dfa_unanchored(&[pattern], 1024, 1 << 16)
        .unwrap_or_else(|e| panic!("pattern {pattern:?} must compile: {e:?}"));
    let dfa = &pipeline.dfa;
    let mut state = 0u32;
    let mut ends = Vec::new();
    for (i, &byte) in haystack.iter().enumerate() {
        state = dfa.transitions[state as usize * 256 + byte as usize];
        if dfa.accept[state as usize] != 0 {
            ends.push(i + 1);
        }
    }
    ends
}

/// Exact-value regressions for fixed-structure token patterns: prefix at a known
/// offset, one end per planted occurrence, at `start + prefix_len + n`.
#[test]
fn fixed_repeat_token_patterns_report_exact_leftmost_longest_ends() {
    // (pattern, haystack, expected end offsets) (every end hand-derived).
    let cases: &[(&str, &[u8], &[usize])] = &[
        // Single token mid-string. "x ghp_aB3d y": ghp_ at 2, body aB3d (4) ends at 9 -> end 10.
        ("ghp_[A-Za-z0-9]{4}", b"x ghp_aB3d y", &[10]),
        // Token at byte 0.
        ("ghp_[A-Za-z0-9]{4}", b"ghp_aB3dxx", &[8]),
        // Token flush at end of input (no trailing byte).
        ("sk-[A-Za-z0-9]{6}", b"key=sk-Ab3Cd9", &[13]),
        // Prefix chars ALSO appear in the body (the g/h/p overlap regression): the
        // `_` terminator means the prefix can only anchor once -> exactly one end.
        (
            "ghp_[A-Za-z0-9]{6}",
            b"= ghp_ghp123 ",
            &[12],
        ),
        // Two non-overlapping occurrences -> two ends.
        (
            "AKIA[A-Z0-9]{4}",
            b"AKIAWXYZ and AKIA1234!",
            &[8, 21],
        ),
        // Body one char too short -> the `{n}` never completes -> no match.
        ("ghp_[A-Za-z0-9]{5}", b"ghp_aB3d ", &[]),
        // Non-class byte inside the body window -> no match.
        ("ghp_[A-Za-z0-9]{5}", b"ghp_aB.d9", &[]),
        // Hex-class token (distinct alphabet from the alnum cases).
        ("v1_[a-f0-9]{8}", b"tok: v1_0a1b2c3d;", &[16]),
    ];
    for (pattern, haystack, expected) in cases {
        assert_eq!(
            ends_for(pattern, haystack),
            expected.to_vec(),
            "pattern {pattern:?} over {:?} must report EXACTLY the leftmost-longest \
             end set {expected:?}",
            String::from_utf8_lossy(haystack),
        );
    }
}

proptest! {
    // A planted fixed-structure token in filler that CANNOT re-form the pattern:
    // filler is spaces (not in the alnum class, cannot start the letter-prefix) and
    // the prefix ends in `_` (absent from the alnum body), so the pattern anchors at
    // exactly one offset. The single correct end is therefore known by construction.
    #![proptest_config(ProptestConfig::with_cases(1500))]

    #[test]
    fn planted_fixed_token_yields_single_end_by_construction(
        prefix_letters in "[a-z]{2,4}",
        n in 8usize..40,
        body in "[A-Za-z0-9]{8,40}",
        pre in 0usize..8,
        post in 0usize..8,
    ) {
        // Force the body to be exactly n class bytes.
        let body: String = body.chars().cycle().take(n).collect();
        let pattern = format!("{prefix_letters}_[A-Za-z0-9]{{{n}}}");

        let mut haystack = String::new();
        haystack.push_str(&" ".repeat(pre));
        let prefix_start = haystack.len();
        haystack.push_str(&prefix_letters);
        haystack.push('_');
        haystack.push_str(&body);
        haystack.push_str(&" ".repeat(post));

        // The prefix ("<letters>_") occurs only at prefix_start; the body is n class
        // bytes; so the one match ends at start + prefix_len + 1 (the `_`) + n.
        let expected_end = prefix_start + prefix_letters.len() + 1 + n;

        prop_assert_eq!(
            ends_for(&pattern, haystack.as_bytes()),
            vec![expected_end],
            "planted token {:?} in {:?} must report the one leftmost-longest end {}",
            pattern,
            haystack,
            expected_end
        );
    }

    // Truncating the planted body below n makes the fixed count unsatisfiable, so the
    // scan must report NO end (the negative twin of the property above).
    #[test]
    fn planted_token_with_short_body_yields_no_end(
        prefix_letters in "[a-z]{2,4}",
        n in 10usize..40,
        deficit in 1usize..6,
        body in "[A-Za-z0-9]{8,40}",
    ) {
        let short_len = n - deficit;
        let body: String = body.chars().cycle().take(short_len).collect();
        let pattern = format!("{prefix_letters}_[A-Za-z0-9]{{{n}}}");
        // Terminate the body run with a space so no extra class bytes can extend it.
        let haystack = format!("  {prefix_letters}_{body} ");

        prop_assert!(
            ends_for(&pattern, haystack.as_bytes()).is_empty(),
            "pattern {:?} needs {} class bytes but only {} follow the prefix in {:?} \
             -> the fixed count is unsatisfiable, so there must be NO match",
            pattern, n, short_len, haystack
        );
    }
}
