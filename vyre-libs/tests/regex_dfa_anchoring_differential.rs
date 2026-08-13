//! Property coverage for the ANCHORING contract of the regex DFA: the anchored
//! build (`build_regex_dfa_pipeline`) must accept a pattern ONLY when it occurs at
//! byte 0, while the unanchored build (`build_regex_dfa_unanchored`) finds it at any
//! offset. This is the invariant the megakernel-fallback port depends on (a secret
//! is rarely at byte 0), and it is the deliberate mirror of the leftmost-longest
//! suite: same fixed-structure token patterns, orthogonal property (position, not
//! length). Green today (fixed `{n}` repeats; range `{n,m}` excluded per item 18/27).
#![cfg(feature = "matching-regex")]

use proptest::prelude::*;
use vyre_libs::scan::regex_dfa::{build_regex_dfa_pipeline, build_regex_dfa_unanchored};

/// Single-pass accept-end walk over the public `CompiledDfa` fields (mirrors the
/// production scan). The DFA type is never named, inferred field access keeps this
/// free of `vyre-primitives` feature coupling.
macro_rules! accept_ends {
    ($pipeline:expr, $haystack:expr) => {{
        let dfa = &$pipeline.dfa;
        let mut state = 0u32;
        let mut ends = Vec::new();
        for (i, &byte) in $haystack.iter().enumerate() {
            state = dfa.transitions[state as usize * 256 + byte as usize];
            if dfa.accept[state as usize] != 0 {
                ends.push(i + 1);
            }
        }
        ends
    }};
}

fn anchored_ends(pattern: &str, haystack: &[u8]) -> Vec<usize> {
    let pipeline =
        build_regex_dfa_pipeline(&[pattern], 1024, 1 << 16).expect("anchored pattern compiles");
    accept_ends!(pipeline, haystack)
}

fn unanchored_ends(pattern: &str, haystack: &[u8]) -> Vec<usize> {
    let pipeline =
        build_regex_dfa_unanchored(&[pattern], 1024, 1 << 16).expect("unanchored pattern compiles");
    accept_ends!(pipeline, haystack)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1200))]

    /// Plant a fixed-structure token at a chosen offset in dot-filler (dots share no
    /// byte with the letter/`_`/alnum token, so the token is the only occurrence).
    /// The anchored DFA matches iff the token sits at byte 0; the unanchored DFA
    /// always finds it at its planted end. Both facts asserted as EXACT end sets.
    #[test]
    fn anchored_matches_only_at_offset_zero(
        prefix_letters in "[a-z]{2,4}",
        n in 8usize..32,
        body in "[A-Za-z0-9]{8,32}",
        lead in 0usize..6,
    ) {
        let body: String = body.chars().cycle().take(n).collect();
        let pattern = format!("{prefix_letters}_[A-Za-z0-9]{{{n}}}");
        let token_len = prefix_letters.len() + 1 + n;

        let mut haystack = String::new();
        haystack.push_str(&".".repeat(lead));
        haystack.push_str(&prefix_letters);
        haystack.push('_');
        haystack.push_str(&body);
        haystack.push('.'); // trailing terminator so the run can't extend
        let bytes = haystack.as_bytes();

        // Unanchored: always finds the one token, ending at lead + token_len.
        prop_assert_eq!(
            unanchored_ends(&pattern, bytes),
            vec![lead + token_len],
            "unanchored must find the planted token; pattern={:?} hay={:?}",
            pattern, haystack
        );

        // Anchored: matches iff the token starts at byte 0 (lead == 0).
        let expected_anchored = if lead == 0 { vec![token_len] } else { Vec::new() };
        prop_assert_eq!(
            anchored_ends(&pattern, bytes),
            expected_anchored,
            "anchored must accept iff token is at offset 0 (lead={}); pattern={:?} hay={:?}",
            lead, pattern, haystack
        );
    }
}

/// Concrete anchoring regressions (exact end sets).
#[test]
fn anchored_vs_unanchored_fixed_cases() {
    // Token at byte 0: both accept, anchored at the single token end.
    assert_eq!(anchored_ends("ab_[A-Za-z0-9]{4}", b"ab_WXYZ.."), vec![7]);
    assert_eq!(unanchored_ends("ab_[A-Za-z0-9]{4}", b"ab_WXYZ.."), vec![7]);

    // Token at byte 2: anchored rejects, unanchored finds it (end 9).
    assert!(anchored_ends("ab_[A-Za-z0-9]{4}", b"..ab_WXYZ").is_empty());
    assert_eq!(unanchored_ends("ab_[A-Za-z0-9]{4}", b"..ab_WXYZ"), vec![9]);

    // No token at all: both empty.
    assert!(anchored_ends("ab_[A-Za-z0-9]{4}", b"nothing here").is_empty());
    assert!(unanchored_ends("ab_[A-Za-z0-9]{4}", b"nothing here").is_empty());
}
