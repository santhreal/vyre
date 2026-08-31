//! Failure-oriented adversarial tests for matching primitives.
//!
//! Focus: hostile boundaries, overflow, invalid offsets, property invariants.
#![cfg(feature = "pattern")]

use vyre_libs::pattern::{
    dfa_compile, dfa_compile_with_budget, DfaCompileError, BRACKET_KIND_CLOSE, BRACKET_KIND_OPEN,
    BRACKET_KIND_OTHER, BRACKET_MATCH_NONE,
};
use vyre_reference::composition_witness::bracket_match_witness as reference_bracket_match;

#[test]
fn bracket_match_reference_empty_inputs() {
    let cases: [(Vec<u32>, u32, Vec<u32>); 3] = [
        (vec![], 0, vec![]),
        (vec![], 1, vec![]),
        (vec![], 100, vec![]),
    ];
    for (kinds, max_depth, expected) in cases {
        let got = reference_bracket_match(&kinds, max_depth);
        assert_eq!(got, expected, "empty kinds must yield empty output");
    }
}

#[test]
fn bracket_match_reference_depth_zero_rejects_all_opens() {
    let kinds = vec![BRACKET_KIND_OPEN, BRACKET_KIND_OPEN, BRACKET_KIND_CLOSE];
    let got = reference_bracket_match(&kinds, 0);
    assert_eq!(
        got,
        vec![BRACKET_MATCH_NONE, BRACKET_MATCH_NONE, BRACKET_MATCH_NONE]
    );
}

#[test]
fn bracket_match_reference_all_closes_with_empty_stack() {
    let got = reference_bracket_match(&[BRACKET_KIND_CLOSE; 5], 10);
    assert_eq!(got, vec![BRACKET_MATCH_NONE; 5]);
}

#[test]
fn bracket_match_reference_overflow_length() {
    let n = 10_000usize;
    let kinds: Vec<u32> = (0..n)
        .map(|i| {
            if i % 2 == 0 {
                BRACKET_KIND_OPEN
            } else {
                BRACKET_KIND_CLOSE
            }
        })
        .collect();
    let got = reference_bracket_match(&kinds, n as u32);
    assert_eq!(got.len(), n);
    for i in (1..n).step_by(2) {
        assert_eq!(
            got[i],
            (i - 1) as u32,
            "close at {i} should match open at {}",
            i - 1
        );
    }
}

#[test]
fn bracket_match_reference_unbalanced_mixed() {
    let kinds = vec![
        BRACKET_KIND_OPEN,
        BRACKET_KIND_OTHER,
        BRACKET_KIND_OPEN,
        BRACKET_KIND_CLOSE,
        BRACKET_KIND_CLOSE,
        BRACKET_KIND_OTHER,
    ];
    let got = reference_bracket_match(&kinds, 10);
    assert_eq!(
        got,
        vec![4, BRACKET_MATCH_NONE, 3, 2, 0, BRACKET_MATCH_NONE]
    );
}

#[test]
fn dfa_compile_empty_patterns() {
    let dfa = dfa_compile(&[]);
    assert_eq!(dfa.state_count, 1);
    assert_eq!(dfa.transitions.len(), 256);
    assert!(dfa.accept.iter().all(|&a| a == 0));
}

#[test]
fn dfa_compile_single_byte_patterns() {
    let owned: Vec<Vec<u8>> = (0..=255).map(|b| vec![b]).collect();
    let patterns: Vec<&[u8]> = owned.iter().map(|v| v.as_slice()).collect();
    let dfa = dfa_compile(&patterns);
    for b in 0..=255u8 {
        let state = dfa.transitions[b as usize];
        assert!(dfa.accept[state as usize] != 0, "byte {b} must be accepted");
    }
}

#[test]
fn dfa_compile_budget_exhaustion() {
    let err = dfa_compile_with_budget(&[b"abcdefghijklmnopqrstuvwxyz"], 64).unwrap_err();
    assert!(matches!(
        err,
        DfaCompileError::TooLarge { .. } | DfaCompileError::TrieStateCapExceeded { .. }
    ));
}

#[test]
fn dfa_compile_zero_budget() {
    let err = dfa_compile_with_budget(&[b"a"], 0).unwrap_err();
    assert!(matches!(
        err,
        DfaCompileError::TrieStateCapExceeded { .. } | DfaCompileError::TooLarge { .. }
    ));
}

#[test]
fn dfa_compile_overlapping_patterns() {
    let patterns: [&[u8]; 4] = [b"he", b"she", b"his", b"hers"];
    let dfa = dfa_compile(&patterns);
    let matches: Vec<u32> = b"ushers"
        .iter()
        .scan(0u32, |state, &byte| {
            *state = dfa.transitions[(*state as usize) * 256 + (byte as usize)];
            Some(dfa.accept[*state as usize])
        })
        .filter(|&accept| accept != 0)
        .map(|accept| accept - 1)
        .collect();
    assert!(matches.contains(&1), "must accept `she`");
    assert!(
        matches.contains(&0) || matches.contains(&3),
        "must accept `he` or `hers`"
    );
}
