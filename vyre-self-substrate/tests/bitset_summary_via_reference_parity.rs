//! End-to-end parity for `data::bitset_summary::{per_word_popcount_via, total_set_bits_via}` through
//! the shared faithful [`common::ReferenceEvalDispatcher`].
//!
//! Closes a mock-dispatcher-coherence gap (see BACKLOG `SWEEP-self-substrate-mock-dispatcher-coherence`):
//! the per-word popcount IR is not run through a faithful dispatch boundary by any
//! `vyre-primitives/tests/*` file. This is the FIRST-EVER execution of the popcount kernel through a
//! dispatch boundary that models the real backend.
//!
//! Contract (audited CLEAN): per_word_popcount binds input RO(0) + out RW(1) = 2 IC, decode
//! outputs[0]. Exact integer arithmetic → BIT-EXACT (no tolerance), compared against a fully
//! independent inline `u32::count_ones` oracle.
#![cfg(feature = "cpu-parity")]

use vyre_self_substrate::data::bitset_summary::{per_word_popcount_via, total_set_bits_via};

mod common;
use common::ReferenceEvalDispatcher;

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

#[test]
fn popcount_via_matches_inline_count_ones_oracle() {
    let dispatcher = ReferenceEvalDispatcher;
    let mut state = 0x90_9C_00_01u32;
    let mut saw_full = 0u32;
    let mut saw_empty = 0u32;
    for case in 0..400u32 {
        let words = 1 + (case % 12) as usize;
        // Mix in all-zero and all-one words so the popcount spans 0..=32 per word.
        let input: Vec<u32> = (0..words)
            .map(|_| match xorshift(&mut state) % 6 {
                0 => 0,
                1 => u32::MAX,
                _ => xorshift(&mut state),
            })
            .collect();

        let want: Vec<u32> = input.iter().map(|w| w.count_ones()).collect();
        let got = per_word_popcount_via(&dispatcher, &input)
            .expect("per_word_popcount_via must dispatch");
        assert_eq!(
            got, want,
            "case {case}: per-word popcount must match count_ones; input={input:?}"
        );

        let want_total: u64 = want.iter().map(|&c| u64::from(c)).sum();
        assert_eq!(
            total_set_bits_via(&dispatcher, &input).unwrap(),
            want_total,
            "case {case}: total set bits must equal the summed popcount"
        );

        if input.iter().any(|&w| w == u32::MAX) {
            saw_full += 1;
        }
        if input.iter().any(|&w| w == 0) {
            saw_empty += 1;
        }
    }
    assert!(
        saw_full > 100 && saw_empty > 100,
        "sweep must exercise full (32-bit) and empty (0-bit) words: full={saw_full} empty={saw_empty}"
    );
}

#[test]
fn popcount_via_hand_checked_cases() {
    let d = ReferenceEvalDispatcher;
    assert_eq!(
        per_word_popcount_via(&d, &[0b1111, 0b101, 0]).unwrap(),
        vec![4, 2, 0],
        "popcount per word"
    );
    assert_eq!(
        per_word_popcount_via(&d, &[u32::MAX, u32::MAX]).unwrap(),
        vec![32, 32],
        "all-ones words each count 32"
    );
    assert_eq!(
        total_set_bits_via(&d, &[u32::MAX, 0b111, 0]).unwrap(),
        35,
        "total = 32 + 3 + 0"
    );
}
