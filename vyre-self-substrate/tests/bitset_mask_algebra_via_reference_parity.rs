//! End-to-end parity for `data::bitset_mask_algebra::mask_*_via` through the shared faithful
//! [`common::ReferenceEvalDispatcher`], across every mask operation the consumer exposes:
//! and / or / xor / not (word-parallel), equal / subset-of (binary predicates), contains / test_bit
//! (bit queries), and set_bit / clear_bit (single-bit rewrites).
//!
//! Closes a mock-dispatcher-coherence gap (see BACKLOG `SWEEP-self-substrate-mock-dispatcher-coherence`):
//! these bitset IRs are not run through a faithful dispatch boundary by any `vyre-primitives/tests/*`
//! file, and the consumer's only coverage is its own in-file dispatcher. This is the FIRST-EVER
//! execution of the mask kernels through a dispatch boundary that models the real backend.
//!
//! Contracts (audited CLEAN): mask_binary/contains bind 3 IC (two RO operands + out RW), not/test_bit
//! bind 2 IC (input RO + out RW). All decode outputs[0] = the sole writable buffer. Every op is exact
//! bitwise arithmetic → BIT-EXACT (no tolerance), compared against the authoritative `reference_mask_*`
//! CPU oracles.
#![cfg(feature = "cpu-parity")]

use vyre_self_substrate::data::bitset_mask_algebra::{
    mask_and_via, mask_clear_bit_via, mask_contains_via, mask_equal_via, mask_not_via, mask_or_via,
    mask_set_bit_via, mask_subset_of_via, mask_test_bit_via, mask_xor_via, reference_mask_and,
    reference_mask_clear_bit, reference_mask_contains, reference_mask_equal, reference_mask_not,
    reference_mask_or, reference_mask_set_bit, reference_mask_subset_of, reference_mask_test_bit,
    reference_mask_xor,
};

mod common;
use common::ReferenceEvalDispatcher;

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

#[test]
fn word_parallel_and_or_xor_not_via_match_cpu_ref() {
    let dispatcher = ReferenceEvalDispatcher;
    let mut state = 0xB1_75_E7_01u32;
    for case in 0..400u32 {
        let words = 1 + (case % 8) as usize;
        let lhs: Vec<u32> = (0..words).map(|_| xorshift(&mut state)).collect();
        let rhs: Vec<u32> = (0..words).map(|_| xorshift(&mut state)).collect();

        assert_eq!(
            mask_and_via(&dispatcher, &lhs, &rhs).unwrap(),
            reference_mask_and(&lhs, &rhs),
            "case {case}: AND; lhs={lhs:?} rhs={rhs:?}"
        );
        assert_eq!(
            mask_or_via(&dispatcher, &lhs, &rhs).unwrap(),
            reference_mask_or(&lhs, &rhs),
            "case {case}: OR"
        );
        assert_eq!(
            mask_xor_via(&dispatcher, &lhs, &rhs).unwrap(),
            reference_mask_xor(&lhs, &rhs),
            "case {case}: XOR"
        );
        assert_eq!(
            mask_not_via(&dispatcher, &lhs).unwrap(),
            reference_mask_not(&lhs),
            "case {case}: NOT"
        );
        // Cross-check AND/OR/XOR against a fully independent inline computation too.
        let inline_and: Vec<u32> = lhs.iter().zip(&rhs).map(|(a, b)| a & b).collect();
        assert_eq!(
            mask_and_via(&dispatcher, &lhs, &rhs).unwrap(),
            inline_and,
            "case {case}: AND vs inline"
        );
    }
}

#[test]
fn binary_predicates_equal_subset_via_match_cpu_ref() {
    let dispatcher = ReferenceEvalDispatcher;
    let mut state = 0xEC_5B_00_01u32;
    let mut saw_equal = 0u32;
    let mut saw_subset = 0u32;
    for case in 0..400u32 {
        let words = 1 + (case % 6) as usize;
        let lhs: Vec<u32> = (0..words).map(|_| xorshift(&mut state)).collect();
        // Bias toward related masks so `equal` and `subset` fire: half the time rhs is a superset
        // of lhs (lhs | extra), sometimes exactly equal.
        let rhs: Vec<u32> = match case % 3 {
            0 => lhs.clone(),                                             // equal
            1 => lhs.iter().map(|&a| a | xorshift(&mut state)).collect(), // superset of lhs
            _ => (0..words).map(|_| xorshift(&mut state)).collect(),      // arbitrary
        };

        assert_eq!(
            mask_equal_via(&dispatcher, &lhs, &rhs).unwrap(),
            reference_mask_equal(&lhs, &rhs),
            "case {case}: EQUAL; lhs={lhs:?} rhs={rhs:?}"
        );
        assert_eq!(
            mask_subset_of_via(&dispatcher, &lhs, &rhs).unwrap(),
            reference_mask_subset_of(&lhs, &rhs),
            "case {case}: SUBSET_OF"
        );
        if reference_mask_equal(&lhs, &rhs) {
            saw_equal += 1;
        }
        if reference_mask_subset_of(&lhs, &rhs) {
            saw_subset += 1;
        }
    }
    assert!(
        saw_equal > 50 && saw_subset > 100,
        "predicate sweep must exercise real equal + subset hits: equal={saw_equal} subset={saw_subset}"
    );
}

#[test]
fn bit_queries_and_single_bit_rewrites_via_match_cpu_ref() {
    let dispatcher = ReferenceEvalDispatcher;
    let mut state = 0x517_00_01u32;
    let mut saw_present = 0u32;
    let mut saw_absent = 0u32;
    for case in 0..400u32 {
        let words = 1 + (case % 6) as usize;
        let input: Vec<u32> = (0..words).map(|_| xorshift(&mut state)).collect();
        let total_bits = (words * 32) as u32;
        let bit_idx = xorshift(&mut state) % total_bits;

        let present = reference_mask_contains(&input, bit_idx);
        assert_eq!(
            mask_contains_via(&dispatcher, &input, bit_idx).unwrap(),
            present,
            "case {case}: CONTAINS bit {bit_idx}; input={input:?}"
        );
        assert_eq!(
            mask_test_bit_via(&dispatcher, &input, bit_idx).unwrap(),
            reference_mask_test_bit(&input, bit_idx),
            "case {case}: TEST_BIT {bit_idx}"
        );
        assert_eq!(
            mask_set_bit_via(&dispatcher, &input, bit_idx).unwrap(),
            reference_mask_set_bit(&input, bit_idx),
            "case {case}: SET_BIT {bit_idx}"
        );
        assert_eq!(
            mask_clear_bit_via(&dispatcher, &input, bit_idx).unwrap(),
            reference_mask_clear_bit(&input, bit_idx),
            "case {case}: CLEAR_BIT {bit_idx}"
        );
        // set then test must be present; clear then test must be absent (round-trip invariant).
        let set = mask_set_bit_via(&dispatcher, &input, bit_idx).unwrap();
        assert!(
            reference_mask_test_bit(&set, bit_idx),
            "case {case}: bit must be present after set_bit"
        );
        let cleared = mask_clear_bit_via(&dispatcher, &input, bit_idx).unwrap();
        assert!(
            !reference_mask_test_bit(&cleared, bit_idx),
            "case {case}: bit must be absent after clear_bit"
        );
        if present {
            saw_present += 1;
        } else {
            saw_absent += 1;
        }
    }
    assert!(
        saw_present > 100 && saw_absent > 100,
        "bit-query sweep must exercise both present and absent bits: present={saw_present} absent={saw_absent}"
    );
}

#[test]
fn bitset_mask_via_hand_checked_cases() {
    let d = ReferenceEvalDispatcher;
    let a = vec![0b1100u32, 0xFF00_00FF];
    let b = vec![0b1010u32, 0x0F0F_0F0F];

    assert_eq!(
        mask_and_via(&d, &a, &b).unwrap(),
        vec![0b1000, 0x0F00_000F],
        "AND"
    );
    assert_eq!(
        mask_or_via(&d, &a, &b).unwrap(),
        vec![0b1110, 0xFF0F_0FFF],
        "OR"
    );
    assert_eq!(
        mask_xor_via(&d, &a, &b).unwrap(),
        vec![0b0110, 0xF00F_0FF0],
        "XOR"
    );
    assert_eq!(
        mask_not_via(&d, &a).unwrap(),
        vec![!0b1100u32, !0xFF00_00FFu32],
        "NOT"
    );

    assert!(mask_equal_via(&d, &a, &a).unwrap(), "a == a");
    assert!(!mask_equal_via(&d, &a, &b).unwrap(), "a != b");

    // bit 3 (value 0b1000) is set in a[0]=0b1100? 0b1100 has bits 2,3 → bit 3 set.
    assert!(mask_test_bit_via(&d, &a, 3).unwrap(), "bit 3 set in 0b1100");
    assert!(
        !mask_test_bit_via(&d, &a, 0).unwrap(),
        "bit 0 clear in 0b1100"
    );

    // set bit 0 in a → a[0] becomes 0b1101.
    let set = mask_set_bit_via(&d, &a, 0).unwrap();
    assert_eq!(set[0], 0b1101, "set_bit(0) turns on bit 0");
    // clear bit 2 in a → a[0] becomes 0b1000.
    let cleared = mask_clear_bit_via(&d, &a, 2).unwrap();
    assert_eq!(cleared[0], 0b1000, "clear_bit(2) turns off bit 2");
}
