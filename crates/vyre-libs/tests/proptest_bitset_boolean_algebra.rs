//! Property gates for bitset boolean algebra across AND, OR, and NOT.
#![cfg(feature = "bitset")]

use proptest::prelude::*;
use vyre_reference::composition_witness::{
    bitset_and_witness, bitset_equal_witness, bitset_not_witness, bitset_or_witness,
};

fn same(left: &[u32], right: &[u32]) -> bool {
    bitset_equal_witness(left, right)
}

fn split_pairs(pairs: Vec<(u32, u32)>) -> (Vec<u32>, Vec<u32>) {
    pairs.into_iter().unzip()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10_000))]

    #[test]
    fn and_or_are_commutative(
        a in proptest::collection::vec(any::<u32>(), 0..=16),
        b in proptest::collection::vec(any::<u32>(), 0..=16),
    ) {
        prop_assert_eq!(bitset_and_witness(&a, &b), bitset_and_witness(&b, &a), "bitset AND must be commutative");
        prop_assert_eq!(bitset_or_witness(&a, &b), bitset_or_witness(&b, &a), "bitset OR must be commutative");
    }

    #[test]
    fn and_or_are_associative(
        a in proptest::collection::vec(any::<u32>(), 0..=8),
        b in proptest::collection::vec(any::<u32>(), 0..=8),
        c in proptest::collection::vec(any::<u32>(), 0..=8),
    ) {
        let and_ab_c = bitset_and_witness(&bitset_and_witness(&a, &b), &c);
        let and_a_bc = bitset_and_witness(&a, &bitset_and_witness(&b, &c));
        prop_assert_eq!(and_ab_c, and_a_bc, "bitset AND must be associative");

        let or_ab_c = bitset_or_witness(&bitset_or_witness(&a, &b), &c);
        let or_a_bc = bitset_or_witness(&a, &bitset_or_witness(&b, &c));
        prop_assert_eq!(or_ab_c, or_a_bc, "bitset OR must be associative");
    }

    #[test]
    fn and_or_are_idempotent(
        a in proptest::collection::vec(any::<u32>(), 0..=16),
    ) {
        prop_assert!(same(&a, &bitset_and_witness(&a, &a)), "a & a must equal a");
        prop_assert!(same(&a, &bitset_or_witness(&a, &a)), "a | a must equal a");
    }

    #[test]
    fn identities_and_annihilators_hold(
        a in proptest::collection::vec(any::<u32>(), 0..=16),
    ) {
        let zeros = vec![0u32; a.len()];
        let ones = vec![0xFFFF_FFFFu32; a.len()];

        prop_assert!(same(&a, &bitset_and_witness(&a, &ones)), "a & 1 must equal a");
        prop_assert!(same(&zeros, &bitset_and_witness(&a, &zeros)), "a & 0 must equal 0");
        prop_assert!(same(&a, &bitset_or_witness(&a, &zeros)), "a | 0 must equal a");
        prop_assert!(same(&ones, &bitset_or_witness(&a, &ones)), "a | 1 must equal 1");
    }

    #[test]
    fn distributive_laws_hold(
        a in proptest::collection::vec(any::<u32>(), 0..=8),
        b in proptest::collection::vec(any::<u32>(), 0..=8),
        c in proptest::collection::vec(any::<u32>(), 0..=8),
    ) {
        let and_over_or_left = bitset_and_witness(&bitset_or_witness(&a, &b), &c);
        let and_over_or_right = bitset_or_witness(&bitset_and_witness(&a, &c), &bitset_and_witness(&b, &c));
        prop_assert_eq!(and_over_or_left, and_over_or_right, "(a | b) & c must equal (a & c) | (b & c)");

        let or_over_and_left = bitset_or_witness(&bitset_and_witness(&a, &b), &c);
        let or_over_and_right = bitset_and_witness(&bitset_or_witness(&a, &c), &bitset_or_witness(&b, &c));
        prop_assert_eq!(or_over_and_left, or_over_and_right, "(a & b) | c must equal (a | c) & (b | c)");
    }

    #[test]
    fn absorption_laws_hold(
        pairs in proptest::collection::vec(any::<(u32, u32)>(), 0..=16),
    ) {
        let (a, b) = split_pairs(pairs);

        prop_assert!(same(&a, &bitset_and_witness(&a, &bitset_or_witness(&a, &b))), "a & (a | b) must equal a");
        prop_assert!(same(&a, &bitset_or_witness(&a, &bitset_and_witness(&a, &b))), "a | (a & b) must equal a");
    }

    #[test]
    fn de_morgan_laws_hold_for_equal_width_inputs(
        pairs in proptest::collection::vec(any::<(u32, u32)>(), 0..=16),
    ) {
        let (a, b) = split_pairs(pairs);

        let not_and = bitset_not_witness(&bitset_and_witness(&a, &b));
        let not_a_or_not_b = bitset_or_witness(&bitset_not_witness(&a), &bitset_not_witness(&b));
        prop_assert!(same(&not_and, &not_a_or_not_b), "!(a & b) must equal !a | !b");

        let not_or = bitset_not_witness(&bitset_or_witness(&a, &b));
        let not_a_and_not_b = bitset_and_witness(&bitset_not_witness(&a), &bitset_not_witness(&b));
        prop_assert!(same(&not_or, &not_a_and_not_b), "!(a | b) must equal !a & !b");
    }
}
