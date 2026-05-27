//! Property gates for `vyre_primitives::bitset::and::cpu_ref`.

#![cfg(all(feature = "bitset", feature = "cpu-parity"))]

use proptest::prelude::*;
use vyre_primitives::bitset::and::cpu_ref;

fn manual_and(lhs: &[u32], rhs: &[u32]) -> Vec<u32> {
    lhs.iter().zip(rhs.iter()).map(|(a, b)| a & b).collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10_000))]

    #[test]
    fn cpu_ref_matches_manual_and(
        lhs in proptest::collection::vec(any::<u32>(), 0..=32),
        rhs in proptest::collection::vec(any::<u32>(), 0..=32),
    ) {
        prop_assert_eq!(cpu_ref(&lhs, &rhs), manual_and(&lhs, &rhs));
    }

    #[test]
    fn and_is_commutative(
        lhs in proptest::collection::vec(any::<u32>(), 1..=16),
        rhs in proptest::collection::vec(any::<u32>(), 1..=16),
    ) {
        let n = lhs.len().min(rhs.len());
        let l = &lhs[..n];
        let r = &rhs[..n];
        prop_assert_eq!(cpu_ref(l, r), cpu_ref(r, l));
    }

    #[test]
    fn and_with_zero_rhs_is_zero(
        lhs in proptest::collection::vec(any::<u32>(), 1..=16),
    ) {
        let zero = vec![0u32; lhs.len()];
        let out = cpu_ref(&lhs, &zero);
        prop_assert!(out.iter().all(|w| *w == 0));
    }

    #[test]
    fn and_with_ones_rhs_preserves_lhs(
        lhs in proptest::collection::vec(any::<u32>(), 1..=16),
    ) {
        let ones = vec![u32::MAX; lhs.len()];
        prop_assert_eq!(cpu_ref(&lhs, &ones), lhs);
    }

    #[test]
    fn and_is_idempotent(
        v in proptest::collection::vec(any::<u32>(), 0..=16),
    ) {
        prop_assert_eq!(cpu_ref(&v, &v), v);
    }
}
