//! Property gates for `vyre_libs::reduce::count_non_zero::cpu_ref`.

#![cfg(feature = "reduce")]

use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10_000))]

    #[test]
    fn matches_manual_count(
        values in proptest::collection::vec(any::<u32>(), 0..=64),
    ) {
        let expected = values.iter().filter(|&&v| v != 0).count() as u32;
        prop_assert_eq!(cpu_ref(&values), expected);
    }

    #[test]
    fn empty_returns_zero(_dummy in 0u32..1) {
        prop_assert_eq!(cpu_ref(&[]), 0);
    }

    #[test]
    fn all_nonzero_returns_len(len in 0usize..64) {
        let values = vec![1u32; len];
        prop_assert_eq!(cpu_ref(&values), len as u32);
    }

    #[test]
    fn all_zero_returns_zero(len in 0usize..64) {
        let values = vec![0u32; len];
        prop_assert_eq!(cpu_ref(&values), 0);
    }
}

#[must_use]
fn cpu_ref(input: &[u32]) -> u32 {
    input.iter().filter(|&&w| w != 0).count() as u32
}
