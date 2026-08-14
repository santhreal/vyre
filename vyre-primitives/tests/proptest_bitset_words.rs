//! Property gates for `bitset::bitset_words`, the one word-count rule every
//! bitset and graph frontier buffer is sized by.

#![cfg(feature = "bitset")]

use proptest::prelude::*;
use vyre_primitives::bitset::bitset_words;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10_000))]

    #[test]
    fn bitset_words_is_ceiling_division_by_32(node_count in 0u32..=10_000) {
        let expected = node_count.div_ceil(32);
        prop_assert_eq!(bitset_words(node_count), expected);
    }

    #[test]
    fn bitset_words_monotone_non_decreasing(
        a in 0u32..=5000,
        delta in 0u32..=5000,
    ) {
        let b = a.saturating_add(delta);
        prop_assert!(bitset_words(a) <= bitset_words(b));
    }

    #[test]
    fn bitset_words_jumps_only_at_multiples_of_32(k in 0u32..=200) {
        let base = k * 32;
        prop_assert_eq!(bitset_words(base), k);
        if base > 0 {
            prop_assert_eq!(bitset_words(base - 1), k);
        }
        prop_assert_eq!(bitset_words(base + 1), k + 1);
    }
}
