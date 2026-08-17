//! `bitset_popcount`  -  per-word population count over a packed bitset.
//!
//! Produces a parallel `count_words[w]` array whose sum reduction
//! yields the total bit count. Reductions to a single scalar live
//! under [`crate::reduce`]; this primitive handles just the per-word
//! popcount so it can be composed.

use vyre_foundation::ir::{Program, UnOp};

use crate::bitset::unary_word::bitset_unary_word_program;

/// Canonical op id.
pub const OP_ID: &str = "vyre-libs::bitset::popcount";

/// Build a Program: `count_words[w] = popcount(input[w])`.
#[must_use]
pub fn bitset_popcount(input: &str, count_words: &str, words: u32) -> Program {
    bitset_unary_word_program(OP_ID, input, count_words, words, UnOp::Popcount)
}

#[cfg(test)]
fn reference_bitset_popcount(input: &[u32]) -> Vec<u32> {
    vyre_reference::composition_witness::bitset_popcount_witness(input)
}

#[cfg(test)]
fn reference_bitset_popcount_into(input: &[u32], output: &mut Vec<u32>) {
    vyre_reference::composition_witness::bitset_popcount_witness_into(input, output);
}

#[cfg(test)]
fn try_reference_bitset_popcount_into(input: &[u32], output: &mut Vec<u32>) -> Result<(), String> {
    reference_bitset_popcount_into(input, output);
    Ok(())
}

#[cfg(test)]
mod non_panic_wrapper_tests {
    use super::{
        reference_bitset_popcount, reference_bitset_popcount_into,
        try_reference_bitset_popcount_into,
    };

    #[test]
    fn compatibility_wrappers_match_fallible_reference() {
        let input = [0, 0xFFFF_FFFF, 0xAAAA_AAAA, 0x8000_0001];
        let mut compat = Vec::with_capacity(8);
        let mut fallible = Vec::with_capacity(8);

        reference_bitset_popcount_into(&input, &mut compat);
        try_reference_bitset_popcount_into(&input, &mut fallible)
            .expect("Fix: small bitset_popcount reference witness must reserve");

        assert_eq!(reference_bitset_popcount(&input), fallible);
        assert_eq!(compat, fallible);
    }
}
const EXPECTED_BITSET_POPCOUNT_OUTPUT_BYTES: [u8; 8] = [4, 0, 0, 0, 32, 0, 0, 0];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || bitset_popcount("input", "count", 2),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![vec![to_bytes(&[0b1111, 0xFFFF_FFFF]), to_bytes(&[0, 0])]]
        }),
        Some(|| {
            vec![vec![EXPECTED_BITSET_POPCOUNT_OUTPUT_BYTES.to_vec()]]
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn popcount_per_word() {
        assert_eq!(
            reference_bitset_popcount(&[0b1111, 0xFFFF_FFFF]),
            vec![4, 32]
        );
    }

    #[test]
    fn popcount_into_reuses_output() {
        let mut out = Vec::with_capacity(4);
        reference_bitset_popcount_into(&[0b1111, 0xFFFF_FFFF], &mut out);
        let capacity = out.capacity();
        assert_eq!(out, vec![4, 32]);

        reference_bitset_popcount_into(&[0b1010], &mut out);
        assert_eq!(out.capacity(), capacity);
        assert_eq!(out, vec![2]);
    }

    #[test]
    fn popcount_into_truncates_stale_tail_without_reallocating() {
        let mut out = Vec::with_capacity(8);
        out.extend([99u32; 8]);
        let ptr = out.as_ptr();

        try_reference_bitset_popcount_into(&[0b1111, 0xFFFF_FFFF], &mut out).unwrap();

        assert_eq!(out, vec![4, 32]);
        assert_eq!(out.as_ptr(), ptr);
    }

    // ------------------------------------------------------------------
    // Adversarial fixtures  -  empty, all-zeros, all-ones, alternating, cross-word.
    // ------------------------------------------------------------------

    #[test]
    fn empty_bitset() {
        assert_eq!(reference_bitset_popcount(&[]), Vec::<u32>::new());
    }

    #[test]
    fn single_word_all_zeros() {
        assert_eq!(reference_bitset_popcount(&[0]), vec![0]);
    }

    #[test]
    fn single_word_all_ones() {
        assert_eq!(reference_bitset_popcount(&[0xFFFF_FFFF]), vec![32]);
    }

    #[test]
    fn alternating_pattern() {
        // 0xAAAA_AAAA = 1010...1010 → 16 ones
        assert_eq!(reference_bitset_popcount(&[0xAAAA_AAAA]), vec![16]);
        // 0x5555_5555 = 0101...0101 → 16 ones
        assert_eq!(reference_bitset_popcount(&[0x5555_5555]), vec![16]);
    }

    #[test]
    fn cross_word_boundary() {
        // Two words: one with bit 31 set, one with bit 0 set.
        assert_eq!(
            reference_bitset_popcount(&[0x8000_0000, 0x0000_0001]),
            vec![1, 1]
        );
    }

    #[test]
    fn generated_popcount_matches_scalar_reference() {
        for len in 0..96usize {
            let input: Vec<u32> = (0..len)
                .map(|idx| {
                    (idx as u32)
                        .wrapping_mul(0x85EB_CA6B)
                        .wrapping_add(len as u32)
                })
                .collect();
            let mut out = Vec::with_capacity(len + 3);

            try_reference_bitset_popcount_into(&input, &mut out).unwrap();

            assert_eq!(
                out,
                input
                    .iter()
                    .map(|word| word.count_ones())
                    .collect::<Vec<_>>(),
                "generated popcount case len={len}"
            );
        }
    }
}
