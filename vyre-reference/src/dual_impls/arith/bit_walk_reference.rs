//! Arithmetic references built from bit operations alone: addition by carry
//! propagation and multiplication by shift-and-add, so neither shares code
//! with the direct `u32` arm of its dual pair.

use crate::dual_impls::evaluator::read_two_words;

/// Wrapping-add reference implemented through carry propagation only.
#[must_use]
pub(crate) fn wrapping_add_bits_reference(input: &[u8]) -> Vec<u8> {
    let Some((left, right)) = read_two_words(input) else {
        return zero_word();
    };
    wrapping_add_bits(left, right).to_le_bytes().to_vec()
}

/// Wrapping-multiply reference implemented as shift-and-add over bits.
#[must_use]
pub(crate) fn wrapping_mul_shift_add_reference(input: &[u8]) -> Vec<u8> {
    let Some((mut multiplicand, mut multiplier)) = read_two_words(input) else {
        return zero_word();
    };
    let mut acc = 0u32;
    while multiplier != 0 {
        if multiplier & 1 != 0 {
            acc = wrapping_add_bits(acc, multiplicand);
        }
        multiplicand = multiplicand.wrapping_shl(1);
        multiplier >>= 1;
    }
    acc.to_le_bytes().to_vec()
}

fn wrapping_add_bits(mut left: u32, mut right: u32) -> u32 {
    while right != 0 {
        let carry = left & right;
        left ^= right;
        right = carry.wrapping_shl(1);
    }
    left
}

fn zero_word() -> Vec<u8> {
    vec![0; 4]
}

// Inline: covers the crate-private `wrapping_add_bits_reference` and `wrapping_mul_shift_add_reference`, which no integration test can reach.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::dual_impls::evaluator::binary_direct;

    #[test]
    fn generated_arithmetic_duals_match_native_wrapping_ops() {
        for case in 0..8192u32 {
            let left = case.wrapping_mul(0x9e37_79b9).rotate_left(case & 31);
            let right = case ^ 0xa5a5_5a5a_u32.rotate_right(case & 31);
            let mut input = Vec::with_capacity(8);
            input.extend_from_slice(&left.to_le_bytes());
            input.extend_from_slice(&right.to_le_bytes());

            assert_eq!(
                binary_direct(&input, u32::wrapping_add),
                wrapping_add_bits_reference(&input),
                "Fix: arithmetic add duals diverged for left={left:#010x} right={right:#010x}"
            );
            assert_eq!(
                binary_direct(&input, u32::wrapping_mul),
                wrapping_mul_shift_add_reference(&input),
                "Fix: arithmetic mul duals diverged for left={left:#010x} right={right:#010x}"
            );
        }
    }

    #[test]
    fn short_inputs_zero_fill_without_panicking() {
        assert_eq!(binary_direct(&[1, 2, 3], u32::wrapping_add), vec![0; 4]);
        assert_eq!(wrapping_add_bits_reference(&[1, 2, 3]), vec![0; 4]);
        assert_eq!(wrapping_mul_shift_add_reference(&[1, 2, 3]), vec![0; 4]);
    }
}
