//! Stochastic computing primitive (#59, research scaffold).
//!
//! Stochastic computing (Gaines 1969, Alaghi 2018 revival) represents
//! numbers as bitstreams; multiplication = AND, addition = MUX.
//! Trades precision for power efficiency. Recent NN inference work
//! (Tehrani 2023) uses it on GPU as bitset operations.
//!
//! This file ships **stochastic-AND multiplication**  -  multiply two
//! bitstream representations elementwise.

use super::binary_word::{binary_word_program, BitwiseBinaryOp};
use vyre_foundation::composition::trap_program;
use vyre_foundation::ir::{DataType, Program};

/// Op id.
pub const OP_ID: &str = "vyre-libs::bitset::stochastic_and_mul";

/// Stochastic multiply (AND of bitstreams).
#[must_use]
pub fn stochastic_and_mul(a: &str, b: &str, out: &str, n_words: u32) -> Program {
    if n_words == 0 {
        return trap_program(
            OP_ID,
            Some((out, DataType::U32)),
            "Fix: stochastic_and_mul requires n_words > 0, got 0.".to_string(),
        );
    }

    binary_word_program(OP_ID, a, b, out, n_words, BitwiseBinaryOp::And)
}

#[cfg(test)]
fn reference_stochastic_and_mul(left: &[u32], right: &[u32]) -> Vec<u32> {
    vyre_reference::composition_witness::bitset_and_witness(left, right)
}

#[cfg(test)]
fn reference_stochastic_and_mul_into(left: &[u32], right: &[u32], output: &mut Vec<u32>) {
    vyre_reference::composition_witness::bitset_and_witness_into(left, right, output);
}

#[cfg(test)]
fn try_reference_stochastic_and_mul_into(
    left: &[u32],
    right: &[u32],
    output: &mut Vec<u32>,
) -> Result<(), String> {
    reference_stochastic_and_mul_into(left, right, output);
    Ok(())
}

#[cfg(test)]
#[must_use]
fn encode_bitstream(p: f64, len_bits: usize, seed: u32) -> Vec<u32> {
    vyre_reference::composition_witness::stochastic_encode_witness(p, len_bits, seed)
}

#[cfg(test)]
fn encode_bitstream_into(p: f64, len_bits: usize, seed: u32, out: &mut Vec<u32>) {
    try_encode_bitstream_into(p, len_bits, seed, out)
        .unwrap_or_else(|error| panic!("stochastic bitstream test witness failed: {error}"));
}

#[cfg(test)]
fn try_encode_bitstream_into(
    p: f64,
    len_bits: usize,
    seed: u32,
    out: &mut Vec<u32>,
) -> Result<(), String> {
    vyre_reference::composition_witness::try_stochastic_encode_witness_into(p, len_bits, seed, out)
}

#[cfg(test)]
#[must_use]
fn decode_bitstream(bitstream: &[u32], len_bits: usize) -> f64 {
    vyre_reference::composition_witness::stochastic_decode_witness(bitstream, len_bits)
}

#[cfg(test)]
mod non_panic_wrapper_tests {
    use super::{
        encode_bitstream, encode_bitstream_into, reference_stochastic_and_mul,
        reference_stochastic_and_mul_into, try_encode_bitstream_into,
        try_reference_stochastic_and_mul_into,
    };

    #[test]
    fn convenience_reference_wrappers_match_fallible_reference() {
        let a = [0xF0F0_F0F0, 0xAAAA_AAAA];
        let b = [0xFF00_00FF, 0x5555_FFFF];
        let mut compat = Vec::with_capacity(4);
        let mut fallible = Vec::with_capacity(4);

        reference_stochastic_and_mul_into(&a, &b, &mut compat);
        try_reference_stochastic_and_mul_into(&a, &b, &mut fallible)
            .expect("Fix: small stochastic reference witness must reserve");

        assert_eq!(reference_stochastic_and_mul(&a, &b), fallible);
        assert_eq!(compat, fallible);
    }

    #[test]
    fn convenience_encoder_wrappers_match_fallible_encoder() {
        let mut compat = Vec::with_capacity(8);
        let mut fallible = Vec::with_capacity(8);

        encode_bitstream_into(0.25, 65, 7, &mut compat);
        try_encode_bitstream_into(0.25, 65, 7, &mut fallible)
            .expect("Fix: small stochastic encoder must reserve");

        assert_eq!(encode_bitstream(0.25, 65, 7), fallible);
        assert_eq!(compat, fallible);
    }
}
const EXPECTED_STOCHASTIC_AND_MUL_OUTPUT_BYTES: [u8; 8] =
    [0xF0, 0x00, 0x00, 0xF0, 0xAA, 0xAA, 0x00, 0x00];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || stochastic_and_mul("a", "b", "out", 2),
        Some(|| {
            vec![vec![
                vyre_primitives::wire::pack_u32_slice(&[0xF0F0_F0F0, 0xAAAA_AAAA]),
                vyre_primitives::wire::pack_u32_slice(&[0xFF00_00FF, 0x5555_FFFF]),
                vyre_primitives::wire::pack_u32_slice(&[0, 0]),
            ]]
        }),
        Some(|| {
            vec![vec![EXPECTED_STOCHASTIC_AND_MUL_OUTPUT_BYTES.to_vec()]]
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_roundtrip_low_p() {
        let bs = encode_bitstream(0.25, 1024, 42);
        let p = decode_bitstream(&bs, 1024);
        assert!((p - 0.25).abs() < 0.05);
    }

    #[test]
    fn encode_decode_roundtrip_high_p() {
        let bs = encode_bitstream(0.75, 1024, 42);
        let p = decode_bitstream(&bs, 1024);
        assert!((p - 0.75).abs() < 0.05);
    }

    #[test]
    fn encode_bitstream_into_reuses_output() {
        let mut bs = Vec::with_capacity(64);
        let ptr = bs.as_ptr();
        encode_bitstream_into(0.25, 1024, 42, &mut bs);
        assert!((decode_bitstream(&bs, 1024) - 0.25).abs() < 0.05);
        assert_eq!(bs.as_ptr(), ptr);
    }

    #[test]
    fn try_encode_bitstream_into_truncates_stale_tail_without_reallocating() {
        let mut bs = Vec::with_capacity(16);
        bs.extend_from_slice(&[u32::MAX; 16]);
        let ptr = bs.as_ptr();

        try_encode_bitstream_into(0.0, 65, 42, &mut bs).unwrap();

        assert_eq!(bs.len(), 3);
        assert_eq!(bs.as_ptr(), ptr);
        assert!(bs.iter().all(|word| *word == 0));
    }

    #[test]
    fn zero_p_yields_zero_bitstream() {
        let bs = encode_bitstream(0.0, 256, 1);
        for w in bs {
            assert_eq!(w, 0);
        }
    }

    #[test]
    fn reference_multiplies_bitstreams_with_and() {
        assert_eq!(
            reference_stochastic_and_mul(&[0xF0F0_F0F0, 0xAAAA_AAAA], &[0xFF00_00FF, 0x5555_FFFF]),
            vec![0xF000_00F0, 0x0000_AAAA]
        );
    }

    #[test]
    fn try_reference_stochastic_and_mul_into_truncates_stale_tail_without_reallocating() {
        let a = [0xffff_0000, 0x1357_9bdf, 0x2468_ace0];
        let b = [0x0f0f_f0f0, 0xffff_ffff];
        let mut out = Vec::with_capacity(8);
        out.extend_from_slice(&[u32::MAX; 8]);
        let ptr = out.as_ptr();

        try_reference_stochastic_and_mul_into(&a, &b, &mut out).unwrap();

        assert_eq!(out, vec![a[0] & b[0], a[1] & b[1]]);
        assert_eq!(out.as_ptr(), ptr);
    }

    #[test]
    fn generated_reference_matches_wordwise_and() {
        let mut out = Vec::new();
        for case in 0..4096_u32 {
            let a = [
                case.rotate_left(case % 31) ^ 0xA5A5_5A5A,
                case.wrapping_mul(0x9E37_79B9),
            ];
            let b = [
                case.rotate_right((case + 7) % 31) ^ 0x5A5A_A5A5,
                case.wrapping_mul(0x85EB_CA6B),
            ];
            reference_stochastic_and_mul_into(&a, &b, &mut out);
            assert_eq!(
                out,
                vec![a[0] & b[0], a[1] & b[1]],
                "generated stochastic AND case {case} must match wordwise multiplication"
            );
        }
    }

    #[test]
    fn ir_program_buffer_layout() {
        let p = stochastic_and_mul("a", "b", "out", 8);
        assert_eq!(p.workgroup_size, [256, 1, 1]);
        for buf in p.buffers.iter() {
            assert_eq!(buf.count(), 8);
        }
    }

    #[test]
    fn zero_n_words_traps() {
        let p = stochastic_and_mul("a", "b", "out", 0);
        assert!(p.stats().trap());
    }
}
