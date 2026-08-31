//! `bitset_not`  -  per-word bitwise NOT over a packed bitset.

use vyre_foundation::ir::{Program, UnOp};

use crate::bitset::unary_word::bitset_unary_word_program;

/// Canonical op id.
pub const OP_ID: &str = "vyre-libs::bitset::not";

/// Build a Program: `out[w] = !input[w]`.
///
/// Per-word bitwise complement for boolean-algebra lowering when a negated
/// predicate needs a complement bitset for a subsequent intersection.
///
/// # Example
///
/// ```
/// use vyre_libs::bitset::not::bitset_not;
///
/// let program = bitset_not("input", "out", 4);
/// assert_eq!(program.entry.len(), 1);
/// ```
#[must_use]
pub fn bitset_not(input: &str, out: &str, words: u32) -> Program {
    bitset_unary_word_program(OP_ID, input, out, words, UnOp::BitNot)
}

const EXPECTED_BITSET_NOT_OUTPUT_BYTES: [u8; 4] = [0xF0, 0xF0, 0xF0, 0xF0];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID,
        || bitset_not("input", "out", 1),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![vec![to_bytes(&[0x0F0F_0F0F])]]
        }),
        Some(|| {
            vec![vec![EXPECTED_BITSET_NOT_OUTPUT_BYTES.to_vec()]]
        }),
    )
}

#[cfg(test)]
mod tests {
    use vyre_reference::composition_witness::{
        bitset_not_witness as reference_bitset_not,
        bitset_not_witness_into as reference_bitset_not_into,
    };

    fn try_reference_bitset_not_into(input: &[u32], out: &mut Vec<u32>) -> Result<(), ()> {
        reference_bitset_not_into(input, out);
        Ok(())
    }

    #[test]
    fn flips_every_bit() {
        assert_eq!(reference_bitset_not(&[0x0F0F_0F0F]), vec![0xF0F0_F0F0]);
    }

    #[test]
    fn empty_bitset() {
        assert_eq!(reference_bitset_not(&[]), Vec::<u32>::new());
    }
    fn single_word_all_bits() {
        assert_eq!(reference_bitset_not(&[0xFFFF_FFFF]), vec![0x0000_0000]);
        assert_eq!(reference_bitset_not(&[0x0000_0000]), vec![0xFFFF_FFFF]);
    }

    #[test]
    fn cross_word_boundary() {
        let input = vec![0x8000_0000, 0x0000_0001];
        assert_eq!(reference_bitset_not(&input), vec![0x7FFF_FFFF, 0xFFFF_FFFE]);
    }

    #[test]
    fn reference_bitset_not_into_truncates_stale_tail_without_reallocating() {
        let input = vec![0x8000_0000, 0x0000_0001];
        let mut out = Vec::with_capacity(8);
        out.extend([0xDEAD_BEEF; 8]);
        let ptr = out.as_ptr();

        try_reference_bitset_not_into(&input, &mut out).unwrap();

        assert_eq!(out, vec![0x7FFF_FFFF, 0xFFFF_FFFE]);
        assert_eq!(out.as_ptr(), ptr);
    }

    #[test]
    fn compatibility_wrappers_match_fallible_reference() {
        let input = [0x1234_5678, 0xFFFF_0000];
        let mut compat = Vec::with_capacity(4);
        let mut fallible = Vec::with_capacity(4);

        reference_bitset_not_into(&input, &mut compat);
        try_reference_bitset_not_into(&input, &mut fallible)
            .expect("Fix: small bitset_not reference witness must reserve");

        assert_eq!(reference_bitset_not(&input), fallible);
        assert_eq!(compat, fallible);
    }

    #[test]
    fn generated_not_is_involutive_and_matches_scalar_reference() {
        for len in 0..96usize {
            let input: Vec<u32> = (0..len)
                .map(|idx| {
                    (idx as u32)
                        .wrapping_mul(0x9E37_79B9)
                        .wrapping_add(len as u32)
                })
                .collect();
            let mut out = Vec::with_capacity(len + 3);
            let mut roundtrip = Vec::with_capacity(len + 3);

            try_reference_bitset_not_into(&input, &mut out).unwrap();
            try_reference_bitset_not_into(&out, &mut roundtrip).unwrap();

            assert_eq!(
                out,
                input.iter().map(|word| !word).collect::<Vec<_>>(),
                "generated not case len={len}"
            );
            assert_eq!(roundtrip, input, "generated involution case len={len}");
        }
    }
}
