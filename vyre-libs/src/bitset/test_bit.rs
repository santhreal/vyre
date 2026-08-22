//! `bitset_test_bit`  -  scalar query: write 1 to `out_scalar` iff
//! the bit at `bit_idx` of `buf` is set, else 0.

use vyre_foundation::composition::wrap_anonymous_region;

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Canonical op id.
pub const OP_ID: &str = "vyre-libs::bitset::test_bit";

/// Build a Program: `out_scalar[0] = (buf[bit_idx/32] >> (bit_idx%32)) & 1`,
/// or `0` when `bit_idx/32 >= words` (out of range), matching reference semantics.
/// `words` is the length of `buf` in u32 words; it bounds the load so an
/// out-of-range `bit_idx` cannot read past the buffer on the GPU.
#[must_use]
pub fn bitset_test_bit(buf: &str, bit_idx: u32, out_scalar: &str, words: u32) -> Program {
    // AUDIT_2026-07-10: gate the `buf[bit_idx/32]` load on an in-bounds check,
    // mirroring the sibling `bitset_contains` fix (F-BSC-01). The load used to be
    // unconditional, so an out-of-range `bit_idx` (word >= words) was an OOB GPU
    // read (undefined behaviour or a page fault on a real GPU) while the reference safely returned 0, a CPU/GPU
    // parity divergence and a GPU safety hole. Out-of-range now stores 0.
    let word = bit_idx / 32;
    let bit = bit_idx % 32;
    let body = vec![Node::if_then_else(
        Expr::lt(Expr::u32(word), Expr::u32(words)),
        vec![Node::store(
            out_scalar,
            Expr::u32(0),
            Expr::bitand(
                Expr::shr(Expr::load(buf, Expr::u32(word)), Expr::u32(bit)),
                Expr::u32(1),
            ),
        )],
        vec![Node::store(out_scalar, Expr::u32(0), Expr::u32(0))],
    )];
    Program::wrapped(
        vec![
            BufferDecl::storage(buf, 0, BufferAccess::ReadOnly, DataType::U32).with_count(words),
            BufferDecl::storage(out_scalar, 1, BufferAccess::ReadWrite, DataType::U32)
                .with_count(1),
        ],
        [1, 1, 1],
        vec![wrap_anonymous_region(OP_ID, body)],
    )
}
const EXPECTED_BITSET_TEST_BIT_OUTPUT_BYTES: [u8; 4] = [1, 0, 0, 0];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || bitset_test_bit("buf", 0, "out", 1),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![vec![
                to_bytes(&[1]),
                to_bytes(&[0]),
            ]]
        }),
        Some(|| {
            vec![vec![EXPECTED_BITSET_TEST_BIT_OUTPUT_BYTES.to_vec()]]
        }),
    )
}

#[cfg(test)]
mod tests {
    use vyre_reference::composition_witness::bitset_test_bit_witness as reference_bitset_test_bit;

    #[test]
    fn bit_set_returns_one() {
        assert_eq!(reference_bitset_test_bit(&[0b1010], 1), 1);
        assert_eq!(reference_bitset_test_bit(&[0b1010], 3), 1);
    }

    #[test]
    fn bit_unset_returns_zero() {
        assert_eq!(reference_bitset_test_bit(&[0b1010], 0), 0);
        assert_eq!(reference_bitset_test_bit(&[0b1010], 2), 0);
    }

    #[test]
    fn out_of_range_returns_zero() {
        assert_eq!(reference_bitset_test_bit(&[0xFFFF_FFFF], 1024), 0);
    }

    #[test]
    fn bit_in_second_word() {
        assert_eq!(reference_bitset_test_bit(&[0, 0b100], 34), 1);
    }

    // ------------------------------------------------------------------
    // Adversarial fixtures  -  empty, single-word all-bits, cross-word boundary.
    // ------------------------------------------------------------------

    #[test]
    fn empty_bitset_returns_zero() {
        assert_eq!(reference_bitset_test_bit(&[], 0), 0);
        assert_eq!(reference_bitset_test_bit(&[], 31), 0);
        assert_eq!(reference_bitset_test_bit(&[], 32), 0);
    }

    #[test]
    fn single_word_all_bits() {
        let word = 0xFFFF_FFFF;
        for bit in 0..32 {
            assert_eq!(
                reference_bitset_test_bit(&[word], bit),
                1,
                "bit {bit} must be 1 in all-ones word"
            );
        }
    }

    #[test]
    fn cross_word_boundary_adjacent_bits() {
        // Word 0 bit 31 and word 1 bit 0 are adjacent node indices.
        let buf = vec![0x8000_0000, 0x0000_0001];
        assert_eq!(reference_bitset_test_bit(&buf, 31), 1, "bit 31 in word 0");
        assert_eq!(reference_bitset_test_bit(&buf, 32), 1, "bit 0 in word 1");
        assert_eq!(
            reference_bitset_test_bit(&buf, 30),
            0,
            "bit 30 in word 0 is unset"
        );
        assert_eq!(
            reference_bitset_test_bit(&buf, 33),
            0,
            "bit 1 in word 1 is unset"
        );
    }
}
