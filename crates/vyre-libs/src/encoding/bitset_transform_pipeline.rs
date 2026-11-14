//! Advanced bitset transform pipeline over `vyre-primitives::bitset`.
//!
//! These wrappers cover the in-place and tiled bitset primitives used when
//! self-substrate passes keep masks resident: frontier subtraction, monotone
//! accumulator growth, symmetric-difference change detection, select navigation,
//! and Method-of-Four-Russians byte-tile application.

use crate::bitset::four_russians::{
    four_russians_dense_matvec_byte_lut, frontier_words_for_byte_tiles,
};
use vyre_foundation::ir::Program;

/// Frontier words needed for dense byte-tile Four-Russians matvec.
#[must_use]
pub const fn dense_matvec_frontier_words(tile_count: u32) -> u32 {
    frontier_words_for_byte_tiles(tile_count)
}

/// Build dense boolean matvec over packed frontier byte tiles.
#[must_use]
pub fn four_russians_dense_matvec_program(
    frontier: &str,
    tile_lut: &str,
    out: &str,
    tile_count: u32,
    dst_words: u32,
) -> Program {
    four_russians_dense_matvec_byte_lut(frontier, tile_lut, out, tile_count, dst_words)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitset::four_russians::{
        dense_matvec_byte_lut as dense_boolean_matvec_lut,
        dense_matvec_byte_lut_words as dense_matvec_lut_words,
    };
    use vyre_reference::composition_witness::four_russians_dense_matvec_witness as reference_dense_boolean_matvec;

    #[test]
    fn program_builder_emits_expected_bitset_primitive() {
        let program = four_russians_dense_matvec_program("frontier", "tile_lut", "out", 2, 2);
        let actual = vyre_test_support::test_parity_oracles::region_operation_id(&program)
            .expect("Fix: build a primitive program that carries a region generator");
        assert_eq!(actual, crate::bitset::four_russians::DENSE_MATVEC_OP_ID);
    }

    #[test]
    fn dense_four_russians_matvec_lut_matches_transform_contract() {
        let columns = [
            0b0001_u32, 0b0010, 0b0100, 0b1000, 0b0001, 0b0010, 0b0100, 0b1000,
        ];
        let lut = dense_boolean_matvec_lut(&columns, 1, 1);

        assert_eq!(dense_matvec_frontier_words(1), 1);
        assert_eq!(dense_matvec_lut_words(1, 1), 256);
        assert_eq!(
            reference_dense_boolean_matvec(&[0b0000_0101], &lut, 1, 1),
            vec![0b0101]
        );
    }
}
