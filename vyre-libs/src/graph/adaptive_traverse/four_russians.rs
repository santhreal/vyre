//! Four-Russians dense traversal: source-byte tile shapes, the column
//! transpose and LUT build feeding them, and the graph-level program.

use vyre_foundation::composition::trap_program;
use vyre_foundation::ir::{DataType, Program};

use super::FOUR_RUSSIANS_DENSE_OP_ID;
use crate::bitset::bitset_words;
use crate::bitset::four_russians::{
    dense_matvec_byte_lut, dense_matvec_byte_lut_words, four_russians_dense_matvec_byte_lut,
    frontier_words_for_byte_tiles,
};

/// Source-byte tile count for Four-Russians dense graph traversal.
#[must_use]
pub const fn four_russians_source_tile_count(node_count: u32) -> u32 {
    node_count.div_ceil(8)
}

/// Frontier word count for graph-level Four-Russians dense traversal.
#[must_use]
pub const fn four_russians_frontier_words(node_count: u32) -> u32 {
    frontier_words_for_byte_tiles(four_russians_source_tile_count(node_count))
}

/// LUT word count for graph-level Four-Russians dense traversal.
#[must_use]
pub fn four_russians_dense_lut_words(node_count: u32) -> u32 {
    dense_matvec_byte_lut_words(
        four_russians_source_tile_count(node_count),
        bitset_words(node_count),
    )
}

/// Transpose dense reverse-adjacency rows into source-column words.
///
/// `adj_rows_dense[dst][src] == 1` becomes `columns[src][dst] == 1`,
/// grouped by 8-source byte tiles for Four-Russians LUT construction.
///
/// # Errors
///
/// Returns an actionable diagnostic when `node_count` is zero, the dense row
/// matrix has the wrong shape, or the derived column table overflows `usize`.
pub fn four_russians_dense_columns_from_adj_rows(
    node_count: u32,
    adj_rows_dense: &[u32],
) -> Result<Vec<u32>, String> {
    if node_count == 0 {
        return Err(
            "Fix: Four-Russians adaptive dense traversal requires node_count > 0.".to_string(),
        );
    }
    let words = bitset_words(node_count) as usize;
    let expected_rows = (node_count as usize).checked_mul(words).ok_or_else(|| {
        format!(
            "Fix: Four-Russians adaptive dense row count overflows usize for {node_count} nodes and {words} words."
        )
    })?;
    if adj_rows_dense.len() != expected_rows {
        return Err(format!(
            "Fix: Four-Russians adaptive dense traversal expected {expected_rows} row words for {node_count} nodes, got {}.",
            adj_rows_dense.len()
        ));
    }
    let tile_count = four_russians_source_tile_count(node_count) as usize;
    let column_count = tile_count
        .checked_mul(8)
        .and_then(|columns| columns.checked_mul(words))
        .ok_or_else(|| {
            format!(
                "Fix: Four-Russians adaptive dense column table overflows usize for {node_count} nodes and {words} destination words."
            )
        })?;
    let mut columns = vec![0u32; column_count];

    for dst in 0..node_count as usize {
        let row_start = dst * words;
        let dst_word = dst / 32;
        let dst_bit = 1u32 << (dst % 32);
        for src_word in 0..words {
            let mut word = adj_rows_dense[row_start + src_word];
            while word != 0 {
                let bit = word.trailing_zeros() as usize;
                let src = src_word * 32 + bit;
                if src < node_count as usize {
                    let source_column = (src / 8) * 8 + (src % 8);
                    let column_idx = source_column * words + dst_word;
                    columns[column_idx] |= dst_bit;
                }
                word &= word - 1;
            }
        }
    }

    Ok(columns)
}

/// Build a Four-Russians dense traversal LUT from dense reverse rows.
///
/// # Errors
///
/// Propagates dense-row validation failures from
/// [`four_russians_dense_columns_from_adj_rows`].
pub fn four_russians_dense_lut_from_adj_rows(
    node_count: u32,
    adj_rows_dense: &[u32],
) -> Result<Vec<u32>, String> {
    let columns = four_russians_dense_columns_from_adj_rows(node_count, adj_rows_dense)?;
    let expected_words = four_russians_dense_lut_words(node_count) as usize;
    let lut = dense_matvec_byte_lut(
        &columns,
        four_russians_source_tile_count(node_count),
        bitset_words(node_count),
    );
    if lut.len() != expected_words {
        return Err(format!(
            "Fix: Four-Russians dense LUT build expected {expected_words} words for {node_count} nodes, got {}.",
            lut.len()
        ));
    }
    Ok(lut)
}

/// Build the graph-level Four-Russians dense traversal Program.
#[must_use]
pub fn adaptive_four_russians_dense_step(
    frontier_in: &str,
    tile_lut: &str,
    frontier_out: &str,
    node_count: u32,
) -> Program {
    if node_count == 0 {
        return trap_program(
            FOUR_RUSSIANS_DENSE_OP_ID,
            Some((frontier_out, DataType::U32)),
            "Fix: adaptive_four_russians_dense_step requires node_count > 0, got 0.".to_string(),
        );
    }
    four_russians_dense_matvec_byte_lut(
        frontier_in,
        tile_lut,
        frontier_out,
        four_russians_source_tile_count(node_count),
        bitset_words(node_count),
    )
}
