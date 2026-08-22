//! Unit tests for packed INT4 quantized primitives.

use super::*;

fn pack_i4_matrix_rows(rows: &[Vec<i32>]) -> Vec<u32> {
    let cols = rows.first().map_or(0, Vec::len) as u32;
    let words_per_row = i4_packed_words(cols) as usize;
    let mut out = Vec::with_capacity(rows.len() * words_per_row);
    for row in rows {
        let mut packed = pack_i4x8_cpu(row);
        packed.resize(words_per_row, 0);
        out.extend_from_slice(&packed);
    }
    out
}

mod batched_matmul_contracts;
mod dot_contracts;
mod layout_contracts;
mod matvec_contracts;
mod pack_unpack_contracts;
mod zero_shape_contracts;
