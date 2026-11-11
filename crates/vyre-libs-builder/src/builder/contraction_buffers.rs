//! Element counts and the read-only operand roster every contraction program declares.
//!
//! A tiled assembly routine and the untiled one it replaces declare the same
//! operands, in the same binding order, from the same shapes. Only the
//! bindings a schedule adds for itself, workgroup tiles and a padded output,
//! differ. Both are stated once here, so a new schedule inherits the roster
//! instead of restating it.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType};

use super::ContractionEpilogue;
use crate::plumbing::operand::tensor_ref::{element_count, TensorRefError};

/// Element counts of a 2D contraction, in binding order: `a` is `m x k`, `b`
/// is `k x n`, and the output is `m x n`.
pub(super) fn matmul_2d_counts(
    a: &str,
    b: &str,
    out: &str,
    m: u32,
    k: u32,
    n: u32,
) -> Result<(u32, u32, u32), TensorRefError> {
    Ok((
        element_count(a, &[m, k])?,
        element_count(b, &[k, n])?,
        element_count(out, &[m, n])?,
    ))
}

/// Element counts of a row-batched projection, in binding order: `x` is
/// `rows x in_dim`, `w` is `in_dim x out_dim`, and the output is
/// `rows x out_dim`.
pub(super) fn projection_counts(
    x: &str,
    w: &str,
    out: &str,
    rows: u32,
    in_dim: u32,
    out_dim: u32,
) -> Result<(u32, u32, u32), TensorRefError> {
    Ok((
        element_count(x, &[rows, in_dim])?,
        element_count(w, &[in_dim, out_dim])?,
        element_count(out, &[rows, out_dim])?,
    ))
}

/// Read-only operands of a 2D contraction, and the first binding slot left
/// free for the caller's own output and tile bindings.
///
/// A quantized epilogue reads one scale per output row and one per batch, and
/// 2D geometry has a single batch, so the batch scale is one element.
pub(super) fn matmul_2d_operands(
    a: &str,
    a_count: u32,
    b: &str,
    b_count: u32,
    bias: Option<&str>,
    epilogue: &ContractionEpilogue,
    dtype: &DataType,
    m: u32,
    n: u32,
) -> (Vec<BufferDecl>, u32) {
    let mut buffers = vec![
        BufferDecl::storage(a, 0, BufferAccess::ReadOnly, dtype.clone()).with_count(a_count),
        BufferDecl::storage(b, 1, BufferAccess::ReadOnly, dtype.clone()).with_count(b_count),
    ];
    let mut next_slot = 2;
    if let Some(bias_name) = bias {
        buffers.push(
            BufferDecl::storage(bias_name, next_slot, BufferAccess::ReadOnly, dtype.clone())
                .with_count(n),
        );
        next_slot += 1;
    }
    if let ContractionEpilogue::QuantizedScale {
        row_scales,
        batch_scales,
    } = epilogue
    {
        buffers.push(
            BufferDecl::storage(row_scales, next_slot, BufferAccess::ReadOnly, dtype.clone())
                .with_count(m),
        );
        buffers.push(
            BufferDecl::storage(
                batch_scales,
                next_slot + 1,
                BufferAccess::ReadOnly,
                dtype.clone(),
            )
            .with_count(1),
        );
        next_slot += 2;
    }
    (buffers, next_slot)
}

/// Read-only operands of a row-batched projection, and the first binding slot
/// left free for the caller's output.
pub(super) fn projection_operands(
    x: &str,
    input_count: u32,
    w: &str,
    weight_count: u32,
    bias: Option<&str>,
    out_dim: u32,
    dtype: &DataType,
) -> (Vec<BufferDecl>, u32) {
    let mut buffers = vec![
        BufferDecl::storage(x, 0, BufferAccess::ReadOnly, dtype.clone()).with_count(input_count),
        BufferDecl::storage(w, 1, BufferAccess::ReadOnly, dtype.clone()).with_count(weight_count),
    ];
    let mut next_slot = 2;
    if let Some(name) = bias {
        buffers.push(
            BufferDecl::storage(name, next_slot, BufferAccess::ReadOnly, dtype.clone())
                .with_count(out_dim),
        );
        next_slot += 1;
    }
    (buffers, next_slot)
}
