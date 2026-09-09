//! Device dispatch of packed INT4 batched matmul reduced to a top-1 score and
//! row index per batch item.
//!
//! The reduction happens on the device, so a caller that only needs the winning
//! row never reads back the full score matrix.

use super::*;
use vyre_megakernel::{SemanticExecutionError, SemanticExecutionPolicy, SemanticExecutor};

/// Compute top-1 scores and row indices for packed signed INT4 batched matmul through the backend.
///
/// The returned scores and indices each have exactly `batch` values.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when dimensions are zero, input shapes are wrong,
/// dispatch fails, or backend readback is malformed.
pub fn i4x8_batched_matmul_top1_f32_scaled_via(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    operands: &PackedI4BatchedMatmul<'_>,
) -> Result<(Vec<f32>, Vec<u32>), SemanticExecutionError> {
    let mut scratch = QuantizedBatchedMatmulTop1GpuScratch::default();
    let mut scores = Vec::new();
    let mut indices = Vec::new();
    i4x8_batched_matmul_top1_f32_scaled_via_with_scratch_into(
        dispatcher,
        policy,
        operands,
        &mut scratch,
        &mut scores,
        &mut indices,
    )?;
    Ok((scores, indices))
}

/// Compute top-1 scores and row indices for packed signed INT4 batched matmul through caller-owned scratch.
///
/// On success, `scores_out` and `indices_out` each contain exactly `batch`
/// values.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] under the same conditions as
/// [`i4x8_batched_matmul_top1_f32_scaled_via`].
pub fn i4x8_batched_matmul_top1_f32_scaled_via_with_scratch_into(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    operands: &PackedI4BatchedMatmul<'_>,
    scratch: &mut QuantizedBatchedMatmulTop1GpuScratch,
    scores_out: &mut Vec<f32>,
    indices_out: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    let batch = operands.batch;
    let batch_usize = batch as usize;
    let expected_words = batch_usize.checked_mul(2).ok_or_else(|| {
        SemanticExecutionError::InvalidRequest(format!(
            "Fix: i4x8_batched_matmul_top1_f32_scaled_via batch={batch} overflows usize."
        ))
    })?;

    let QuantizedBatchedMatmulTop1GpuScratch {
        inputs,
        program_cache,
    } = scratch;
    let mut values = Vec::new();
    dispatch_packed_batched_matmul(
        "i4x8_batched_matmul_top1_f32_scaled_via",
        dispatcher,
        policy,
        operands,
        inputs,
        program_cache,
        Some(expected_words),
        || {
            i4x8_batched_matmul_top1_f32_scaled(
                "weights",
                "activations",
                "row_scales",
                "batch_scales",
                "scores",
                operands.batch,
                operands.rows,
                operands.cols,
            )
        },
        &mut values,
    )?;
    // The kernel stores exact integer indices as f32 values in the output's second half.
    scores_out.clear();
    indices_out.clear();
    scores_out.reserve(batch_usize);
    indices_out.reserve(batch_usize);
    scores_out.extend_from_slice(&values[..batch_usize]);
    indices_out.extend(values[batch_usize..].iter().map(|&v| v as u32));
    Ok(())
}
