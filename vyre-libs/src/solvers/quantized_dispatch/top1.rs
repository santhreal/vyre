use super::*;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

/// Compute top-1 scores and row indices for packed signed INT4 batched matmul through the backend.
///
/// `weights_packed` is row-major `[rows][i4_packed_words(cols)]`.
/// `activation_batches_packed` is batch-major `[batch][i4_packed_words(cols)]`.
/// `row_scales` has `rows` f32 values and `batch_scales` has `batch` f32
/// values. The returned scores and indices each have exactly `batch` values.
///
/// # Errors
///
/// Returns [`DispatchError`] when dimensions are zero, input shapes are wrong,
/// dispatch fails, or backend readback is malformed.
pub fn i4x8_batched_matmul_top1_f32_scaled_via(
    dispatcher: &impl ProgramDispatcher,
    weights_packed: &[u32],
    activation_batches_packed: &[u32],
    row_scales: &[f32],
    batch_scales: &[f32],
    batch: u32,
    rows: u32,
    cols: u32,
) -> Result<(Vec<f32>, Vec<u32>), DispatchError> {
    let mut scratch = QuantizedBatchedMatmulTop1GpuScratch::default();
    let mut scores = Vec::new();
    let mut indices = Vec::new();
    i4x8_batched_matmul_top1_f32_scaled_via_with_scratch_into(
        dispatcher,
        weights_packed,
        activation_batches_packed,
        row_scales,
        batch_scales,
        batch,
        rows,
        cols,
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
/// Returns [`DispatchError`] under the same conditions as
/// [`i4x8_batched_matmul_top1_f32_scaled_via`].
pub fn i4x8_batched_matmul_top1_f32_scaled_via_with_scratch_into(
    dispatcher: &impl ProgramDispatcher,
    weights_packed: &[u32],
    activation_batches_packed: &[u32],
    row_scales: &[f32],
    batch_scales: &[f32],
    batch: u32,
    rows: u32,
    cols: u32,
    scratch: &mut QuantizedBatchedMatmulTop1GpuScratch,
    scores_out: &mut Vec<f32>,
    indices_out: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    let batch_usize = batch as usize;
    let expected_words = batch_usize.checked_mul(2).ok_or_else(|| {
        DispatchError::BadInputs(format!(
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
        weights_packed,
        activation_batches_packed,
        row_scales,
        batch_scales,
        batch,
        rows,
        cols,
        inputs,
        program_cache,
        Some(batch),
        Some(expected_words),
        || {
            i4x8_batched_matmul_top1_f32_scaled(
                "weights",
                "activations",
                "row_scales",
                "batch_scales",
                "scores",
                batch,
                rows,
                cols,
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
