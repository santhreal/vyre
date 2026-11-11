//! Device dispatch of packed INT4 batched matmul.

use super::*;
use vyre_megakernel::{SemanticExecutionError, SemanticExecutionPolicy, SemanticExecutor};

/// Compute packed signed INT4 batched matrix multiply through the backend.
///
/// The returned vector has `batch * rows` f32 values in batch-major order.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when dimensions are zero, input shapes are wrong,
/// dispatch fails, or backend readback is malformed.
pub fn i4x8_batched_matmul_f32_scaled_via(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    operands: &PackedI4BatchedMatmul<'_>,
) -> Result<Vec<f32>, SemanticExecutionError> {
    let mut scratch = QuantizedBatchedMatmulGpuScratch::default();
    let mut out = Vec::new();
    i4x8_batched_matmul_f32_scaled_via_with_scratch_into(
        dispatcher,
        policy,
        operands,
        &mut scratch,
        &mut out,
    )?;
    Ok(out)
}

/// Compute packed signed INT4 batched matrix multiply through caller-owned scratch.
///
/// On success, `out` contains exactly `batch * rows` f32 values.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] under the same conditions as
/// [`i4x8_batched_matmul_f32_scaled_via`].
pub fn i4x8_batched_matmul_f32_scaled_via_with_scratch_into(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    operands: &PackedI4BatchedMatmul<'_>,
    scratch: &mut QuantizedBatchedMatmulGpuScratch,
    out: &mut Vec<f32>,
) -> Result<(), SemanticExecutionError> {
    let QuantizedBatchedMatmulGpuScratch {
        inputs,
        program_cache,
    } = scratch;
    dispatch_packed_batched_matmul(
        "i4x8_batched_matmul_f32_scaled_via",
        dispatcher,
        policy,
        operands,
        inputs,
        program_cache,
        None,
        || {
            i4x8_batched_matmul_f32_scaled(
                "weights",
                "activations",
                "row_scales",
                "batch_scales",
                "out",
                operands.batch,
                operands.rows,
                operands.cols,
            )
        },
        out,
    )
}
