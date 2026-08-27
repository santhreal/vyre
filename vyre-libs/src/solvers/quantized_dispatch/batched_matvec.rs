//! Device dispatch of packed INT4 batched matrix-vector products.
//!
//! One weight matrix is reused across the batch, so the packed rows are
//! uploaded once and only the activations vary per item.

use super::*;
use vyre_megakernel::{
    execute_single_program, SemanticExecutionError, SemanticExecutionPolicy, SemanticExecutor,
};

/// Compute a batch of packed signed INT4 row-scaled matrix-vector products through the backend.
///
/// `weights_packed` is row-major with `i4_packed_words(cols)` u32 words per
/// row and is reused for every batch item. `x_batches` has `batch * cols` f32
/// values. `row_scales` has `rows` f32 values. The returned vector has
/// `batch * rows` f32 values in batch-major order.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when dimensions are zero, input shapes are wrong,
/// dispatch fails, or backend readback is malformed.
pub fn i4x8_batched_matvec_f32_scaled_via(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    weights_packed: &[u32],
    x_batches: &[f32],
    row_scales: &[f32],
    batch: u32,
    rows: u32,
    cols: u32,
) -> Result<Vec<f32>, SemanticExecutionError> {
    let mut scratch = QuantizedBatchedMatvecGpuScratch::default();
    let mut out = Vec::new();
    i4x8_batched_matvec_f32_scaled_via_with_scratch_into(
        dispatcher,
        policy,
        weights_packed,
        x_batches,
        row_scales,
        batch,
        rows,
        cols,
        &mut scratch,
        &mut out,
    )?;
    Ok(out)
}

/// Compute a batch of packed signed INT4 row-scaled matrix-vector products through caller-owned scratch.
///
/// On success, `out` contains exactly `batch * rows` f32 values.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] under the same conditions as
/// [`i4x8_batched_matvec_f32_scaled_via`].
pub fn i4x8_batched_matvec_f32_scaled_via_with_scratch_into(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    weights_packed: &[u32],
    x_batches: &[f32],
    row_scales: &[f32],
    batch: u32,
    rows: u32,
    cols: u32,
    scratch: &mut QuantizedBatchedMatvecGpuScratch,
    out: &mut Vec<f32>,
) -> Result<(), SemanticExecutionError> {
    if batch == 0 || rows == 0 || cols == 0 {
        return Err(SemanticExecutionError::InvalidRequest(format!(
        "Fix: i4x8_batched_matvec_f32_scaled_via requires batch > 0, rows > 0, and cols > 0, got batch={batch} rows={rows} cols={cols}."
    )));
    }
    let words_per_row = i4_packed_words(cols) as usize;
    let expected_weight_words = (rows as usize).checked_mul(words_per_row).ok_or_else(|| {
    SemanticExecutionError::InvalidRequest(format!(
        "Fix: i4x8_batched_matvec_f32_scaled_via weight word count overflows usize for rows={rows} cols={cols}."
    ))
})?;
    if weights_packed.len() != expected_weight_words {
        return Err(SemanticExecutionError::InvalidRequest(format!(
        "Fix: i4x8_batched_matvec_f32_scaled_via requires weights_packed.len() == rows*i4_packed_words(cols), got len={} expected={expected_weight_words} for rows={rows} cols={cols}.",
        weights_packed.len()
    )));
    }
    let expected_x = (batch as usize).checked_mul(cols as usize).ok_or_else(|| {
    SemanticExecutionError::InvalidRequest(format!(
        "Fix: i4x8_batched_matvec_f32_scaled_via x batch length overflows usize for batch={batch} cols={cols}."
    ))
})?;
    if x_batches.len() != expected_x {
        return Err(SemanticExecutionError::InvalidRequest(format!(
        "Fix: i4x8_batched_matvec_f32_scaled_via requires x_batches.len() == batch*cols, got len={} expected={expected_x} for batch={batch} cols={cols}.",
        x_batches.len()
    )));
    }
    if row_scales.len() != rows as usize {
        return Err(SemanticExecutionError::InvalidRequest(format!(
        "Fix: i4x8_batched_matvec_f32_scaled_via requires row_scales.len() == rows, got len={} rows={rows}.",
        row_scales.len()
    )));
    }
    let out_words = (batch as usize).checked_mul(rows as usize).ok_or_else(|| {
    SemanticExecutionError::InvalidRequest(format!(
        "Fix: i4x8_batched_matvec_f32_scaled_via output word count overflows usize for batch={batch} rows={rows}."
    ))
})?;
    let QuantizedBatchedMatvecGpuScratch {
        inputs,
        program_cache,
    } = scratch;
    let program = program_cache.get_or_insert_with((batch, rows, cols), || {
        i4x8_batched_matvec_f32_scaled(
            "weights",
            "x_batches",
            "row_scales",
            "out",
            batch,
            rows,
            cols,
        )
    });
    // Three input-consuming buffers: weights/x_batches/row_scales ReadOnly(0-2). `out` is
    // `BufferDecl::output`(3) (backend-allocated, consumes NO dispatch input).
    ensure_input_slots(inputs, 3);
    write_u32_slice_le_bytes(&mut inputs[0], weights_packed);
    write_f32_slice_le_bytes(&mut inputs[1], x_batches);
    write_f32_slice_le_bytes(&mut inputs[2], row_scales);

    let outputs = execute_single_program(
        dispatcher,
        crate::dispatch_buffers::HOST_WRAPPER_NODE,
        program.clone(),
        &inputs[..3],
        policy,
    )
    .map(|output| output.outputs)?;
    decode_f32_output_exact(
        expect_one_output("i4x8_batched_matvec_f32_scaled_via", &outputs)?,
        out_words,
        "i4x8_batched_matvec_f32_scaled_via",
        out,
    )
}
