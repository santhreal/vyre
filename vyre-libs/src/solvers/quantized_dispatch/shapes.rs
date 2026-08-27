//! Shape validation and buffer marshalling shared by every packed INT4
//! dispatch in this module.
//!
//! Each entry point would otherwise repeat the same word-count arithmetic, and
//! a single wrong `ceil_div` is the difference between a short readback and a
//! silent truncation.

use crate::dispatch_buffers::{
    decode_f32_output_exact, ensure_input_slots, write_f32_slice_le_bytes, write_u32_slice_le_bytes,
};
use crate::math::quantized::i4_packed_words;
use crate::plumbing::host::program_cache::ProgramCache;
use vyre_foundation::ir::Program;
use vyre_megakernel::{
    execute_single_program, SemanticExecutionError, SemanticExecutionPolicy, SemanticExecutor,
};
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct PackedI4BatchedMatmulShape {
    pub(super) output_words: usize,
}

pub(super) fn validate_batched_packed_matmul_shape(
    context: &str,
    weights_packed: &[u32],
    activation_batches_packed: &[u32],
    row_scales: &[f32],
    batch_scales: &[f32],
    batch: u32,
    rows: u32,
    cols: u32,
) -> Result<PackedI4BatchedMatmulShape, SemanticExecutionError> {
    if batch == 0 || rows == 0 || cols == 0 {
        return Err(SemanticExecutionError::InvalidRequest(format!(
            "Fix: {context} requires batch > 0, rows > 0, and cols > 0, got batch={batch} rows={rows} cols={cols}."
        )));
    }
    let words_per_row = i4_packed_words(cols) as usize;
    let expected_weight_words = (rows as usize).checked_mul(words_per_row).ok_or_else(|| {
        SemanticExecutionError::InvalidRequest(format!(
            "Fix: {context} weight word count overflows usize for rows={rows} cols={cols}."
        ))
    })?;
    if weights_packed.len() != expected_weight_words {
        return Err(SemanticExecutionError::InvalidRequest(format!(
            "Fix: {context} requires weights_packed.len() == rows*i4_packed_words(cols), got len={} expected={expected_weight_words} for rows={rows} cols={cols}.",
            weights_packed.len()
        )));
    }
    let expected_activation_words =
        (batch as usize).checked_mul(words_per_row).ok_or_else(|| {
            SemanticExecutionError::InvalidRequest(format!(
                "Fix: {context} activation word count overflows usize for batch={batch} cols={cols}."
            ))
        })?;
    if activation_batches_packed.len() != expected_activation_words {
        return Err(SemanticExecutionError::InvalidRequest(format!(
            "Fix: {context} requires activation_batches_packed.len() == batch*i4_packed_words(cols), got len={} expected={expected_activation_words} for batch={batch} cols={cols}.",
            activation_batches_packed.len()
        )));
    }
    if row_scales.len() != rows as usize {
        return Err(SemanticExecutionError::InvalidRequest(format!(
            "Fix: {context} requires row_scales.len() == rows, got len={} rows={rows}.",
            row_scales.len()
        )));
    }
    if batch_scales.len() != batch as usize {
        return Err(SemanticExecutionError::InvalidRequest(format!(
            "Fix: {context} requires batch_scales.len() == batch, got len={} batch={batch}.",
            batch_scales.len()
        )));
    }
    let output_words = (batch as usize).checked_mul(rows as usize).ok_or_else(|| {
        SemanticExecutionError::InvalidRequest(format!(
            "Fix: {context} output word count overflows usize for batch={batch} rows={rows}."
        ))
    })?;

    Ok(PackedI4BatchedMatmulShape { output_words })
}

pub(super) fn expect_one_output<'a>(
    context: &str,
    outputs: &'a [Vec<u8>],
) -> Result<&'a [u8], SemanticExecutionError> {
    if outputs.len() != 1 {
        return Err(SemanticExecutionError::Backend(format!(
            "Fix: {context} expected exactly one output buffer, got {}.",
            outputs.len()
        )));
    }
    Ok(&outputs[0])
}
pub(super) fn write_packed_batched_matmul_inputs(
    inputs: &mut Vec<Vec<u8>>,
    weights_packed: &[u32],
    activation_batches_packed: &[u32],
    row_scales: &[f32],
    batch_scales: &[f32],
) {
    ensure_input_slots(inputs, 4);
    write_u32_slice_le_bytes(&mut inputs[0], weights_packed);
    write_u32_slice_le_bytes(&mut inputs[1], activation_batches_packed);
    write_f32_slice_le_bytes(&mut inputs[2], row_scales);
    write_f32_slice_le_bytes(&mut inputs[3], batch_scales);
}
/// Validate, materialize, dispatch, and decode one packed batched-matmul program.
pub(super) fn dispatch_packed_batched_matmul<F>(
    context: &str,
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    weights_packed: &[u32],
    activation_batches_packed: &[u32],
    row_scales: &[f32],
    batch_scales: &[f32],
    batch: u32,
    rows: u32,
    cols: u32,
    inputs: &mut Vec<Vec<u8>>,
    program_cache: &mut ProgramCache<(u32, u32, u32), Program>,
    expected_words: Option<usize>,
    build_program: F,
    out: &mut Vec<f32>,
) -> Result<(), SemanticExecutionError>
where
    F: FnOnce() -> Program,
{
    let shape = validate_batched_packed_matmul_shape(
        context,
        weights_packed,
        activation_batches_packed,
        row_scales,
        batch_scales,
        batch,
        rows,
        cols,
    )?;

    let output_words = expected_words.unwrap_or(shape.output_words);

    let program = program_cache.get_or_insert_with((batch, rows, cols), build_program);
    write_packed_batched_matmul_inputs(
        inputs,
        weights_packed,
        activation_batches_packed,
        row_scales,
        batch_scales,
    );

    let outputs = execute_single_program(
        dispatcher,
        crate::dispatch_buffers::HOST_WRAPPER_NODE,
        program.clone(),
        &inputs[..4],
        policy,
    )?
    .outputs;
    decode_f32_output_exact(
        expect_one_output(context, &outputs)?,
        output_words,
        context,
        out,
    )
}
