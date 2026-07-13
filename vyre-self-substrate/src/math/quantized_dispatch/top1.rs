use super::*;

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
    dispatcher: &impl OptimizerDispatcher,
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
    dispatcher: &impl OptimizerDispatcher,
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
    validate_batched_packed_matmul_shape(
        "i4x8_batched_matmul_top1_f32_scaled_via",
        weights_packed,
        activation_batches_packed,
        row_scales,
        batch_scales,
        batch,
        rows,
        cols,
    )?;

    let QuantizedBatchedMatmulTop1GpuScratch {
        inputs,
        program_cache,
    } = scratch;
    let program = program_cache.get_or_insert_with((batch, rows, cols), || {
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
    });
    // Four input-consuming buffers: weights/activations/row_scales/batch_scales ReadOnly(0-3).
    // `scores` is `BufferDecl::output`(4), backend-allocated, consumes NO dispatch input. It is a
    // SINGLE `batch*2` f32 buffer SPLIT into two halves: `out[b]=best_score_b` for the first `batch`
    // words, then `out[batch+b]=cast(f32, best_index_b)` for the next `batch` (see
    // quantized/programs.rs (the kernel stores scores and indices into one `out` buffer)).
    ensure_input_slots(inputs, 4);
    write_u32_slice_le_bytes(&mut inputs[0], weights_packed);
    write_u32_slice_le_bytes(&mut inputs[1], activation_batches_packed);
    write_f32_slice_le_bytes(&mut inputs[2], row_scales);
    write_f32_slice_le_bytes(&mut inputs[3], batch_scales);

    let outputs =
        dispatcher.dispatch(program, &inputs[..4], Some([ceil_div_u32(batch, 64), 1, 1]))?;
    // The kernel writes ONE `batch*2` f32 buffer: scores in `[0, batch)`, indices-as-f32 in
    // `[batch, 2*batch)`. Split it into the public `(scores, indices)` contract.
    let packed = expect_one_output("i4x8_batched_matmul_top1_f32_scaled_via", &outputs)?;
    let mut values = Vec::new();
    decode_f32_output_exact(
        packed,
        batch as usize * 2,
        "i4x8_batched_matmul_top1_f32_scaled_via",
        &mut values,
    )?;
    let batch = batch as usize;
    scores_out.clear();
    indices_out.clear();
    scores_out.reserve(batch);
    indices_out.reserve(batch);
    for b in 0..batch {
        scores_out.push(values[b]);
        // The kernel stored `cast(f32, best_index)`; recover the integer index from that exact f32.
        indices_out.push(values[batch + b] as u32);
    }
    Ok(())
}
