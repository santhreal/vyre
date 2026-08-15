use super::*;

#[test]
fn i4x8_batched_matmul_f32_scaled_via_dispatches_boundary_batches() {
    let batch = 2_u32;
    let rows = 3_u32;
    let cols = 9_u32;
    let weights = pack_i4_rows(&[
        &[-8, -7, -1, 0, 1, 2, 6, 7, -3],
        &[7, 6, 2, 1, 0, -1, -7, -8, 3],
        &[-4, 5, -6, 4, -2, 3, -5, 2, 1],
    ]);
    let activations = pack_i4_rows(&[
        &[7, 5, 3, 1, -1, -3, -5, -7, 6],
        &[-8, -6, -4, -2, 0, 2, 4, 6, 7],
    ]);
    let row_scales = [0.125, 0.25, 0.5];
    let batch_scales = [0.25, 0.375];

    let out = i4x8_batched_matmul_f32_scaled_via(
        &QuantizedBatchedMatmulDispatcher,
        &weights,
        &activations,
        &row_scales,
        &batch_scales,
        batch,
        rows,
        cols,
    )
    .expect("Fix: fake batched matmul dispatcher must complete scaled INT4 matmul without a backend");
    let expected = i4x8_batched_matmul_f32_scaled_cpu(
        &weights,
        &activations,
        &row_scales,
        &batch_scales,
        batch,
        rows,
        cols,
    );

    assert_eq!(out.len(), (batch * rows) as usize);
    assert_f32_bits_eq(&out, &expected, "INT4 batched matmul");
}

#[test]
fn i4x8_batched_matmul_f32_scaled_via_reuses_cached_program_for_same_shape() {
    let rows = 2_u32;
    let cols = 8_u32;
    let weights = pack_i4_rows(&[&[-8, -1, 0, 7, 3, -3, 6, -6], &[7, 1, -1, -8, 2, -2, 5, -5]]);
    let activations = pack_i4_rows(&[&[7, 5, 3, 1, -1, -3, -5, -7], &[-8, -6, -4, -2, 0, 2, 4, 6]]);
    let changed_activations = pack_i4_rows(&[
        &[7, 5, 3, 1, -1, -3, -5, -7],
        &[-8, -6, -4, -2, 0, 2, 4, 6],
        &[1, -1, 2, -2, 3, -3, 4, -4],
    ]);
    let row_scales = [0.25, 0.5];
    let batch_scales = [0.125, 0.375, 0.625];
    let mut out = Vec::new();

    assert_program_cache_keys_on_shape(
        "INT4 batched matmul",
        "batch/rows/cols",
        |scratch: &QuantizedBatchedMatmulGpuScratch| scratch.program_cache.builds(),
        |scratch, changed| {
            let (activations, batch_scales, batch) = if changed {
                (&changed_activations, &batch_scales[..], 3)
            } else {
                (&activations, &batch_scales[..2], 2)
            };
            i4x8_batched_matmul_f32_scaled_via_with_scratch_into(
                &QuantizedBatchedMatmulDispatcher,
                &weights,
                activations,
                &row_scales,
                batch_scales,
                batch,
                rows,
                cols,
                scratch,
                &mut out,
            )
            .expect(
                "Fix: fake batched matmul dispatcher must complete every shape this cache test drives",
            );
        },
    );
}

#[test]
fn i4x8_batched_matmul_f32_scaled_via_rejects_shape_errors_before_dispatch() {
    assert_rejects_batched_shape_errors(
        |weights, activations, row_scales, batch_scales, batch, rows, cols| {
            i4x8_batched_matmul_f32_scaled_via(
                &QuantizedBatchedMatmulDispatcher,
                weights,
                activations,
                row_scales,
                batch_scales,
                batch,
                rows,
                cols,
            )
        },
    );
}
