use super::*;

use vyre_primitives::math::quantized::{
    i4x8_batched_matmul_f32_scaled, i4x8_batched_matmul_f32_scaled_cpu,
    i4x8_batched_matmul_top1_f32_scaled, i4x8_batched_matmul_top1_f32_scaled_cpu,
};

#[test]
fn cuda_dispatch_matches_packed_int4_batched_scaled_matmul_oracle() {
    let backend = cuda_backend();

    for (batch, rows, cols) in BATCHED_SHAPES {
        let inputs = patterned_batched_matmul_inputs(batch, rows, cols);
        let program = i4x8_batched_matmul_f32_scaled(
            "weights",
            "activations",
            "row_scales",
            "batch_scales",
            "out",
            batch,
            rows,
            cols,
        );
        let outputs = backend
            .dispatch(&program, &inputs.bindings(), &DispatchConfig::default())
            .expect(
                "Fix: CUDA must execute packed-activation batched INT4 matmul without CPU fallback.",
            );
        let expected = i4x8_batched_matmul_f32_scaled_cpu(
            &inputs.weights_packed,
            &inputs.activations_packed,
            &inputs.row_scales,
            &inputs.batch_scales,
            batch,
            rows,
            cols,
        );
        let actual = read_f32_lanes(&outputs[0], (batch * rows) as usize);

        assert_eq!(
            f32_bits(&actual),
            f32_bits(&expected),
            "batch={batch} rows={rows} cols={cols}"
        );
    }
}

#[test]
fn cuda_dispatch_matches_packed_int4_batched_scaled_matmul_top1_oracle() {
    let backend = cuda_backend();

    for (batch, rows, cols) in BATCHED_SHAPES {
        let inputs = patterned_batched_matmul_inputs(batch, rows, cols);
        let program = i4x8_batched_matmul_top1_f32_scaled(
            "weights",
            "activations",
            "row_scales",
            "batch_scales",
            "out",
            batch,
            rows,
            cols,
        );
        let outputs = backend
            .dispatch(&program, &inputs.bindings(), &DispatchConfig::default())
            .expect(
                "Fix: CUDA must execute packed-activation INT4 top1 routing without CPU fallback.",
            );
        let (expected_scores, expected_indices) = i4x8_batched_matmul_top1_f32_scaled_cpu(
            &inputs.weights_packed,
            &inputs.activations_packed,
            &inputs.row_scales,
            &inputs.batch_scales,
            batch,
            rows,
            cols,
        );
        let (actual_scores, actual_indices) = split_top1(&outputs[0], batch);

        assert_eq!(
            f32_bits(&actual_scores),
            f32_bits(&expected_scores),
            "batch={batch} rows={rows} cols={cols}"
        );
        assert_eq!(
            actual_indices, expected_indices,
            "batch={batch} rows={rows} cols={cols}"
        );
    }
}
