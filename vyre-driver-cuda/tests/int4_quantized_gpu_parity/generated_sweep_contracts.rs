use super::*;

use vyre_libs::math::quantized::{
    i4x8_batched_matmul_f32_scaled, i4x8_batched_matmul_top1_f32_scaled,
    i4x8_batched_matvec_f32_scaled, i4x8_dot_f32_scaled, i4x8_dot_i32, i4x8_matvec_f32_scaled,
};
use vyre_reference::composition_witness::{
    i4x8_batched_matmul_f32_scaled_witness as i4x8_batched_matmul_f32_scaled_cpu,
    i4x8_batched_matmul_top1_f32_scaled_witness as i4x8_batched_matmul_top1_f32_scaled_cpu,
    i4x8_batched_matvec_f32_scaled_witness as i4x8_batched_matvec_f32_scaled_cpu,
    i4x8_dot_f32_scaled_witness as i4x8_dot_f32_scaled_cpu,
    i4x8_dot_i32_witness as i4x8_dot_i32_cpu,
    i4x8_matvec_f32_scaled_witness as i4x8_matvec_f32_scaled_cpu,
};

#[test]
fn generated_cuda_int4_release_parity_sweeps_boundary_shapes() {
    let backend = cuda_backend();

    for seed in 0_u32..8 {
        for lane_count in GENERATED_DOT_LANE_COUNTS {
            let lhs_packed = pack_i4x8_cpu(&generated_i4_values(
                lane_count as usize,
                seed.wrapping_mul(17) + 1,
            ));
            let rhs_packed = pack_i4x8_cpu(&generated_i4_values(
                lane_count as usize,
                seed.wrapping_mul(31) + 7,
            ));

            let dot_program = i4x8_dot_i32("lhs", "rhs", "out", lane_count);
            let dot_outputs = backend
                .dispatch(
                    &dot_program,
                    &[pack_u32_slice(&lhs_packed), pack_u32_slice(&rhs_packed)],
                    &DispatchConfig::default(),
                )
                .expect("Fix: generated CUDA INT4 i32 dot parity must dispatch on live GPU.");
            assert_eq!(
                read_i32(&dot_outputs[0]),
                i4x8_dot_i32_cpu(&lhs_packed, &rhs_packed, lane_count),
                "generated i32 dot seed={seed} lane_count={lane_count}"
            );

            let lhs_scale = 0.0625_f32 * (1 + (seed % 7)) as f32;
            let rhs_scale = 0.03125_f32 * (1 + (lane_count % 9)) as f32;
            let program =
                i4x8_dot_f32_scaled("lhs", "rhs", "lhs_scale", "rhs_scale", "out", lane_count);
            let outputs = backend
                .dispatch(
                    &program,
                    &[
                        pack_u32_slice(&lhs_packed),
                        pack_u32_slice(&rhs_packed),
                        pack_f32_slice(&[lhs_scale]),
                        pack_f32_slice(&[rhs_scale]),
                    ],
                    &DispatchConfig::default(),
                )
                .expect("Fix: generated CUDA INT4 dot parity must dispatch on live GPU.");
            assert_eq!(
                read_f32(&outputs[0]).to_bits(),
                i4x8_dot_f32_scaled_cpu(&lhs_packed, &rhs_packed, lhs_scale, rhs_scale, lane_count)
                    .to_bits(),
                "generated dot seed={seed} lane_count={lane_count}"
            );
        }
    }

    for seed in 0_u32..6 {
        for (rows, cols) in GENERATED_MATVEC_SHAPES {
            let weights_packed =
                pack_i4_matrix_rows(&generated_i4_rows(rows, cols, seed.wrapping_mul(101) + 11));
            let x = generated_f32_values(cols as usize, seed.wrapping_mul(109) + rows + cols);
            let scales = generated_positive_scales(rows as usize, seed + rows * 13 + cols);
            let program = i4x8_matvec_f32_scaled("weights", "x", "scales", "out", rows, cols);
            let outputs = backend
                .dispatch(
                    &program,
                    &[
                        pack_u32_slice(&weights_packed),
                        pack_f32_slice(&x),
                        pack_f32_slice(&scales),
                    ],
                    &DispatchConfig::default(),
                )
                .expect("Fix: generated CUDA INT4 matvec parity must dispatch on live GPU.");
            assert_eq!(
                f32_bits(&read_f32_lanes(&outputs[0], rows as usize)),
                f32_bits(&i4x8_matvec_f32_scaled_cpu(
                    &weights_packed,
                    &x,
                    &scales,
                    rows,
                    cols
                )),
                "generated matvec seed={seed} rows={rows} cols={cols}"
            );
        }
    }

    for seed in 0_u32..5 {
        for (batch, rows, cols) in GENERATED_BATCHED_SHAPES {
            let weights_packed =
                pack_i4_matrix_rows(&generated_i4_rows(rows, cols, seed.wrapping_mul(127) + 19));
            let x_batches =
                generated_f32_values((batch * cols) as usize, seed.wrapping_mul(131) + 23);
            let scales = generated_positive_scales(rows as usize, seed + 29);
            let program =
                i4x8_batched_matvec_f32_scaled("weights", "x", "scales", "out", batch, rows, cols);
            let outputs = backend
                .dispatch(
                    &program,
                    &[
                        pack_u32_slice(&weights_packed),
                        pack_f32_slice(&x_batches),
                        pack_f32_slice(&scales),
                    ],
                    &DispatchConfig::default(),
                )
                .expect("Fix: generated CUDA INT4 batched matvec parity must dispatch.");
            assert_eq!(
                f32_bits(&read_f32_lanes(&outputs[0], (batch * rows) as usize)),
                f32_bits(&i4x8_batched_matvec_f32_scaled_cpu(
                    &weights_packed,
                    &x_batches,
                    &scales,
                    batch,
                    rows,
                    cols
                )),
                "generated batched matvec seed={seed} batch={batch} rows={rows} cols={cols}"
            );
        }
    }

    for seed in 0_u32..5 {
        for (batch, rows, cols) in GENERATED_BATCHED_SHAPES {
            let inputs = generated_batched_matmul_inputs(batch, rows, cols, seed);
            let bindings = inputs.bindings();
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
                .dispatch(&program, &bindings, &DispatchConfig::default())
                .expect("Fix: generated CUDA INT4 batched matmul parity must dispatch.");
            assert_eq!(
                f32_bits(&read_f32_lanes(&outputs[0], (batch * rows) as usize)),
                f32_bits(&i4x8_batched_matmul_f32_scaled_cpu(
                    &inputs.weights_packed,
                    &inputs.activations_packed,
                    &inputs.row_scales,
                    &inputs.batch_scales,
                    batch,
                    rows,
                    cols
                )),
                "generated batched matmul seed={seed} batch={batch} rows={rows} cols={cols}"
            );

            let top1_program = i4x8_batched_matmul_top1_f32_scaled(
                "weights",
                "activations",
                "row_scales",
                "batch_scales",
                "out",
                batch,
                rows,
                cols,
            );
            let top1_outputs = backend
                .dispatch(&top1_program, &bindings, &DispatchConfig::default())
                .expect("Fix: generated CUDA INT4 top1 parity must dispatch.");
            let (expected_scores, expected_indices) = i4x8_batched_matmul_top1_f32_scaled_cpu(
                &inputs.weights_packed,
                &inputs.activations_packed,
                &inputs.row_scales,
                &inputs.batch_scales,
                batch,
                rows,
                cols,
            );
            let (actual_scores, actual_indices) = split_top1(&top1_outputs[0], batch);
            assert_eq!(
                f32_bits(&actual_scores),
                f32_bits(&expected_scores),
                "generated top1 score seed={seed} batch={batch} rows={rows} cols={cols}"
            );
            assert_eq!(
                actual_indices, expected_indices,
                "generated top1 index seed={seed} batch={batch} rows={rows} cols={cols}"
            );
        }
    }
}
