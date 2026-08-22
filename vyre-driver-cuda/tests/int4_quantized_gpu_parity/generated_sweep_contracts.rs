use super::*;

use vyre_reference::composition_witness::i4x8_dot_i32_witness as i4x8_dot_i32_cpu;

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

            let dot_actual = dispatch_i4_dot_i32(&backend, &lhs_packed, &rhs_packed, lane_count);
            assert_eq!(
                dot_actual,
                i4x8_dot_i32_cpu(&lhs_packed, &rhs_packed, lane_count),
                "generated i32 dot seed={seed} lane_count={lane_count}"
            );

            let lhs_scale = 0.0625_f32 * (1 + (seed % 7)) as f32;
            let rhs_scale = 0.03125_f32 * (1 + (lane_count % 9)) as f32;
            assert_dot_f32_scaled_parity(
                &backend,
                &lhs_packed,
                &rhs_packed,
                lhs_scale,
                rhs_scale,
                lane_count,
                "generated dot",
            );
        }
    }

    for seed in 0_u32..6 {
        for (rows, cols) in GENERATED_MATVEC_SHAPES {
            let weights_packed =
                pack_i4_matrix_rows(&generated_i4_rows(rows, cols, seed.wrapping_mul(101) + 11));
            let x = generated_f32_values(cols as usize, seed.wrapping_mul(109) + rows + cols);
            let scales = generated_positive_scales(rows as usize, seed + rows * 13 + cols);
            assert_matvec_parity(
                &backend,
                &weights_packed,
                &x,
                &scales,
                rows,
                cols,
                "generated matvec",
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
            assert_batched_matvec_parity(
                &backend,
                &weights_packed,
                &x_batches,
                &scales,
                (batch, rows, cols),
                "generated batched matvec",
            );
        }
    }

    for seed in 0_u32..5 {
        for (batch, rows, cols) in GENERATED_BATCHED_SHAPES {
            let inputs = generated_batched_matmul_inputs(batch, rows, cols, seed);
            assert_batched_matmul_parity(
                &backend,
                &inputs,
                batch,
                rows,
                cols,
                "generated batched matmul",
            );

            assert_batched_matmul_top1_parity(
                &backend,
                &inputs,
                batch,
                rows,
                cols,
                "generated top1",
            );
        }
    }
}
