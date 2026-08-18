use super::*;

use vyre_libs::math::quantized::{i4x8_batched_matvec_f32_scaled, i4x8_matvec_f32_scaled};
use vyre_reference::composition_witness::{
    i4x8_batched_matvec_f32_scaled_witness as i4x8_batched_matvec_f32_scaled_cpu,
    i4x8_matvec_f32_scaled_witness as i4x8_matvec_f32_scaled_cpu,
};

#[test]
fn cuda_dispatch_matches_packed_int4_scaled_matvec_oracle() {
    let backend = cuda_backend();

    for (rows, cols) in MATVEC_SHAPES {
        let weights_packed = pack_i4_matrix_rows(&cycled_rows(&WEIGHT_PATTERN, rows, cols, 3));
        let x = (0..cols)
            .map(|col| (col % 13) as f32 * 0.125 - 0.75)
            .collect::<Vec<_>>();
        let scales = patterned_row_scales(rows);
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
            .expect("Fix: CUDA must execute fused packed INT4 scaled matvec without CPU fallback.");
        let expected = i4x8_matvec_f32_scaled_cpu(&weights_packed, &x, &scales, rows, cols);
        let actual = read_f32_lanes(&outputs[0], rows as usize);

        assert_eq!(
            f32_bits(&actual),
            f32_bits(&expected),
            "rows={rows} cols={cols}"
        );
    }
}

#[test]
fn cuda_dispatch_matches_packed_int4_batched_scaled_matvec_oracle() {
    let backend = cuda_backend();

    for (batch, rows, cols) in BATCHED_SHAPES {
        let weights_packed = pack_i4_matrix_rows(&cycled_rows(&WEIGHT_PATTERN, rows, cols, 5));
        let x_batches = (0..batch * cols)
            .map(|index| (index % 17) as f32 * 0.0625 - 0.5)
            .collect::<Vec<_>>();
        let scales = patterned_row_scales(rows);
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
            .expect(
                "Fix: CUDA must execute batched fused packed INT4 scaled matvec without CPU fallback.",
            );
        let expected = i4x8_batched_matvec_f32_scaled_cpu(
            &weights_packed,
            &x_batches,
            &scales,
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
