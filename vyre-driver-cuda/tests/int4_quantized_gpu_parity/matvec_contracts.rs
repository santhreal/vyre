use super::*;

#[test]
fn cuda_dispatch_matches_packed_int4_scaled_matvec_oracle() {
    let backend = cuda_backend();

    for (rows, cols) in MATVEC_SHAPES {
        let weights_packed = pack_i4_matrix_rows(&cycled_rows(&WEIGHT_PATTERN, rows, cols, 3));
        let x = (0..cols)
            .map(|col| (col % 13) as f32 * 0.125 - 0.75)
            .collect::<Vec<_>>();
        let scales = patterned_row_scales(rows);
        assert_matvec_parity(
            &backend,
            &weights_packed,
            &x,
            &scales,
            rows,
            cols,
            "patterned matvec",
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
        assert_batched_matvec_parity(
            &backend,
            &weights_packed,
            &x_batches,
            &scales,
            (batch, rows, cols),
            "patterned batched matvec",
        );
    }
}
