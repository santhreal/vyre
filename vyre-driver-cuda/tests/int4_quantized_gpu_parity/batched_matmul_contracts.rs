#![cfg(feature = "device-tests")]

use super::*;

#[test]
fn cuda_dispatch_matches_packed_int4_batched_scaled_matmul_oracle() {
    let backend = cuda_backend();

    for (batch, rows, cols) in BATCHED_SHAPES {
        let inputs = patterned_batched_matmul_inputs(batch, rows, cols);
        assert_batched_matmul_parity(
            &backend,
            &inputs,
            batch,
            rows,
            cols,
            "patterned batched matmul",
        );
    }
}

#[test]
fn cuda_dispatch_matches_packed_int4_batched_scaled_matmul_top1_oracle() {
    let backend = cuda_backend();

    for (batch, rows, cols) in BATCHED_SHAPES {
        let inputs = patterned_batched_matmul_inputs(batch, rows, cols);
        assert_batched_matmul_top1_parity(
            &backend,
            &inputs,
            batch,
            rows,
            cols,
            "patterned batched matmul top1",
        );
    }
}
