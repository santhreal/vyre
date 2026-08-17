//! Parity test: GPU K-FAC block inverse matches CPU oracle.

#![cfg(test)]

mod harness;

use harness::with_cuda_optimizer_dispatcher;
use vyre_libs::solvers::kfac_autotune_step::kfac_autotune_step_via;
fn cpu_ref(blocks_in: &[f32], blocks: u32, dim: u32) -> Vec<f32> {
    let d = dim as usize;
    let block_size = d * d;
    let mut out = vec![0.0f32; (blocks as usize) * block_size];
    for b in 0..blocks as usize {
        let block_slice = &blocks_in[b * block_size..(b + 1) * block_size];
        if d == 2 {
            let det = block_slice[0] * block_slice[3] - block_slice[1] * block_slice[2];
            let inv_det = if det.abs() > 1e-12 { 1.0 / det } else { 0.0 };
            out[b * 4] = block_slice[3] * inv_det;
            out[b * 4 + 1] = -block_slice[1] * inv_det;
            out[b * 4 + 2] = -block_slice[2] * inv_det;
            out[b * 4 + 3] = block_slice[0] * inv_det;
        } else {
            for i in 0..d {
                out[b * block_size + i * d + i] = 1.0;
            }
        }
    }
    out
}

fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-3 * (1.0 + a.abs() + b.abs())
}

fn approx_slice_eq(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(approx_eq(x, y), "divergence at index {i}: gpu={x}, cpu={y}");
    }
}

fn assert_kfac_autotune_step_matches_reference(
    label: &str,
    blocks_in: &[f32],
    blocks: u32,
    dim: u32,
) {
    let cpu = cpu_ref(blocks_in, blocks, dim);
    with_cuda_optimizer_dispatcher(label, |dispatcher| {
        let gpu = kfac_autotune_step_via(dispatcher, blocks_in, blocks, dim).expect("dispatch");
        approx_slice_eq(&gpu, &cpu);
    });
}

#[test]
fn cuda_kfac_diagonal_block() {
    // Diagonal [2, 4]: inverse [0.5, 0.25].
    let blocks_in = vec![2.0f32, 0.0, 0.0, 4.0];
    assert_kfac_autotune_step_matches_reference("diagonal block", &blocks_in, 1, 2);
}

#[test]
fn cuda_kfac_dense_block() {
    let blocks_in = vec![4.0f32, 3.0, 3.0, 2.0];
    assert_kfac_autotune_step_matches_reference("dense block", &blocks_in, 1, 2);
}

#[test]
fn cuda_kfac_two_blocks() {
    // Block 0: identity. Block 1: diagonal [2, 4].
    let blocks_in = vec![1.0f32, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 4.0];
    assert_kfac_autotune_step_matches_reference("two blocks", &blocks_in, 2, 2);
}
