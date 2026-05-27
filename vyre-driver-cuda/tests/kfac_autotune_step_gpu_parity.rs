//! Parity test: GPU K-FAC block inverse matches CPU oracle.

#![cfg(test)]

mod common;

use common::{live_dispatcher, CudaOptimizerDispatcher};
use vyre_primitives::math::kfac_block_inverse::cpu_ref;
use vyre_self_substrate::kfac_autotune_step::kfac_autotune_step_via;

fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-3 * (1.0 + a.abs() + b.abs())
}

fn approx_slice_eq(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(approx_eq(x, y), "divergence at index {i}: gpu={x}, cpu={y}");
    }
}

#[test]
fn cuda_kfac_diagonal_block() {
    let backend = live_dispatcher();
    let dispatcher = CudaOptimizerDispatcher { backend: &backend };
    // Diagonal [2, 4]: inverse [0.5, 0.25].
    let blocks_in = vec![2.0f32, 0.0, 0.0, 4.0];
    let cpu = cpu_ref(&blocks_in, 1, 2);
    let gpu = kfac_autotune_step_via(&dispatcher, &blocks_in, 1, 2).expect("dispatch");
    approx_slice_eq(&gpu, &cpu);
}

#[test]
fn cuda_kfac_dense_block() {
    let backend = live_dispatcher();
    let dispatcher = CudaOptimizerDispatcher { backend: &backend };
    let blocks_in = vec![4.0f32, 3.0, 3.0, 2.0];
    let cpu = cpu_ref(&blocks_in, 1, 2);
    let gpu = kfac_autotune_step_via(&dispatcher, &blocks_in, 1, 2).expect("dispatch");
    approx_slice_eq(&gpu, &cpu);
}

#[test]
fn cuda_kfac_two_blocks() {
    let backend = live_dispatcher();
    let dispatcher = CudaOptimizerDispatcher { backend: &backend };
    // Block 0: identity. Block 1: diagonal [2, 4].
    let blocks_in = vec![1.0f32, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 4.0];
    let cpu = cpu_ref(&blocks_in, 2, 2);
    let gpu = kfac_autotune_step_via(&dispatcher, &blocks_in, 2, 2).expect("dispatch");
    approx_slice_eq(&gpu, &cpu);
}
