//! Parity test: vyre-primitives tensor_scc_fixpoint matches CPU oracle.

#![cfg(test)]

mod common;

use common::{bytes_u32, live_dispatcher, u32_bytes};
use vyre::DispatchConfig;
use vyre_driver_cuda::CudaBackend;
use vyre_primitives::math::tensor_scc::{cpu_ref, tensor_scc_fixpoint};

fn run(
    backend: &CudaBackend,
    matrix_rows: &[u32],
    seed_mask: u32,
    group_mask: u32,
    iteration_limit: u32,
) -> u32 {
    let program = tensor_scc_fixpoint(
        "rows",
        "seed",
        "group",
        "out",
        matrix_rows.len() as u32,
        iteration_limit,
    );
    let inputs: Vec<Vec<u8>> = vec![
        u32_bytes(matrix_rows),
        u32_bytes(&[seed_mask]),
        u32_bytes(&[group_mask]),
        vec![0u8; 4],
    ];
    let mut config = DispatchConfig::default();
    config.grid_override = Some([1, 1, 1]);
    let outputs = backend
        .dispatch(&program, &inputs, &config)
        .expect("dispatch");
    bytes_u32(&outputs[0])[0]
}

#[test]
fn cuda_tensor_scc_closes_cycle_inside_group() {
    let backend = live_dispatcher();
    let rows = [0b0010, 0b0100, 0b0001, 0b1000];
    let cpu = cpu_ref(&rows, 0b0001, 0b0111, 8);
    let gpu = run(&backend, &rows, 0b0001, 0b0111, 8);
    assert_eq!(gpu, cpu);
    assert_eq!(gpu, 0b0111);
}

#[test]
fn cuda_tensor_scc_smaller_group_caps_closure() {
    let backend = live_dispatcher();
    let rows = [0b0010, 0b0100, 0b0001, 0b1000];
    let cpu = cpu_ref(&rows, 0b0001, 0b0011, 8);
    let gpu = run(&backend, &rows, 0b0001, 0b0011, 8);
    assert_eq!(gpu, cpu);
    assert_eq!(gpu, 0b0011);
}

#[test]
fn cuda_tensor_scc_no_seed_yields_zero() {
    let backend = live_dispatcher();
    let rows = [0b0010, 0b0100, 0b0001, 0b1000];
    let cpu = cpu_ref(&rows, 0, 0b1111, 4);
    let gpu = run(&backend, &rows, 0, 0b1111, 4);
    assert_eq!(gpu, cpu);
    assert_eq!(gpu, 0);
}
