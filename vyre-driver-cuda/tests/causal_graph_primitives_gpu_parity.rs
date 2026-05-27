//! Parity test: vyre-primitives causal-graph primitives (adjustment_set,
//! do_calculus do_intervention_delete_incoming) match CPU oracles.

#![cfg(test)]

mod common;

use common::{bytes_u32, live_dispatcher, u32_bytes};
use vyre::DispatchConfig;
use vyre_driver_cuda::CudaBackend;
use vyre_primitives::graph::adjustment_set::{
    backdoor_descendants_check, backdoor_descendants_check_cpu,
};
use vyre_primitives::graph::do_calculus::{
    do_intervention_delete_incoming, do_intervention_delete_incoming_cpu,
};

fn run_backdoor_check(
    backend: &CudaBackend,
    candidate_z: &[u32],
    descendants_of_x: &[u32],
    n: u32,
) -> u32 {
    let program = backdoor_descendants_check("z", "dx", "out", n);
    let inputs: Vec<Vec<u8>> = vec![
        u32_bytes(candidate_z),
        u32_bytes(descendants_of_x),
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
fn cuda_backdoor_check_violation_when_overlap() {
    let backend = live_dispatcher();
    let z = vec![0u32, 1, 0, 1];
    let dx = vec![0u32, 0, 0, 1];
    let cpu = backdoor_descendants_check_cpu(&z, &dx);
    let gpu = run_backdoor_check(&backend, &z, &dx, 4);
    assert_eq!(gpu == 1, cpu);
    assert_eq!(gpu, 1);
}

#[test]
fn cuda_backdoor_check_no_violation_disjoint() {
    let backend = live_dispatcher();
    let z = vec![1u32, 0, 1, 0];
    let dx = vec![0u32, 1, 0, 1];
    let cpu = backdoor_descendants_check_cpu(&z, &dx);
    let gpu = run_backdoor_check(&backend, &z, &dx, 4);
    assert_eq!(gpu == 1, cpu);
    assert_eq!(gpu, 0);
}

fn run_intervention(backend: &CudaBackend, adjacency: &[u32], mask: &[u32], n: u32) -> Vec<u32> {
    let program = do_intervention_delete_incoming("adj", "mask", "out", n);
    let inputs: Vec<Vec<u8>> = vec![
        u32_bytes(adjacency),
        u32_bytes(mask),
        vec![0u8; (n * n) as usize * 4],
    ];
    let mut config = DispatchConfig::default();
    let workgroup_x = 256u32;
    let cells = n * n;
    let grid_x = ((cells + workgroup_x - 1) / workgroup_x).max(1);
    config.grid_override = Some([grid_x, 1, 1]);
    let outputs = backend
        .dispatch(&program, &inputs, &config)
        .expect("dispatch");
    let mut out = bytes_u32(&outputs[0]);
    out.truncate(cells as usize);
    out
}

#[test]
fn cuda_do_intervention_no_op_preserves_adjacency() {
    let backend = live_dispatcher();
    let a = vec![1u32, 2, 3, 4];
    let mask = vec![0u32, 0];
    let cpu = do_intervention_delete_incoming_cpu(&a, &mask, 2);
    let gpu = run_intervention(&backend, &a, &mask, 2);
    assert_eq!(gpu, cpu);
    assert_eq!(gpu, a);
}

#[test]
fn cuda_do_intervention_zeros_target_column() {
    let backend = live_dispatcher();
    let a = vec![1u32, 2, 3, 4];
    let mask = vec![1u32, 0];
    let cpu = do_intervention_delete_incoming_cpu(&a, &mask, 2);
    let gpu = run_intervention(&backend, &a, &mask, 2);
    assert_eq!(gpu, cpu);
    assert_eq!(gpu, vec![0u32, 2, 0, 4]);
}

#[test]
fn cuda_do_intervention_all_columns_zeroed() {
    let backend = live_dispatcher();
    let a = vec![1u32, 2, 3, 4];
    let mask = vec![1u32, 1];
    let cpu = do_intervention_delete_incoming_cpu(&a, &mask, 2);
    let gpu = run_intervention(&backend, &a, &mask, 2);
    assert_eq!(gpu, cpu);
    assert_eq!(gpu, vec![0u32; 4]);
}

#[test]
fn cuda_do_intervention_three_node_graph() {
    let backend = live_dispatcher();
    // 3x3 adj
    let a: Vec<u32> = (1..=9).collect();
    let mask = vec![0u32, 1, 0]; // intervene on node 1 → zero col 1.
    let cpu = do_intervention_delete_incoming_cpu(&a, &mask, 3);
    let gpu = run_intervention(&backend, &a, &mask, 3);
    assert_eq!(gpu, cpu);
}
