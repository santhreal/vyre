//! Parity test: vyre-primitives predicate node_kind_eq + literal_of
//! match CPU oracles.

#![cfg(test)]

mod common;

use common::{bytes_u32, live_dispatcher, u32_bytes};
use vyre::DispatchConfig;
use vyre_driver_cuda::CudaBackend;
use vyre_primitives::predicate::literal_of::{cpu_ref as literal_cpu, literal_of};
use vyre_primitives::predicate::node_kind_eq::{cpu_ref as kind_eq_cpu, node_kind_eq};

fn run_node_kind_eq(backend: &CudaBackend, nodes: &[u32], kind: u32) -> Vec<u32> {
    let n = nodes.len() as u32;
    let words = (n.div_ceil(32)).max(1);
    let program = node_kind_eq("nodes", "nodeset", n, kind);
    let inputs: Vec<Vec<u8>> = vec![u32_bytes(nodes), vec![0u8; words as usize * 4]];
    let mut config = DispatchConfig::default();
    let workgroup_x = 256u32;
    let grid_x = ((n + workgroup_x - 1) / workgroup_x).max(1);
    config.grid_override = Some([grid_x, 1, 1]);
    let outputs = backend
        .dispatch(&program, &inputs, &config)
        .expect("dispatch");
    let mut out = bytes_u32(&outputs[0]);
    out.truncate(words as usize);
    out
}

#[test]
fn cuda_node_kind_eq_basic() {
    let backend = live_dispatcher();
    let nodes = vec![1u32, 2, 1, 3, 1, 4];
    let kind = 1u32;
    let cpu = kind_eq_cpu(&nodes, kind);
    let gpu = run_node_kind_eq(&backend, &nodes, kind);
    assert_eq!(gpu, cpu);
    // Bits 0, 2, 4 should be set.
    assert_eq!(gpu, vec![0b010101]);
}

#[test]
fn cuda_node_kind_eq_no_matches() {
    let backend = live_dispatcher();
    let nodes = vec![1u32, 2, 3, 4];
    let kind = 99u32;
    let cpu = kind_eq_cpu(&nodes, kind);
    let gpu = run_node_kind_eq(&backend, &nodes, kind);
    assert_eq!(gpu, cpu);
    assert_eq!(gpu, vec![0u32]);
}

#[test]
fn cuda_node_kind_eq_all_match() {
    let backend = live_dispatcher();
    let nodes = vec![5u32; 8];
    let cpu = kind_eq_cpu(&nodes, 5);
    let gpu = run_node_kind_eq(&backend, &nodes, 5);
    assert_eq!(gpu, cpu);
    assert_eq!(gpu, vec![0b1111_1111]);
}

fn run_literal_of(backend: &CudaBackend, nodes: &[u32]) -> Vec<u32> {
    let n = nodes.len() as u32;
    let words = (n.div_ceil(32)).max(1);
    let program = literal_of("nodes", "nodeset", n);
    let inputs: Vec<Vec<u8>> = vec![u32_bytes(nodes), vec![0u8; words as usize * 4]];
    let mut config = DispatchConfig::default();
    let workgroup_x = 256u32;
    let grid_x = ((n + workgroup_x - 1) / workgroup_x).max(1);
    config.grid_override = Some([grid_x, 1, 1]);
    let outputs = backend
        .dispatch(&program, &inputs, &config)
        .expect("dispatch");
    let mut out = bytes_u32(&outputs[0]);
    out.truncate(words as usize);
    out
}

#[test]
fn cuda_literal_of_matches_cpu() {
    let backend = live_dispatcher();
    let nodes = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
    let cpu = literal_cpu(&nodes);
    let gpu = run_literal_of(&backend, &nodes);
    assert_eq!(gpu, cpu);
}
