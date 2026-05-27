//! Parity test: vyre-primitives predicate size_argument_of matches CPU oracle.

#![cfg(test)]

mod common;

use common::{bytes_u32, live_dispatcher, u32_bytes};
use vyre::DispatchConfig;
use vyre_driver_cuda::CudaBackend;
use vyre_primitives::graph::program_graph::ProgramGraphShape;
use vyre_primitives::predicate::edge_kind;
use vyre_primitives::predicate::node_kind;
use vyre_primitives::predicate::size_argument_of::{cpu_ref as size_arg_cpu, size_argument_of};

fn run(
    backend: &CudaBackend,
    node_count: u32,
    nodes: &[u32],
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_in: &[u32],
) -> Vec<u32> {
    let words = ((node_count + 31) / 32).max(1);
    let pg_node_tags = vec![0u32; node_count as usize];
    let edge_count = edge_targets.len() as u32;
    let program = size_argument_of(
        ProgramGraphShape::new(node_count, edge_count.max(1)),
        "frontier_in",
        "frontier_out",
    );
    let inputs: Vec<Vec<u8>> = vec![
        u32_bytes(nodes),
        u32_bytes(edge_offsets),
        u32_bytes(edge_targets),
        u32_bytes(edge_kind_mask),
        u32_bytes(&pg_node_tags),
        u32_bytes(frontier_in),
        // frontier_out: zero-init.
        vec![0u8; words as usize * 4],
    ];
    let mut config = DispatchConfig::default();
    let workgroup_x = 256u32;
    let grid_x = ((node_count + workgroup_x - 1) / workgroup_x).max(1);
    config.grid_override = Some([grid_x, 1, 1]);
    let outputs = backend
        .dispatch(&program, &inputs, &config)
        .expect("dispatch");
    let mut out = bytes_u32(&outputs[0]);
    out.truncate(words as usize);
    out
}

#[test]
fn cuda_size_arg_marks_callers_of_callee_set() {
    let backend = live_dispatcher();
    let nodes = vec![
        node_kind::LITERAL,
        node_kind::CALL,
        node_kind::LITERAL,
        node_kind::CALL,
    ];
    let edge_offsets = vec![0u32, 1, 2, 3, 4];
    let edge_targets = vec![1u32, 2, 3, 0];
    let edge_kind_mask = vec![edge_kind::CALL_ARG, 0, edge_kind::CALL_ARG, 0];
    let frontier_in = vec![0b1010u32];
    let cpu = size_arg_cpu(
        4,
        &nodes,
        &edge_offsets,
        &edge_targets,
        &edge_kind_mask,
        &frontier_in,
    );
    let gpu = run(
        &backend,
        4,
        &nodes,
        &edge_offsets,
        &edge_targets,
        &edge_kind_mask,
        &frontier_in,
    );
    assert_eq!(gpu, cpu);
    assert_eq!(gpu, vec![0b0101u32]);
}

#[test]
fn cuda_size_arg_no_call_arg_edges_yields_zero() {
    let backend = live_dispatcher();
    let nodes = vec![1u32, 2, 3];
    let edge_offsets = vec![0u32, 1, 2, 2];
    let edge_targets = vec![1u32, 2];
    // ASSIGNMENT, not CALL_ARG  -  should be filtered.
    let edge_kind_mask = vec![edge_kind::ASSIGNMENT, edge_kind::ASSIGNMENT];
    let frontier_in = vec![0b110u32];
    let cpu = size_arg_cpu(
        3,
        &nodes,
        &edge_offsets,
        &edge_targets,
        &edge_kind_mask,
        &frontier_in,
    );
    let gpu = run(
        &backend,
        3,
        &nodes,
        &edge_offsets,
        &edge_targets,
        &edge_kind_mask,
        &frontier_in,
    );
    assert_eq!(gpu, cpu);
    assert_eq!(gpu, vec![0u32]);
}

#[test]
fn cuda_size_arg_empty_frontier_yields_zero() {
    let backend = live_dispatcher();
    let nodes = vec![1u32, 2];
    let edge_offsets = vec![0u32, 1, 1];
    let edge_targets = vec![1u32];
    let edge_kind_mask = vec![edge_kind::CALL_ARG];
    let frontier_in = vec![0u32];
    let cpu = size_arg_cpu(
        2,
        &nodes,
        &edge_offsets,
        &edge_targets,
        &edge_kind_mask,
        &frontier_in,
    );
    let gpu = run(
        &backend,
        2,
        &nodes,
        &edge_offsets,
        &edge_targets,
        &edge_kind_mask,
        &frontier_in,
    );
    assert_eq!(gpu, cpu);
    assert_eq!(gpu, vec![0u32]);
}
