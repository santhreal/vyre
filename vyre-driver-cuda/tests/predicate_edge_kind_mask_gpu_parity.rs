//! Parity test: `vyre_primitives::predicate::edge` on CUDA matches its CPU
//! reference for bare CSR forward traversal under an edge-kind mask.

#![cfg(test)]

mod common;

use common::{bytes_u32, u32_bytes, with_live_backend};
use vyre_driver::DispatchConfig;
use vyre_primitives::graph::csr_forward_traverse::csr_forward_traverse_dispatch_grid;
use vyre_primitives::graph::program_graph::ProgramGraphShape;
use vyre_primitives::predicate::edge::{cpu_ref as edge_cpu, edge};
use vyre_primitives::predicate::edge_kind;

fn run_edge(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier: &[u32],
    allow_mask: u32,
) -> Vec<u32> {
    let words = node_count.div_ceil(32).max(1);
    let pg_nodes = vec![0u32; node_count as usize];
    let pg_node_tags = vec![0u32; node_count as usize];
    let edge_count = edge_targets.len() as u32;
    let program = edge(
        ProgramGraphShape::new(node_count, edge_count.max(1)),
        "frontier_in",
        "frontier_out",
        allow_mask,
    );
    let inputs: Vec<Vec<u8>> = vec![
        u32_bytes(&pg_nodes),
        u32_bytes(edge_offsets),
        u32_bytes(edge_targets),
        u32_bytes(edge_kind_mask),
        u32_bytes(&pg_node_tags),
        u32_bytes(frontier),
        vec![0u8; words as usize * 4],
    ];
    let mut config = DispatchConfig::default();
    config.grid_override = Some(csr_forward_traverse_dispatch_grid(node_count));
    let outputs = with_live_backend("predicate edge batch", |backend| {
        backend
            .dispatch(&program, &inputs, &config)
            .unwrap_or_else(|error| panic!("Fix: CUDA predicate-edge dispatch failed: {error}"))
    });
    let mut out = bytes_u32(&outputs[0]);
    out.truncate(words as usize);
    out
}

#[test]
fn cuda_predicate_edge_one_step() {
    // 0 -> 1 via ASSIGNMENT.
    let edge_offsets = vec![0u32, 1, 1];
    let edge_targets = vec![1u32];
    let edge_kind_mask = vec![edge_kind::ASSIGNMENT];
    let frontier = vec![0b01u32];
    let allow = edge_kind::ASSIGNMENT;
    let cpu = edge_cpu(
        2,
        &edge_offsets,
        &edge_targets,
        &edge_kind_mask,
        &frontier,
        allow,
    );
    let gpu = run_edge(
        2,
        &edge_offsets,
        &edge_targets,
        &edge_kind_mask,
        &frontier,
        allow,
    );
    assert_eq!(gpu, cpu);
    assert_eq!(gpu, vec![0b10u32]);
}

#[test]
fn cuda_predicate_edge_kind_mask_skips() {
    let edge_offsets = vec![0u32, 1, 1];
    let edge_targets = vec![1u32];
    let edge_kind_mask = vec![edge_kind::ASSIGNMENT];
    let frontier = vec![0b01u32];
    let allow = edge_kind::CALL_ARG;
    let cpu = edge_cpu(
        2,
        &edge_offsets,
        &edge_targets,
        &edge_kind_mask,
        &frontier,
        allow,
    );
    let gpu = run_edge(
        2,
        &edge_offsets,
        &edge_targets,
        &edge_kind_mask,
        &frontier,
        allow,
    );
    assert_eq!(gpu, cpu);
    assert_eq!(gpu, vec![0u32]);
}

#[test]
fn cuda_predicate_edge_reaches_source_past_first_workgroup() {
    let node_count = 513u32;
    let words = node_count.div_ceil(32) as usize;
    let mut edge_offsets = vec![0u32; node_count as usize + 1];
    for offset in edge_offsets.iter_mut().skip(301) {
        *offset = 1;
    }
    let edge_targets = vec![512u32];
    let edge_kind_mask = vec![edge_kind::ASSIGNMENT];
    let mut frontier = vec![0u32; words];
    frontier[300 / 32] |= 1u32 << (300 % 32);
    let allow = edge_kind::ASSIGNMENT;

    let cpu = edge_cpu(
        node_count,
        &edge_offsets,
        &edge_targets,
        &edge_kind_mask,
        &frontier,
        allow,
    );
    let gpu = run_edge(
        node_count,
        &edge_offsets,
        &edge_targets,
        &edge_kind_mask,
        &frontier,
        allow,
    );

    let mut expected = vec![0u32; words];
    expected[512 / 32] |= 1u32 << (512 % 32);
    assert_eq!(gpu, cpu);
    assert_eq!(gpu, expected);
}
