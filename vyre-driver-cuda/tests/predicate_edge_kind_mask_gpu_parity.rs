//! Parity test: `vyre_libs::predicate::edge` on CUDA matches its CPU
//! reference for bare CSR forward traversal under an edge-kind mask.

#![cfg(all(test, feature = "device-tests"))]

mod harness;

use harness::{bytes_u32, csr_traversal_inputs, with_live_backend};
use vyre_driver::DispatchConfig;
use vyre_libs::graph::program_graph::ProgramGraphShape;
use vyre_libs::predicate::edge::edge;
use vyre_libs::predicate::edge_kind;
use vyre_reference::composition_witness::csr_forward_traverse_witness;

fn run_edge(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier: &[u32],
    allow_mask: u32,
) -> Vec<u32> {
    let words = node_count.div_ceil(32).max(1);
    let edge_count = edge_targets.len() as u32;
    let program = edge(
        ProgramGraphShape::new(node_count, edge_count.max(1)),
        "frontier_in",
        "frontier_out",
        allow_mask,
    );
    let inputs = csr_traversal_inputs(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        frontier,
    );
    let config = DispatchConfig::default();
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
    let cpu = csr_forward_traverse_witness(
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
    let cpu = csr_forward_traverse_witness(
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

    let cpu = csr_forward_traverse_witness(
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
