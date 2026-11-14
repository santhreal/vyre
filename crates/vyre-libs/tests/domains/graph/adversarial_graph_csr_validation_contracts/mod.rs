//! Adversarial / property tests for `ProgramGraphShape` and
//! `csr_forward_traverse` CPU reference.
//!
//! Coverage:
//!   - zero-edge sentinel contract
//!   • malformed CSR buffer lengths
//!   • non-monotonic edge_offsets
//!   • out-of-bounds edge targets
//!   • edge mask filtering
//!   • high node counts / multi-word bitsets
//!   • edge_offsets last-entry vs edge_count mismatch

#![cfg(feature = "graph")]
#![allow(clippy::needless_range_loop)]

use vyre_libs::graph::program_graph::{
    validate_program_graph, GraphValidationError, ProgramGraphShape,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn bitset_words(node_count: u32) -> usize {
    ((node_count + 31) / 32) as usize
}

fn zero_frontier(node_count: u32) -> Vec<u32> {
    vec![0u32; bitset_words(node_count)]
}

fn csr_cpu_ref(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_in: &[u32],
    allow_mask: u32,
) -> Vec<u32> {
    let expected_offsets = node_count as usize + 1;
    assert_eq!(
        edge_offsets.len(),
        expected_offsets,
        "csr_forward_traverse CPU oracle received {} row offsets for node_count={node_count}; Fix: pass exactly node_count + 1 CSR offsets.",
        edge_offsets.len()
    );
    let edge_count = edge_offsets[expected_offsets - 1] as usize;
    assert!(
        edge_targets.len() >= edge_count && edge_kind_mask.len() >= edge_count,
        "csr_forward_traverse CPU oracle received edge_count={edge_count} but targets_len={} kind_mask_len={}. Fix: pass complete CSR edge buffers.",
        edge_targets.len(),
        edge_kind_mask.len()
    );
    for pair in edge_offsets.windows(2) {
        assert!(
            pair[0] <= pair[1],
            "csr_forward_traverse test oracle received non-monotonic CSR offsets"
        );
    }
    vyre_reference::composition_witness::csr_forward_traverse_witness(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        frontier_in,
        allow_mask,
    )
}

// ---------------------------------------------------------------------------
// Zero-edge sentinel contract
// ---------------------------------------------------------------------------

mod edge_count_contracts;
mod frontier_shape_contracts;
mod length_contracts;
mod mask_contracts;
mod multiword_contracts;
mod offset_contracts;
mod target_contracts;
mod zero_edge_contracts;
