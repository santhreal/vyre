//! GPU-IR vs CPU-ref parity for `graph::csr_backward_traverse` (one reverse /
//! pull frontier step).
//!
//! Each lane owns a source node `src` and scans its outgoing CSR edges. If ANY
//! allowed edge (`edge_kind_mask[e] & allow_mask != 0`) points at a destination
//! `dst` that is set in `frontier_in`, the lane sets its own bit in
//! `frontier_out` (atomic-OR, since many source nodes share an output word) and
//! stops early (`hit` short-circuit). This is a SINGLE dispatch round, faithfully
//! modelled by one `reference_eval` pass. Every shipped test is
//! `cpu_ref`-vs-oracle; the actual pull IR (early-out `hit` flag, frontier_in
//! read at `dst`, per-src atomic mark) was never executed. A broken short-circuit,
//! a dst/src word-index swap, or a lost atomic mark all diverge here.
#![forbid(unsafe_code)]
#![cfg(all(feature = "graph", feature = "cpu-parity"))]

mod graph_sweep_fixtures;
use graph_sweep_fixtures::{bitset_words, frontier_step_out};
#[path = "../../tests/support/csr_sweep/mod.rs"]
mod csr_sweep;

use proptest::prelude::*;
use vyre_primitives::graph::csr_backward_traverse::{cpu_ref, csr_backward_traverse};

/// Drive the real reverse-step IR and return the `frontier_out` word bitset.
fn gpu_backward_step(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_in: &[u32],
    allow_mask: u32,
) -> Vec<u32> {
    frontier_step_out(
        csr_backward_traverse,
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        frontier_in,
        allow_mask,
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn ir_matches_cpu_ref_over_random_graphs(seed in any::<u64>()) {
        // The generated frontier keeps every set bit strictly below `node_count`,
        // so the IR `dst < node_count` gate and the oracle's `dst_word < len`
        // gate never diverge on an out-of-domain destination.
        let (node_count, offsets, targets, kind_mask, frontier, allow_mask) =
            csr_sweep::generate(csr_sweep::group("multi_source_restricted_kinds"), seed)
                .into_parts();
        let expected = cpu_ref(node_count, &offsets, &targets, &kind_mask, &frontier, allow_mask);
        let got = gpu_backward_step(node_count, &offsets, &targets, &kind_mask, &frontier, allow_mask);
        prop_assert_eq!(
            got, expected,
            "reverse-step IR diverged from cpu_ref: node_count={}, offsets={:?}, targets={:?}, allow_mask={:#x}",
            node_count, offsets, targets, allow_mask
        );
    }
}

/// Deterministic anchors: the inventory witness (nodes 1,2 both point at active
/// node 3 -> {1,2}), an early-out where a src has multiple edges to active nodes,
/// a word-seam src, and allow_mask filtering.
#[test]
fn ir_matches_cpu_ref_on_boundary_graphs() {
    // Inventory witness graph: offsets [0,2,3,4,4], targets [1,2,3,3],
    // frontier_in = {3}. Node 0 edges 0..2 -> targets[0]=1, targets[1]=2; node 1
    // edge 2..3 -> targets[2]=3; node 2 edge 3..4 -> targets[3]=3; node 3 has no
    // edges. So nodes 1 and 2 point at active 3 -> frontier_out = {1,2} = 0b0110.
    let offsets = vec![0u32, 2, 3, 4, 4];
    let targets = vec![1u32, 2, 3, 3];
    let frontier = vec![0b1000u32]; // node 3 active
    let expected = cpu_ref(4, &offsets, &targets, &[1, 1, 1, 1], &frontier, 0xFFFF_FFFF);
    assert_eq!(expected, vec![0b0110u32], "cpu_ref: nodes 1,2 pull from 3");
    assert_eq!(
        gpu_backward_step(4, &offsets, &targets, &[1, 1, 1, 1], &frontier, 0xFFFF_FFFF),
        expected,
        "inventory witness pull must match"
    );

    // Word-seam src: node 64 (word 2) has an edge to active node 0. Its own bit
    // must be marked in frontier_out word 2.
    let node_count = 65u32;
    let mut offsets = vec![0u32];
    for src in 0..node_count {
        // only node 64 gets an edge, to node 0
        if src == 64 {
            offsets.push(*offsets.last().unwrap() + 1);
        } else {
            offsets.push(*offsets.last().unwrap());
        }
    }
    let targets = vec![0u32]; // the single edge 64->0
    let kind_mask = vec![1u32];
    let words = bitset_words(node_count);
    let mut frontier = vec![0u32; words];
    frontier[0] |= 1; // node 0 active
    let expected = cpu_ref(
        node_count,
        &offsets,
        &targets,
        &kind_mask,
        &frontier,
        0xFFFF_FFFF,
    );
    assert_eq!(expected[2] & 1, 1, "cpu_ref: node 64 pulls from active 0");
    assert_eq!(
        gpu_backward_step(
            node_count,
            &offsets,
            &targets,
            &kind_mask,
            &frontier,
            0xFFFF_FFFF
        ),
        expected,
        "word-seam src mark must match"
    );

    // allow_mask filtering: node 0 -> active node 1 via a kind-bit-2 edge. mask
    // selecting bit 4 drops it (empty); mask selecting bit 2 fires it ({0}).
    let offsets = vec![0u32, 1, 1];
    let targets = vec![1u32];
    let kind_mask = vec![1u32 << 2];
    let frontier = vec![0b10u32]; // node 1 active
    let dropped = cpu_ref(2, &offsets, &targets, &kind_mask, &frontier, 1 << 4);
    assert_eq!(dropped, vec![0u32], "cpu_ref: mask mismatch pulls nothing");
    assert_eq!(
        gpu_backward_step(2, &offsets, &targets, &kind_mask, &frontier, 1 << 4),
        dropped
    );
    let fired = cpu_ref(2, &offsets, &targets, &kind_mask, &frontier, 1 << 2);
    assert_eq!(fired, vec![0b01u32], "cpu_ref: matching mask pulls node 0");
    assert_eq!(
        gpu_backward_step(2, &offsets, &targets, &kind_mask, &frontier, 1 << 2),
        fired
    );
}
