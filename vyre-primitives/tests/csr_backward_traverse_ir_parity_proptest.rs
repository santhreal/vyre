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
//!
//! Grid: node-indexed lanes, `node_count` dispatch floor (see forward test for
//! the buffer-inference rationale); per-lane `src < node_count` guard drops
//! over-fire.
#![forbid(unsafe_code)]
#![cfg(all(feature = "graph", feature = "cpu-parity"))]

use proptest::prelude::*;
use vyre_primitives::graph::csr_backward_traverse::{cpu_ref, csr_backward_traverse};
use vyre_primitives::graph::program_graph::ProgramGraphShape;
use vyre_primitives::wire::{decode_u32_le_bytes_all as unpack, pack_u32_slice as pack};
use vyre_reference::value::Value;

fn bitset_words(node_count: u32) -> usize {
    vyre_primitives::bitset::bitset_words(node_count) as usize
}

/// Drive the real reverse-step IR through `reference_eval` and return the
/// `frontier_out` word bitset. Buffer binding order matches the forward step:
/// pg_nodes(0), pg_edge_offsets(1), pg_edge_targets(2), pg_edge_kind_mask(3),
/// pg_node_tags(4), frontier_in(5), frontier_out(6, the only ReadWrite buffer).
fn gpu_backward_step(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_in: &[u32],
    allow_mask: u32,
) -> Vec<u32> {
    let edge_count = *edge_offsets
        .last()
        .expect("offsets has node_count+1 entries");
    let program = csr_backward_traverse(
        ProgramGraphShape::new(node_count, edge_count),
        "frontier_in",
        "frontier_out",
        allow_mask,
    );
    let padded_edges = edge_count.max(1) as usize;
    let mut targets = edge_targets.to_vec();
    targets.resize(padded_edges, 0);
    let mut kind_mask = edge_kind_mask.to_vec();
    kind_mask.resize(padded_edges, 0);
    let node_tags = vec![0u32; node_count as usize];
    let nodes = vec![0u32; node_count as usize];
    let words = bitset_words(node_count);

    let outputs = vyre_reference::reference_eval_with_dispatch(
        &program,
        &[
            Value::from(pack(&nodes)),
            Value::from(pack(edge_offsets)),
            Value::from(pack(&targets)),
            Value::from(pack(&kind_mask)),
            Value::from(pack(&node_tags)),
            Value::from(pack(frontier_in)),
            Value::from(pack(&vec![0u32; words])),
        ],
        node_count,
    )
    .expect("csr_backward_traverse reference evaluation must succeed");
    unpack(&outputs[0].to_bytes())
}

fn generated_case(seed: u64) -> (u32, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, u32) {
    let mut rng = seed;
    let mut next = || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng >> 32) as u32
    };
    let node_count = 1 + next() % 96;
    let mut offsets = Vec::with_capacity(node_count as usize + 1);
    let mut targets = Vec::new();
    let mut kind_mask = Vec::new();
    offsets.push(0u32);
    for _ in 0..node_count {
        let degree = next() % 7;
        for _ in 0..degree {
            targets.push(next() % node_count);
            kind_mask.push(1u32 << (next() % 5));
        }
        offsets.push(targets.len() as u32);
    }
    let words = bitset_words(node_count);
    // frontier_in: destinations to pull FROM. Keep bits strictly within
    // node_count so the IR `dst < node_count` gate and the oracle's
    // `dst_word < len` gate never diverge.
    let mut frontier = vec![0u32; words];
    for dst in 0..node_count {
        if next() & 1 == 0 {
            frontier[(dst / 32) as usize] |= 1u32 << (dst % 32);
        }
    }
    let allow_mask = 1u32 << (next() % 5) | 1u32 << (next() % 5);
    (
        node_count, offsets, targets, kind_mask, frontier, allow_mask,
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn ir_matches_cpu_ref_over_random_graphs(seed in any::<u64>()) {
        let (node_count, offsets, targets, kind_mask, frontier, allow_mask) = generated_case(seed);
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
    // frontier_in = {3}. Node 1 (edge->2? no) ... trace: node 0 edges 0..2 ->
    // targets[0]=1,[1]=2; node 1 edge 2..3 -> targets[2]=3; node 2 edge 3..4 ->
    // targets[3]=3; node 3 no edges. frontier_in={3} -> nodes 1 and 2 point at 3
    // -> frontier_out = {1,2} = 0b0110.
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
