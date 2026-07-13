//! GPU-IR vs CPU-ref parity for `graph::csr_frontier_degree_sum`.
//!
//! The op launches one lane per source node; each lane in the active frontier
//! loads `edge_offsets[gid+1] - edge_offsets[gid]` and ATOMIC-ADDs its degree
//! into the single `degree_sum_out[0]` slot. Every existing test for this
//! primitive is CPU-oracle self-consistency (`csr_frontier_degree_sum_cpu`
//! sweeps) or a shape check; the ACTUAL grid-stride atomic-accumulation IR was
//! never driven through a faithful executor. A dropped tail lane, an off-by-one
//! `off_hi`/`off_lo` load, or a non-atomic add (lost update when many frontier
//! lanes hit the one output word concurrently) all diverge here and nowhere
//! else. Pins the kernel against `reference_eval`.
#![forbid(unsafe_code)]
#![cfg(all(feature = "graph", feature = "cpu-parity"))]

use proptest::prelude::*;
use vyre_primitives::graph::csr_frontier_degree_sum::{
    csr_frontier_degree_sum, csr_frontier_degree_sum_cpu,
};
use vyre_primitives::graph::program_graph::ProgramGraphShape;
use vyre_primitives::wire::{decode_u32_le_bytes_all as unpack, pack_u32_slice as pack};
use vyre_reference::value::Value;

fn bitset_words(node_count: u32) -> usize {
    vyre_primitives::bitset::bitset_words(node_count) as usize
}

/// Drive the real IR through `reference_eval` and return `degree_sum_out[0]`.
///
/// Buffer binding order (all non-workgroup): pg_nodes(0), pg_edge_offsets(1),
/// pg_edge_targets(2), pg_edge_kind_mask(3), pg_node_tags(4), frontier_in(5),
/// degree_sum_out(6, the only ReadWrite buffer, fed a zeroed slot). The single
/// returned writable buffer is degree_sum_out.
fn gpu_degree_sum(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    frontier_in: &[u32],
) -> u32 {
    let edge_count = *edge_offsets
        .last()
        .expect("offsets has node_count+1 entries");
    let program = csr_frontier_degree_sum(ProgramGraphShape::new(node_count, edge_count));
    // edge_targets / edge_kind_mask are declared with count edge_count.max(1);
    // the degree-sum kernel never reads them, but the ABI slot must be sized.
    let padded_edges = edge_count.max(1) as usize;
    let mut targets = edge_targets.to_vec();
    targets.resize(padded_edges, 0);
    let kind_mask = vec![1u32; padded_edges];
    let node_tags = vec![0u32; node_count as usize];
    let nodes = vec![0u32; node_count as usize];

    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack(&nodes)),       // pg_nodes
            Value::from(pack(edge_offsets)), // pg_edge_offsets
            Value::from(pack(&targets)),     // pg_edge_targets
            Value::from(pack(&kind_mask)),   // pg_edge_kind_mask
            Value::from(pack(&node_tags)),   // pg_node_tags
            Value::from(pack(frontier_in)),  // frontier_in
            Value::from(pack(&[0u32])),      // degree_sum_out
        ],
    )
    .expect("csr_frontier_degree_sum reference evaluation must succeed");
    let words = unpack(&outputs[0].to_bytes());
    words[0]
}

/// Build a random monotonic CSR layout plus a random multi-node frontier.
fn generated_case(seed: u64) -> (u32, Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut rng = seed;
    let mut next = || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng >> 32) as u32
    };
    let node_count = 1 + next() % 96;
    let mut offsets = Vec::with_capacity(node_count as usize + 1);
    let mut targets = Vec::new();
    offsets.push(0u32);
    for _ in 0..node_count {
        let degree = next() % 7;
        for _ in 0..degree {
            targets.push(next() % node_count);
        }
        offsets.push(targets.len() as u32);
    }
    let words = bitset_words(node_count);
    let mut frontier = vec![0u32; words];
    // Activate a random subset (each node ~50%), so many lanes race the atomic add.
    for src in 0..node_count {
        if next() & 1 == 0 {
            frontier[(src / 32) as usize] |= 1u32 << (src % 32);
        }
    }
    (node_count, offsets, targets, frontier)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn ir_matches_cpu_ref_over_random_frontiers(seed in any::<u64>()) {
        let (node_count, offsets, targets, frontier) = generated_case(seed);
        let expected = csr_frontier_degree_sum_cpu(&frontier, &offsets, node_count);
        let got = gpu_degree_sum(node_count, &offsets, &targets, &frontier);
        prop_assert_eq!(
            got, expected,
            "degree-sum IR diverged from cpu_ref: node_count={}, offsets={:?}, frontier={:?}",
            node_count, offsets, frontier
        );
    }
}

/// Deterministic anchors covering the word-seam and all-active/all-inactive
/// extremes that a random subset almost never hits.
#[test]
fn ir_matches_cpu_ref_on_boundary_frontiers() {
    // The inventory witness graph: 5 nodes, offsets [0,2,3,4,4], frontier {0,1}.
    // deg(0)=2, deg(1)=1 -> 3.
    let offsets = vec![0u32, 2, 3, 4, 4];
    let targets = vec![1u32, 2, 3, 3];
    let frontier = vec![0b0011u32];
    assert_eq!(csr_frontier_degree_sum_cpu(&frontier, &offsets, 4), 3);
    assert_eq!(gpu_degree_sum(4, &offsets, &targets, &frontier), 3);

    // Word-seam: 65 nodes so the frontier spans 3 words; every node has degree 2
    // and the ENTIRE frontier is active (nodes 0..65 across the 32/64 seams). The
    // full sum 65*2=130 requires every lane in all three words to fire and add.
    let node_count = 65u32;
    let mut offsets = Vec::with_capacity(node_count as usize + 1);
    let mut targets = Vec::new();
    offsets.push(0);
    for i in 0..node_count {
        targets.push((i + 1) % node_count);
        targets.push((i + 2) % node_count);
        offsets.push(targets.len() as u32);
    }
    let words = bitset_words(node_count);
    let mut frontier = vec![0u32; words];
    for src in 0..node_count {
        frontier[(src / 32) as usize] |= 1u32 << (src % 32);
    }
    assert_eq!(
        csr_frontier_degree_sum_cpu(&frontier, &offsets, node_count),
        130
    );
    assert_eq!(
        gpu_degree_sum(node_count, &offsets, &targets, &frontier),
        130
    );

    // Empty frontier -> 0 (no lane adds).
    let empty = vec![0u32; words];
    assert_eq!(csr_frontier_degree_sum_cpu(&empty, &offsets, node_count), 0);
    assert_eq!(gpu_degree_sum(node_count, &offsets, &targets, &empty), 0);

    // Only the LAST node active (node 64, the sole bit in word 2): a dropped tail
    // lane returns 0 instead of its degree 2.
    let mut tail = vec![0u32; words];
    tail[2] |= 1u32 << 0; // node 64 -> bit 0 of word 2
    assert_eq!(csr_frontier_degree_sum_cpu(&tail, &offsets, node_count), 2);
    assert_eq!(gpu_degree_sum(node_count, &offsets, &targets, &tail), 2);
}
