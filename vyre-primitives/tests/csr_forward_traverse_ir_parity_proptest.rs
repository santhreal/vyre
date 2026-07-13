//! GPU-IR vs CPU-ref parity for `graph::csr_forward_traverse` (one forward
//! frontier-expansion step).
//!
//! Each lane owns a source node; for every outgoing edge whose
//! `edge_kind_mask[e] & allow_mask != 0` it computes `dst = edge_targets[e]`,
//! bounds-checks `dst < node_count`, and ATOMIC-ORs the destination bit into
//! `frontier_out`. This is a SINGLE dispatch round (transitive closure is a
//! separate bitset_fixpoint composition), so it is faithfully modelled by one
//! `reference_eval` pass, unlike the multi-iteration data-dependent fixpoints.
//! Every shipped test for this op is `cpu_ref`-vs-independent-oracle; the actual
//! scatter IR (edge-kind gate, dst bound, concurrent atomic_or into shared
//! output words) was never executed. A missing `allow_mask` gate, a dropped
//! `dst < node_count` bound, or a non-atomic OR (lost bit when two source lanes
//! set the same output word) all diverge here.
//!
//! Grid note: the interpreter infers its grid from the largest buffer; the
//! node-indexed lanes must all fire, so we pass a `node_count` dispatch FLOOR
//! (the `pg_edge_offsets` buffer is always `node_count + 1` and already forces
//! this, the floor makes it explicit and robust to sparse graphs). The
//! per-lane `src < node_count` guard drops any over-fire.
#![forbid(unsafe_code)]
#![cfg(all(feature = "graph", feature = "cpu-parity"))]

use proptest::prelude::*;
use vyre_primitives::graph::csr_forward_traverse::{cpu_ref, csr_forward_traverse};
use vyre_primitives::graph::program_graph::ProgramGraphShape;
use vyre_primitives::wire::{decode_u32_le_bytes_all as unpack, pack_u32_slice as pack};
use vyre_reference::value::Value;

fn bitset_words(node_count: u32) -> usize {
    vyre_primitives::bitset::bitset_words(node_count) as usize
}

/// Drive the real forward-step IR through `reference_eval` and return the
/// `frontier_out` word bitset.
///
/// Buffer binding order (all non-workgroup): pg_nodes(0), pg_edge_offsets(1),
/// pg_edge_targets(2), pg_edge_kind_mask(3), pg_node_tags(4), frontier_in(5),
/// frontier_out(6, the only ReadWrite buffer, fed a zeroed slot). The single
/// returned writable buffer is frontier_out.
fn gpu_forward_step(
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
    let program = csr_forward_traverse(
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
            Value::from(pack(&nodes)),             // pg_nodes
            Value::from(pack(edge_offsets)),       // pg_edge_offsets
            Value::from(pack(&targets)),           // pg_edge_targets
            Value::from(pack(&kind_mask)),         // pg_edge_kind_mask
            Value::from(pack(&node_tags)),         // pg_node_tags
            Value::from(pack(frontier_in)),        // frontier_in
            Value::from(pack(&vec![0u32; words])), // frontier_out
        ],
        node_count, // dispatch floor: one lane per source node
    )
    .expect("csr_forward_traverse reference evaluation must succeed");
    unpack(&outputs[0].to_bytes())
}

/// Random monotonic CSR layout, random per-edge kind masks, a random multi-node
/// frontier, and a random allow_mask so the kind-intersect branch fires both
/// ways.
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
            kind_mask.push(1u32 << (next() % 5)); // one of 5 edge kinds
        }
        offsets.push(targets.len() as u32);
    }
    let words = bitset_words(node_count);
    let mut frontier = vec![0u32; words];
    for src in 0..node_count {
        if next() & 1 == 0 {
            frontier[(src / 32) as usize] |= 1u32 << (src % 32);
        }
    }
    // allow_mask drawn to sometimes filter a subset of kinds (never trivially 0,
    // which would empty every frontier and make the test vacuous).
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
        let got = gpu_forward_step(node_count, &offsets, &targets, &kind_mask, &frontier, allow_mask);
        prop_assert_eq!(
            got, expected,
            "forward-step IR diverged from cpu_ref: node_count={}, offsets={:?}, targets={:?}, allow_mask={:#x}",
            node_count, offsets, targets, allow_mask
        );
    }
}

/// Deterministic anchors: word-seam scatter, allow_mask filtering, and the
/// dst-bound rejection of an out-of-range edge target.
#[test]
fn ir_matches_cpu_ref_on_boundary_graphs() {
    // 65 nodes: node 0 points at nodes 32 and 64 (crossing both word seams), so
    // frontier_out must set bits in words 1 and 2 from a single source lane.
    let node_count = 65u32;
    let offsets = {
        let mut o = vec![0u32];
        o.push(2); // node 0 has 2 edges
        for _ in 1..node_count {
            o.push(2); // every other node has 0 edges
        }
        o
    };
    let targets = vec![32u32, 64];
    let kind_mask = vec![1u32, 1];
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
    assert_eq!(
        expected[1] & (1 << 0),
        1 << 0,
        "cpu_ref: bit 32 set in word 1"
    );
    assert_eq!(
        expected[2] & (1 << 0),
        1 << 0,
        "cpu_ref: bit 64 set in word 2"
    );
    assert_eq!(
        gpu_forward_step(
            node_count,
            &offsets,
            &targets,
            &kind_mask,
            &frontier,
            0xFFFF_FFFF
        ),
        expected,
        "cross-word-seam scatter must match"
    );

    // allow_mask filters: the only edge has kind bit 2, allow_mask selects bit 4
    // -> no intersection -> empty frontier_out.
    let node_count = 4u32;
    let offsets = vec![0u32, 1, 1, 1, 1];
    let targets = vec![1u32];
    let kind_mask = vec![1u32 << 2];
    let mut frontier = vec![0u32; bitset_words(node_count)];
    frontier[0] |= 1; // node 0 active
    let filtered = cpu_ref(
        node_count,
        &offsets,
        &targets,
        &kind_mask,
        &frontier,
        1 << 4,
    );
    assert_eq!(
        filtered,
        vec![0u32],
        "cpu_ref: mask mismatch drops the edge"
    );
    assert_eq!(
        gpu_forward_step(
            node_count,
            &offsets,
            &targets,
            &kind_mask,
            &frontier,
            1 << 4
        ),
        filtered,
        "allow_mask non-intersection must drop the edge in IR too"
    );
    // Same graph, allow_mask now selects bit 2 -> the edge fires, bit 1 set.
    let passed = cpu_ref(
        node_count,
        &offsets,
        &targets,
        &kind_mask,
        &frontier,
        1 << 2,
    );
    assert_eq!(
        passed,
        vec![0b10u32],
        "cpu_ref: matching mask sets dst bit 1"
    );
    assert_eq!(
        gpu_forward_step(
            node_count,
            &offsets,
            &targets,
            &kind_mask,
            &frontier,
            1 << 2
        ),
        passed,
        "allow_mask intersection must fire the edge in IR too"
    );

    // Out-of-range dst (target == node_count): the bound gate must drop it so no
    // bit is set and no OOB write occurs.
    let oob_targets = vec![node_count]; // == node_count, out of range
    let oob = cpu_ref(
        node_count,
        &offsets,
        &oob_targets,
        &kind_mask,
        &frontier,
        1 << 2,
    );
    assert_eq!(oob, vec![0u32], "cpu_ref: OOB dst is skipped");
    assert_eq!(
        gpu_forward_step(
            node_count,
            &offsets,
            &oob_targets,
            &kind_mask,
            &frontier,
            1 << 2
        ),
        oob,
        "OOB dst must be bound-gated in IR too"
    );
}
