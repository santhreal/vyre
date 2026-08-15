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
#![forbid(unsafe_code)]
#![cfg(all(feature = "graph", feature = "cpu-parity"))]

mod graph_sweep_support;
use graph_sweep_support::{bitset_words, frontier_step_out};
#[path = "../../tests/support/csr_sweep/mod.rs"]
mod csr_sweep;

use proptest::prelude::*;
use vyre_primitives::graph::csr_forward_traverse::{cpu_ref, csr_forward_traverse};

/// Drive the real forward-step IR and return the `frontier_out` word bitset.
fn gpu_forward_step(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_in: &[u32],
    allow_mask: u32,
) -> Vec<u32> {
    frontier_step_out(
        csr_forward_traverse,
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
        let (node_count, offsets, targets, kind_mask, frontier, allow_mask) =
            csr_sweep::generate(csr_sweep::group("multi_source_restricted_kinds"), seed)
                .into_parts();
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
        let mut offsets = vec![2u32; node_count as usize + 1];
        offsets[0] = 0;
        offsets
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
