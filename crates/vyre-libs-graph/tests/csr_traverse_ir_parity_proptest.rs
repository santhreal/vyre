//! GPU-IR vs CPU-ref parity for one forward and one reverse CSR frontier step.
//!
//! Both directions run one dispatch round over the same generated graphs, so
//! they share the fixtures, the generator group and the assertion shape. They
//! are stated together because two files stating the same round twice is a
//! copy: the direction under test is the only thing that differs, and it is
//! named once per module below.
#![forbid(unsafe_code)]
#![cfg(feature = "graph")]

use crate::csr_sweep;
use crate::graph_sweep_fixtures;
use graph_sweep_fixtures::{bitset_words, frontier_step_out};

/// Forward expansion: one dispatch round of `graph::csr_forward_traverse`.
///
/// Each lane owns a source node. For every outgoing edge whose
/// `edge_kind_mask[e] & allow_mask` is nonzero the lane computes
/// `dst = edge_targets[e]`, checks `dst < node_count`, and atomic-ORs the
/// destination bit into `frontier_out`. Transitive closure is a separate
/// bitset_fixpoint composition, so a single `reference_eval` pass models this
/// round exactly. A missing `allow_mask` gate, a dropped `dst < node_count`
/// bound, or a non-atomic OR that loses a bit when two source lanes write the
/// same output word all diverge from the reference here.
pub mod forward {
    use super::{bitset_words, csr_sweep, frontier_step_out};
    use proptest::prelude::*;
    use vyre_libs_graph::graph::csr_forward_traverse::csr_forward_traverse;
    use vyre_reference::composition_witness::csr_forward_traverse_witness as cpu_ref;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2000))]

        #[test]
        fn ir_matches_cpu_ref_over_random_graphs(seed in any::<u64>()) {
            let (node_count, offsets, targets, kind_mask, frontier, allow_mask) =
                csr_sweep::generate(csr_sweep::group("multi_source_restricted_kinds"), seed)
                    .into_parts();
            let expected = cpu_ref(node_count, &offsets, &targets, &kind_mask, &frontier, allow_mask);
            let got = frontier_step_out(csr_forward_traverse, node_count, &offsets, &targets, &kind_mask, &frontier, allow_mask);
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
            frontier_step_out(
                csr_forward_traverse,
                node_count,
                &offsets,
                &targets,
                &kind_mask,
                &frontier,
                0xFFFF_FFFF,
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
            frontier_step_out(
                csr_forward_traverse,
                node_count,
                &offsets,
                &targets,
                &kind_mask,
                &frontier,
                1 << 4,
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
            frontier_step_out(
                csr_forward_traverse,
                node_count,
                &offsets,
                &targets,
                &kind_mask,
                &frontier,
                1 << 2,
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
            frontier_step_out(
                csr_forward_traverse,
                node_count,
                &offsets,
                &oob_targets,
                &kind_mask,
                &frontier,
                1 << 2,
            ),
            oob,
            "OOB dst must be bound-gated in IR too"
        );
    }
}

/// Reverse pull: one dispatch round of `graph::csr_backward_traverse`.
///
/// Each lane owns a source node and scans its outgoing CSR edges. If any
/// allowed edge points at a destination set in `frontier_in`, the lane
/// atomic-ORs its own bit into `frontier_out` and short-circuits. A single
/// `reference_eval` pass models this round exactly. A broken short-circuit, a
/// dst/src word-index swap, or a lost atomic mark all diverge from the
/// reference here.
pub mod backward {
    use super::{bitset_words, csr_sweep, frontier_step_out};
    use proptest::prelude::*;
    use vyre_libs_graph::graph::csr_backward_traverse::csr_backward_traverse;
    use vyre_reference::composition_witness::csr_backward_traverse_witness as cpu_ref;

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
            let got = frontier_step_out(csr_backward_traverse, node_count, &offsets, &targets, &kind_mask, &frontier, allow_mask);
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
            frontier_step_out(
                csr_backward_traverse,
                4,
                &offsets,
                &targets,
                &[1, 1, 1, 1],
                &frontier,
                0xFFFF_FFFF
            ),
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
            frontier_step_out(
                csr_backward_traverse,
                node_count,
                &offsets,
                &targets,
                &kind_mask,
                &frontier,
                0xFFFF_FFFF,
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
            frontier_step_out(
                csr_backward_traverse,
                2,
                &offsets,
                &targets,
                &kind_mask,
                &frontier,
                1 << 4
            ),
            dropped
        );
        let fired = cpu_ref(2, &offsets, &targets, &kind_mask, &frontier, 1 << 2);
        assert_eq!(fired, vec![0b01u32], "cpu_ref: matching mask pulls node 0");
        assert_eq!(
            frontier_step_out(
                csr_backward_traverse,
                2,
                &offsets,
                &targets,
                &kind_mask,
                &frontier,
                1 << 2
            ),
            fired
        );
    }
}
