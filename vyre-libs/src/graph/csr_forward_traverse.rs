//! `csr_forward_traverse`  -  one BFS frontier step over a
//! `super::program_graph::ProgramGraph`.
//!
//! Given an input frontier bitset (`frontier_in`) and a per-edge
//! allow-mask, the primitive emits the next frontier: every node
//! that has at least one predecessor in `frontier_in` reached via
//! an edge whose `edge_kind_mask` intersects the allowed mask.
//!
//! One dispatch is one step. Transitive closure is driven by
//! composing with `super::super::bitset` primitives and
//! `super::super::fixpoint::bitset_fixpoint`.
//!
//! CPU-independent witnesses live in `vyre-reference`.

use vyre_foundation::ir::Program;

use crate::graph::csr_frontier_step::{
    csr_forward_step_excluding_program, csr_frontier_step_program, CsrFrontierStepKind,
};
use crate::graph::program_graph::ProgramGraphShape;

/// Canonical op id.
pub const OP_ID: &str = "vyre-libs::graph::csr_forward_traverse";
/// Canonical op id for a forward step that excludes selected source nodes.
pub const EXCLUDING_OP_ID: &str = "vyre-libs::graph::csr_forward_traverse_excluding";

pub use crate::graph::csr_frontier_step::{
    csr_frontier_step_dispatch_grid as csr_forward_traverse_dispatch_grid,
    BINDING_EXCLUDED_SOURCES, BINDING_EXCLUDING_FRONTIER_OUT, BINDING_FRONTIER_IN,
    BINDING_FRONTIER_OUT,
};

/// Build the IR `Program` for one BFS forward step.
///
/// Each invocation owns one source node `src`. For each outgoing edge
/// whose `edge_kind_mask` intersects `allow_mask`, the program computes
/// `dst = edge_targets[e]` and atomically ORs the destination bit into
/// `frontier_out`. Transitive closure is driven by composing this step
/// with `bitset_fixpoint`.
///
/// Backward-edge iteration would be cheap given a CSC side-car; for
/// forward-only CSR, the atomic-OR path keeps the primitive
/// substrate-neutral without requiring two index layouts.
///
/// `dst` is bounds-checked against `shape.node_count` before
/// `atomic_or` so malformed edge lists cannot write outside the
/// node-indexed `frontier_out` bitset.
#[must_use]
pub fn csr_forward_traverse(
    shape: ProgramGraphShape,
    frontier_in: &str,
    frontier_out: &str,
    allow_mask: u32,
) -> Program {
    csr_frontier_step_program(
        OP_ID,
        CsrFrontierStepKind::Forward,
        shape,
        frontier_in,
        frontier_out,
        allow_mask,
    )
}

/// Build one CSR forward step while excluding source nodes selected by a bitset.
///
/// Destination nodes remain observable when reached. Exclusion applies only
/// when a selected node would become the source of a later traversal step.
#[must_use]
pub fn csr_forward_traverse_excluding(
    shape: ProgramGraphShape,
    frontier_in: &str,
    excluded_sources: &str,
    frontier_out: &str,
    allow_mask: u32,
) -> Program {
    csr_forward_step_excluding_program(
        EXCLUDING_OP_ID,
        shape,
        frontier_in,
        excluded_sources,
        frontier_out,
        allow_mask,
    )
}

const EXPECTED_CSR_FORWARD_TRAVERSE_OUTPUT_BYTES: [u8; 4] = [6, 0, 0, 0];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        EXCLUDING_OP_ID,
        || csr_forward_traverse_excluding(
            ProgramGraphShape::new(4, 4),
            "fin",
            "excluded",
            "fout",
            0xFFFF_FFFF,
        ),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![crate::graph::program_graph::sample_program_graph_inputs(&[
                to_bytes(&[0b0011]),              // frontier_in = {0, 1}
                to_bytes(&[0b0010]),              // excluded_sources = {1}
                to_bytes(&[0]),                   // frontier_out
            ])]
        }),
        Some(|| {
            // Source 0 reaches {1, 2}; excluded source 1 does not reach 3.
            vec![vec![EXPECTED_CSR_FORWARD_TRAVERSE_OUTPUT_BYTES.to_vec()]]
        }),
    )
    .with_laws(&["monotonic"])
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || csr_forward_traverse(ProgramGraphShape::new(4, 4), "fin", "fout", 0xFFFF_FFFF),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![crate::graph::program_graph::sample_program_graph_inputs(&[
                to_bytes(&[0b0001]),              // frontier_in = {0}
                to_bytes(&[0]),                   // frontier_out
            ])]
        }),
        Some(|| {
            // After one forward step starting from {0}: frontier = {1, 2}.
            vec![vec![EXPECTED_CSR_FORWARD_TRAVERSE_OUTPUT_BYTES.to_vec()]]
        }),
    )
    .with_laws(&["monotonic"])
}

#[cfg(test)]
pub(crate) use crate::graph::csr_frontier_step::{
    csr_forward_traverse_cpu_ref as cpu_ref, csr_forward_traverse_cpu_ref_into as cpu_ref_into,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_step_reaches_immediate_successors() {
        // 0→1, 0→2, 1→3, 2→3
        let got = cpu_ref(
            4,
            &[0, 2, 3, 4, 4],
            &[1, 2, 3, 3],
            &[1, 1, 1, 1],
            &[0b0001],
            0xFFFF_FFFF,
        );
        assert_eq!(got, vec![0b0110]);
    }

    #[test]
    fn edge_mask_filters_disallowed_edges() {
        // Same graph but one edge (0→1) has mask 0b10, others 0b01.
        // Allow only 0b01: out frontier should exclude node 1.
        let got = cpu_ref(
            4,
            &[0, 2, 3, 4, 4],
            &[1, 2, 3, 3],
            &[0b10, 0b01, 0b01, 0b01],
            &[0b0001],
            0b01,
        );
        assert_eq!(got, vec![0b0100]);
    }

    #[test]
    fn empty_frontier_produces_empty_output() {
        let got = cpu_ref(
            4,
            &[0, 2, 3, 4, 4],
            &[1, 2, 3, 3],
            &[1, 1, 1, 1],
            &[0],
            0xFFFF_FFFF,
        );
        assert_eq!(got, vec![0]);
    }

    #[test]
    #[should_panic(expected = "complete CSR offset table")]
    fn malformed_csr_short_offsets_fail_loudly() {
        let _ = cpu_ref(4, &[0, 1], &[1], &[1], &[0b0001], 0xFFFF_FFFF);
    }

    #[test]
    fn cpu_ref_into_reuses_output_buffer() {
        let mut out = Vec::with_capacity(8);
        let ptr = out.as_ptr();
        cpu_ref_into(
            4,
            &[0, 2, 3, 4, 4],
            &[1, 2, 3, 3],
            &[1, 1, 1, 1],
            &[0b0001],
            0xFFFF_FFFF,
            &mut out,
        );
        assert_eq!(out.as_ptr(), ptr);
        assert_eq!(out, vec![0b0110]);
    }

    // ------------------------------------------------------------------
    // Adversarial fixtures  -  hostile boundaries, empty graphs, kind-mask
    // diversity (M8), malformed CSR, cross-word bitsets.
    // ------------------------------------------------------------------

    #[test]
    fn empty_graph_zero_nodes_zero_edges() {
        let got = cpu_ref(0, &[0], &[], &[], &[], 0xFFFF_FFFF);
        assert!(got.is_empty(), "0-node graph must produce empty bitset");
    }

    #[test]
    fn single_node_no_edges_returns_empty() {
        // 1 node, 0 edges, frontier {0} → no successors → empty output.
        let got = cpu_ref(1, &[0, 0], &[0], &[0], &[0b0001], 0xFFFF_FFFF);
        assert_eq!(got, vec![0]);
    }

    #[test]
    fn self_loops_only_preserve_frontier() {
        // 2 nodes, each has a self-loop. frontier {0,1} → out {0,1}
        let got = cpu_ref(2, &[0, 1, 2], &[0, 1], &[1, 1], &[0b0011], 0xFFFF_FFFF);
        assert_eq!(got, vec![0b0011]);
    }

    #[test]
    fn disconnected_components_only_reach_within_component() {
        // Component A: 0→1. Component B: 2→3. frontier {0} → out {1}
        let got = cpu_ref(
            4,
            &[0, 1, 1, 2, 2],
            &[1, 3],
            &[1, 1],
            &[0b0001],
            0xFFFF_FFFF,
        );
        assert_eq!(got, vec![0b0010]);
    }

    #[test]
    fn max_node_count_cross_word_boundary() {
        // 65 nodes (3 words), one edge from node 64 to node 0.
        let mut offsets = vec![0u32; 66];
        offsets[64] = 0;
        offsets[65] = 1;
        let mut frontier = vec![0u32; 3];
        frontier[2] = 1; // node 64 is set
        let got = cpu_ref(65, &offsets, &[0], &[1], &frontier, 0xFFFF_FFFF);
        assert_eq!(got.len(), 3);
        assert_eq!(got[0], 1, "node 0 reached from node 64");
        assert_eq!(got[1], 0);
        assert_eq!(got[2], 0, "node 64 is not its own successor");
    }

    #[test]
    fn edge_mask_filters_all_edges_yields_empty() {
        // All edges have mask 0b01; allow_mask is 0b10 → no overlap.
        let got = cpu_ref(
            4,
            &[0, 2, 3, 4, 4],
            &[1, 2, 3, 3],
            &[0b01, 0b01, 0b01, 0b01],
            &[0b0001],
            0b10,
        );
        assert_eq!(got, vec![0], "mask mismatch must block every edge");
    }

    #[test]
    fn edge_mask_universal_allow_mask_behaves_like_all_ones() {
        let got = cpu_ref(
            4,
            &[0, 2, 3, 4, 4],
            &[1, 2, 3, 3],
            &[1, 1, 1, 1],
            &[0b0001],
            0xFFFF_FFFF,
        );
        assert_eq!(got, vec![0b0110]);
    }

    #[test]
    fn edge_kind_diversity_ignoring_mask_would_fail() {
        // M8 regression: graph with two edge kinds.
        // 0→1 (DOMINANCE=0x01), 0→2 (ASSIGNMENT=0x02).
        // Mask only DOMINANCE → only node 1 reached.
        // A broken implementation that ignores kind_mask would reach BOTH.
        let got = cpu_ref(4, &[0, 2, 2, 2, 2], &[1, 2], &[0x01, 0x02], &[0b0001], 0x01);
        assert_eq!(
            got,
            vec![0b0010],
            "only DOMINANCE edge 0→1 must be traversed; broken impl ignoring mask would produce 0b0110"
        );
    }

    #[test]
    #[should_panic(expected = "non-monotonic CSR offsets")]
    fn malformed_csr_non_monotonic_offsets_fail_loudly() {
        let _ = cpu_ref(
            4,
            &[0, 2, 1, 1, 1],
            &[1, 2],
            &[1, 1],
            &[0b0001],
            0xFFFF_FFFF,
        );
    }

    #[test]
    fn frontier_word_oob_is_safely_skipped() {
        // 40 nodes need 2 words, but frontier only has 1 word.
        let offsets: Vec<u32> = (0..41).map(|_| 0).collect();
        let got = cpu_ref(40, &offsets, &[0], &[0], &[0], 0xFFFF_FFFF);
        assert_eq!(got, vec![0, 0], "short frontier must be handled safely");
    }
}
