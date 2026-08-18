//! `csr_bidirectional`  -  one BFS step over BOTH forward + backward
//! edges of a ProgramGraph CSR. Used for undirected reachability
//! (e.g. component discovery, alias unification).

use vyre_foundation::composition::trap_program;
use vyre_foundation::execution_plan::fusion::fuse_programs;
use vyre_foundation::ir::{DataType, Program};

use crate::graph::csr_backward_traverse::csr_backward_traverse;
use crate::graph::csr_forward_traverse::csr_forward_traverse;
use crate::graph::program_graph::ProgramGraphShape;

/// Canonical op id.
pub const OP_ID: &str = "vyre-libs::graph::csr_bidirectional";
/// Canonical dispatch input label for graph node scratch.
pub const CSR_BIDIRECTIONAL_NODES_BUFFER: &str = "csr_bidirectional nodes";
/// Canonical dispatch input label for CSR offsets.
pub const CSR_BIDIRECTIONAL_OFFSETS_BUFFER: &str = "csr_bidirectional edge_offsets";
/// Canonical dispatch input label for CSR targets.
pub const CSR_BIDIRECTIONAL_TARGETS_BUFFER: &str = "csr_bidirectional edge_targets";
/// Canonical dispatch input label for edge-kind masks.
pub const CSR_BIDIRECTIONAL_EDGE_KIND_BUFFER: &str = "csr_bidirectional edge_kind_mask";
/// Canonical dispatch input label for node tags.
pub const CSR_BIDIRECTIONAL_NODE_TAGS_BUFFER: &str = "csr_bidirectional node_tags";
/// Canonical dispatch input label for the incoming frontier.
pub const CSR_BIDIRECTIONAL_FRONTIER_IN_BUFFER: &str = "csr_bidirectional frontier_in";
/// Canonical dispatch output label for the outgoing frontier.
pub const CSR_BIDIRECTIONAL_FRONTIER_OUT_BUFFER: &str = "csr_bidirectional frontier_out";

/// Build a Program: emit one forward step + one backward step,
/// fused into one Region. Both writes target `frontier_out` so a
/// single dispatch covers both directions.
#[must_use]
pub fn csr_bidirectional(
    shape: ProgramGraphShape,
    frontier_in: &str,
    frontier_out: &str,
    edge_kind_mask: u32,
) -> Program {
    let fwd = csr_forward_traverse(shape, frontier_in, frontier_out, edge_kind_mask);
    let bwd = csr_backward_traverse(shape, frontier_in, frontier_out, edge_kind_mask);
    fuse_programs(&[fwd, bwd]).unwrap_or_else(|error| {
        trap_program(
            OP_ID,
            Some((frontier_out, DataType::U32)),
            format!("Fix: csr_bidirectional forward+backward fusion failed: {error}"),
        )
    })
}

#[path = "csr_bidirectional_plan.rs"]
mod csr_bidirectional_plan;
pub use csr_bidirectional_plan::*;
#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::csr_closure_inputs::{graphs, CsrClosureInputs};
    use vyre_reference::composition_witness::{
        csr_bidirectional_closure_witness_into, csr_bidirectional_step_witness_into,
    };

    fn try_cpu_ref_into(
        node_count: u32,
        edge_offsets: &[u32],
        edge_targets: &[u32],
        edge_kind_mask: &[u32],
        frontier_in: &[u32],
        allow_mask: u32,
        out: &mut Vec<u32>,
    ) -> Result<(), String> {
        let layout = validate_csr_inputs(
            node_count,
            edge_offsets,
            edge_targets,
            edge_kind_mask,
            frontier_in,
        )?;
        csr_bidirectional_step_witness_into(
            node_count,
            edge_offsets,
            edge_targets,
            edge_kind_mask,
            frontier_in,
            allow_mask,
            out,
        );
        if out.len() < layout.words as usize {
            out.resize(layout.words as usize, 0);
        }
        Ok(())
    }

    fn try_cpu_ref(
        node_count: u32,
        edge_offsets: &[u32],
        edge_targets: &[u32],
        edge_kind_mask: &[u32],
        frontier_in: &[u32],
        allow_mask: u32,
    ) -> Result<Vec<u32>, String> {
        let mut out = Vec::new();
        try_cpu_ref_into(
            node_count,
            edge_offsets,
            edge_targets,
            edge_kind_mask,
            frontier_in,
            allow_mask,
            &mut out,
        )?;
        Ok(out)
    }

    fn cpu_ref_into(
        node_count: u32,
        edge_offsets: &[u32],
        edge_targets: &[u32],
        edge_kind_mask: &[u32],
        frontier_in: &[u32],
        allow_mask: u32,
        out: &mut Vec<u32>,
    ) {
        try_cpu_ref_into(
            node_count,
            edge_offsets,
            edge_targets,
            edge_kind_mask,
            frontier_in,
            allow_mask,
            out,
        )
        .expect("cpu_ref_into failed");
    }

    fn cpu_ref(
        node_count: u32,
        edge_offsets: &[u32],
        edge_targets: &[u32],
        edge_kind_mask: &[u32],
        frontier_in: &[u32],
        allow_mask: u32,
    ) -> Vec<u32> {
        try_cpu_ref(
            node_count,
            edge_offsets,
            edge_targets,
            edge_kind_mask,
            frontier_in,
            allow_mask,
        )
        .expect("cpu_ref failed")
    }

    fn try_cpu_ref_closure_into(
        inputs: CsrClosureInputs<'_>,
        seed: &[u32],
        current: &mut Vec<u32>,
        next: &mut Vec<u32>,
    ) -> Result<(), String> {
        let _layout = validate_csr_inputs(
            inputs.graph.node_count,
            inputs.graph.edge_offsets,
            inputs.graph.edge_targets,
            inputs.graph.edge_kind_mask,
            seed,
        )?;
        csr_bidirectional_closure_witness_into(
            inputs.graph.node_count,
            inputs.graph.edge_offsets,
            inputs.graph.edge_targets,
            inputs.graph.edge_kind_mask,
            seed,
            inputs.allow_mask,
            inputs.max_iters,
            current,
            next,
        );
        Ok(())
    }

    fn try_cpu_ref_closure(inputs: CsrClosureInputs<'_>, seed: &[u32]) -> Result<Vec<u32>, String> {
        let mut current = Vec::new();
        let mut next = Vec::new();
        try_cpu_ref_closure_into(inputs, seed, &mut current, &mut next)?;
        Ok(current)
    }

    fn cpu_ref_closure_into(
        inputs: CsrClosureInputs<'_>,
        seed: &[u32],
        current: &mut Vec<u32>,
        next: &mut Vec<u32>,
    ) {
        try_cpu_ref_closure_into(inputs, seed, current, next).expect("cpu_ref_closure_into failed");
    }

    fn cpu_ref_closure(inputs: CsrClosureInputs<'_>, seed: &[u32]) -> Vec<u32> {
        try_cpu_ref_closure(inputs, seed).expect("cpu_ref_closure failed")
    }

    #[test]
    fn forward_step_propagates() {
        let g = graphs::CHAIN_4;
        let out = cpu_ref(
            g.node_count,
            g.edge_offsets,
            g.edge_targets,
            g.edge_kind_mask,
            &[0b0001],
            u32::MAX,
        );
        // 0's forward neighbor = 1 → bit 1 set.
        assert!(out[0] & 0b0010 != 0);
    }

    #[test]
    fn empty_seed_yields_empty_step() {
        let g = graphs::CHAIN_4;
        let out = cpu_ref(
            g.node_count,
            g.edge_offsets,
            g.edge_targets,
            g.edge_kind_mask,
            &[0],
            u32::MAX,
        );
        assert_eq!(out, vec![0]);
    }

    #[test]
    fn allow_mask_zero_blocks_all() {
        let g = graphs::CHAIN_4;
        let out = cpu_ref(
            g.node_count,
            g.edge_offsets,
            g.edge_targets,
            g.edge_kind_mask,
            &[0b0001],
            0,
        );
        assert_eq!(out, vec![0]);
    }

    #[test]
    fn bidirectional_includes_both_directions() {
        let g = graphs::CHAIN_4;
        // From {1}, forward reaches {2}; backward reaches {0}.
        let out = cpu_ref(
            g.node_count,
            g.edge_offsets,
            g.edge_targets,
            g.edge_kind_mask,
            &[0b0010],
            u32::MAX,
        );
        assert!(out[0] & 0b0001 != 0, "bwd should reach node 0");
        assert!(out[0] & 0b0100 != 0, "fwd should reach node 2");
    }

    #[test]
    fn closure_reaches_full_linear_component() {
        let out = cpu_ref_closure(CsrClosureInputs::allow_all(graphs::CHAIN_4, 5), &[0b0001]);
        assert_eq!(out, vec![0b1111]);
    }

    #[test]
    fn closure_into_reuses_caller_buffers() {
        let mut current = Vec::with_capacity(8);
        let mut next = Vec::with_capacity(8);
        cpu_ref_closure_into(
            CsrClosureInputs::allow_all(graphs::CHAIN_4, 5),
            &[0b0001],
            &mut current,
            &mut next,
        );
        assert_eq!(current, vec![0b1111]);
        assert_eq!(current.capacity(), 8);
        assert_eq!(next.capacity(), 8);
    }

    #[test]
    fn merge_frontier_reports_change_and_or_merges_words() {
        let mut current = [0b0001u32, 0b1000];
        let next = [0b0110u32, 0b1000];
        assert!(merge_frontier_or_changed(&mut current, &next));
        assert_eq!(current, [0b0111, 0b1000]);
        assert!(!merge_frontier_or_changed(&mut current, &next));
    }

    #[test]
    fn try_merge_frontier_rejects_mismatched_word_counts_without_panic() {
        let mut current = [0u32];
        let next = [1u32, 2];
        let err = try_merge_frontier_or_changed(&mut current, &next)
            .expect_err("mismatched frontier word counts must be a typed error");
        assert!(err.contains("equal bitset word counts"));
        assert_eq!(current, [0u32]);
    }

    #[test]
    #[should_panic(
        expected = "Fix: bidirectional frontier merge requires equal bitset word counts"
    )]
    fn merge_frontier_rejects_mismatched_word_counts() {
        let mut current = [0u32];
        let next = [1u32, 2];
        let _ = merge_frontier_or_changed(&mut current, &next);
    }

    #[test]
    fn validate_csr_inputs_accepts_empty_and_canonical_graphs() {
        assert_eq!(
            validate_csr_inputs(0, &[0], &[], &[], &[]).unwrap(),
            CsrBidirectionalLayout {
                node_count: 0,
                words: 0,
                node_words: 0,
                edge_count: 0,
                edge_storage_words: 1,
            }
        );

        let g = graphs::CHAIN_4;
        assert_eq!(
            validate_csr_inputs(
                g.node_count,
                g.edge_offsets,
                g.edge_targets,
                g.edge_kind_mask,
                &[0]
            )
            .unwrap(),
            CsrBidirectionalLayout {
                node_count: 4,
                words: 1,
                node_words: 4,
                edge_count: 3,
                edge_storage_words: 3,
            }
        );
    }

    #[test]
    fn validate_csr_inputs_rejects_frontier_and_csr_contract_violations() {
        let err = validate_csr_inputs(2, &[0, 1, 1], &[1], &[1], &[]).unwrap_err();
        assert!(err.contains("expected frontier length"));

        let err = validate_csr_inputs(2, &[0, 1, 1], &[1], &[], &[0]).unwrap_err();
        assert!(err.contains("edge_targets.len() == edge_kind_mask.len()"));

        let err = validate_csr_inputs(2, &[0, 2, 1], &[1], &[1], &[0]).unwrap_err();
        assert!(err.contains("offsets must be monotonic"));

        let err = validate_csr_inputs(2, &[0, 1, 1], &[5], &[1], &[0]).unwrap_err();
        assert!(err.contains("outside node_count"));
    }

    #[test]
    fn try_cpu_ref_into_rejects_bad_csr_without_clobbering_output() {
        let mut out = vec![0xCAFE_BABEu32];
        let capacity = out.capacity();
        let err = try_cpu_ref_into(2, &[0, 1, 1], &[1], &[], &[0], u32::MAX, &mut out)
            .expect_err("mismatched edge arrays must return an error");
        assert!(err.contains("edge_targets.len() == edge_kind_mask.len()"));
        assert_eq!(out, vec![0xCAFE_BABEu32]);
        assert_eq!(out.capacity(), capacity);
    }

    #[test]
    fn try_cpu_ref_closure_rejects_bad_seed_without_clobbering_buffers() {
        let mut current = vec![0xCAFE_BABEu32];
        let mut next = vec![0xDEAD_BEEFu32];
        let current_capacity = current.capacity();
        let next_capacity = next.capacity();
        let err = try_cpu_ref_closure_into(
            CsrClosureInputs::allow_all(graphs::CHAIN_4, 4),
            &[],
            &mut current,
            &mut next,
        )
        .expect_err("bad seed width must be rejected");
        assert!(err.contains("expected frontier length"));
        assert_eq!(current, vec![0xCAFE_BABEu32]);
        assert_eq!(next, vec![0xDEAD_BEEFu32]);
        assert_eq!(current.capacity(), current_capacity);
        assert_eq!(next.capacity(), next_capacity);
    }

    #[test]
    fn fallible_cpu_reference_matches_compatibility_wrappers() {
        let g = graphs::CHAIN_4;
        let step = try_cpu_ref(
            g.node_count,
            g.edge_offsets,
            g.edge_targets,
            g.edge_kind_mask,
            &[0b0010],
            u32::MAX,
        )
        .expect("Fix: operation must return Err on failure; tests may use expect only with Fix: recovery text - valid step should succeed");
        assert_eq!(
            step,
            cpu_ref(
                g.node_count,
                g.edge_offsets,
                g.edge_targets,
                g.edge_kind_mask,
                &[0b0010],
                u32::MAX
            )
        );

        let inputs = CsrClosureInputs::allow_all(graphs::CHAIN_4, 5);
        let closure = try_cpu_ref_closure(inputs, &[0b0001])
            .expect("Fix: operation must return Err on failure; tests may use expect only with Fix: recovery text - valid closure should succeed");
        assert_eq!(closure, cpu_ref_closure(inputs, &[0b0001]));
    }

    #[test]
    fn cpu_ref_into_validates_before_resizing_output() {
        let mut out = vec![0xCAFE_BABEu32];
        let original_capacity = out.capacity();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cpu_ref_into(u32::MAX, &[0], &[], &[], &[], u32::MAX, &mut out);
        }));

        assert!(result.is_err(), "malformed CSR must still be rejected");
        assert_eq!(
            out,
            vec![0xCAFE_BABEu32],
            "invalid input must not clear or resize caller output before validation"
        );
        assert_eq!(
            out.capacity(),
            original_capacity,
            "invalid input must not allocate based on hostile node_count"
        );
    }
}
