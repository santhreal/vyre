//! `bounded_by_comparison`  -  backward reachability along DOMINANCE edges.
//!
//! AUDIT_2026-04-24 F-BBC-02 (doc fix): the primitive computes
//! reverse reachability along dominance edges  -  i.e. the set of
//! dominance-tree *ancestors* of each node in `frontier_in`. The
//! stdlib rule intersects that ancestor set with the bound-check
//! NodeSet. Prior doc text claimed "every access is reachable
//! backward along dominance edges from some bound check," which
//! describes descendant reachability, not ancestor reachability  -
//! the directions were swapped. Correct framing: "for each access
//! in `frontier_in`, compute the dominators via ancestor walk,
//! then a bound-check intersects to prove the access is covered
//! by some dominating bound-check."

use crate::graph::program_graph::ProgramGraphShape;
use crate::predicate::edge_kind;
use vyre_foundation::ir::Program;

use crate::security::flow_composition::{
    dominance_fixture_expected, dominance_fixture_inputs, security_flow_program, FlowPredicate,
    SecurityFlowOptions, FLOW_MAX_ITERATIONS,
};

pub(crate) const OP_ID: &str = "vyre-libs::security::bounded_by_comparison";

/// Build one reverse-traversal step filtered to dominance edges.
#[must_use]
pub fn bounded_by_comparison(
    shape: ProgramGraphShape,
    frontier_in: &str,
    frontier_out: &str,
) -> Program {
    security_flow_program(SecurityFlowOptions::reach(
        OP_ID,
        shape,
        FlowPredicate::backward(edge_kind::DOMINANCE),
        frontier_in,
        frontier_out,
    ))
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || bounded_by_comparison(ProgramGraphShape::new(4, 4), "fin", "fout"),
        Some(dominance_fixture_inputs),
        Some(dominance_fixture_expected),
    )
    .with_category("security")
}

inventory::submit! {
    crate::operation_catalog::ConvergenceContract {
        op_id: OP_ID,
        max_iterations: FLOW_MAX_ITERATIONS,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::security::flow_composition::diamond_dominance_tree;
    use vyre_reference::composition_witness::csr_backward_traverse_witness as cpu_ref;

    #[test]
    fn bounded_by_comparison_mask_is_dominance_only() {
        let _p = bounded_by_comparison(ProgramGraphShape::new(4, 4), "fin", "fout");
        // The primitive is a wrapper around csr_backward_traverse;
        // we verify the mask constant at the module level.
        assert_eq!(edge_kind::DOMINANCE & edge_kind::ASSIGNMENT, 0);
        assert_eq!(edge_kind::DOMINANCE & edge_kind::CONTROL, 0);
    }

    #[test]
    fn bounded_by_comparison_backward_step_reaches_ancestors() {
        let (node_count, offsets, targets, masks) = diamond_dominance_tree();
        let frontier_in = vec![0b1000]; // {3}
        let out = cpu_ref(
            node_count,
            &offsets,
            &targets,
            &masks,
            &frontier_in,
            edge_kind::DOMINANCE,
        );
        // Seed is NOT merged by cpu_ref; it returns only newly reached bits.
        assert_eq!(out[0], 0b0110, "backward from 3 must reach 1 and 2");
    }

    #[test]
    fn bounded_by_comparison_program_emits_frontier_buffers() {
        let _p = bounded_by_comparison(ProgramGraphShape::new(4, 4), "fin", "fout");
        let names: Vec<&str> = _p.buffers().iter().map(|b| b.name()).collect();
        assert!(names.contains(&"fin"));
        assert!(names.contains(&"fout"));
    }

    #[test]
    fn bounded_by_comparison_deep_chain_reaches_all_ancestors() {
        let node_count = 10u32;
        let mut offsets = vec![0u32; (node_count + 1) as usize];
        let mut targets = Vec::new();
        let mut masks = Vec::new();
        for i in 0..node_count {
            offsets[i as usize] = i;
            if i + 1 < node_count {
                targets.push(i + 1);
                masks.push(edge_kind::DOMINANCE);
            }
        }
        offsets[node_count as usize] = node_count.saturating_sub(1);

        let mut accumulated = vec![0u32; 1];
        accumulated[0] = 1 << (node_count - 1);

        for _ in 0..node_count {
            let out = cpu_ref(
                node_count,
                &offsets,
                &targets,
                &masks,
                &accumulated,
                edge_kind::DOMINANCE,
            );
            let new_accumulated: Vec<u32> =
                accumulated.iter().zip(&out).map(|(a, b)| a | b).collect();
            if new_accumulated == accumulated {
                break;
            }
            accumulated = new_accumulated;
        }

        let expected = (1u32 << node_count) - 1;
        assert_eq!(
            accumulated[0],
            expected,
            "backward reachability from node {} must reach all ancestors in a {}-node chain; \
             if max_iterations truncates, this test fails",
            node_count - 1,
            node_count
        );

        let contract = crate::operation_catalog::convergence_contract(OP_ID)
            .expect("Fix: bounded_by_comparison must have a ConvergenceContract");
        assert!(
            contract.max_iterations >= node_count,
            "ConvergenceContract max_iterations ({}) must be >= chain depth ({}) to avoid silent truncation",
            contract.max_iterations, node_count
        );
    }

    #[test]
    #[should_panic(expected = "node_count must be positive")]
    fn bounded_by_comparison_zero_node_count_should_panic() {
        let _ = bounded_by_comparison(ProgramGraphShape::new(0, 0), "fin", "fout");
    }

    #[test]
    #[should_panic(expected = "empty buffer name")]
    fn bounded_by_comparison_empty_buffer_name_should_panic() {
        let _ = bounded_by_comparison(ProgramGraphShape::new(4, 4), "", "fout");
    }
}
