//! One-step predecessor query over dominance and block-membership edges.
//!
//! This operation performs one reverse CSR traversal. It does not compute a
//! dominator tree or a transitive dominance closure. Exact strict-dominance
//! checks use the independent `cpu_dominator_sets` test oracle.

use crate::graph::program_graph::ProgramGraphShape;
use crate::predicate::edge_kind;
use vyre_foundation::ir::Program;

use crate::security::flow_composition::{
    dominance_fixture_expected, dominance_fixture_inputs, security_flow_program, FlowPredicate,
    SecurityFlowOptions, FLOW_MAX_ITERATIONS,
};

pub(crate) const OP_ID: &str = "vyre-libs::security::dominance_predecessors";

/// Build one reverse-traversal step along dominance and block-membership edges.
///
/// The output contains the input frontier and its immediate matching
/// predecessors. It makes no transitive or strict-dominance claim.
#[must_use]
pub fn dominance_predecessors(
    shape: ProgramGraphShape,
    frontier_in: &str,
    frontier_out: &str,
) -> Program {
    security_flow_program(SecurityFlowOptions::reach(
        OP_ID,
        shape,
        FlowPredicate::backward(edge_kind::DOMINANCE | edge_kind::BLOCK_MEMBER),
        frontier_in,
        frontier_out,
    ))
}

/// CPU reference oracle for strict dominator sets.
///
/// Implements the iterative Cooper-Harvey-Kennedy dataflow algorithm and
/// returns the sorted dominator set for each node.
#[must_use]
#[cfg(test)]
pub(crate) fn cpu_dominator_sets(
    num_nodes: u32,
    entry: u32,
    edges: &[(u32, u32)],
) -> Vec<Vec<u32>> {
    let idoms = crate::graph::dominator_tree::cpu_ref(num_nodes, entry, edges);
    crate::graph::dominator_tree::idoms_to_dominator_sets(&idoms, num_nodes)
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || dominance_predecessors(ProgramGraphShape::new(4, 4), "fin", "fout"),
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
    use crate::graph::csr_backward_traverse::cpu_ref;
    use crate::security::flow_composition::diamond_dominance_tree;

    #[test]
    fn cpu_dominator_sets_linear_chain() {
        // 0 -> 1 -> 2 -> 3
        let edges = &[(0, 1), (1, 2), (2, 3)];
        let dom = cpu_dominator_sets(4, 0, edges);
        assert_eq!(dom[0], vec![0]);
        assert_eq!(dom[1], vec![0, 1]);
        assert_eq!(dom[2], vec![0, 1, 2]);
        assert_eq!(dom[3], vec![0, 1, 2, 3]);
    }

    #[test]
    fn cpu_dominator_sets_diamond() {
        // 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
        let edges = &[(0, 1), (0, 2), (1, 3), (2, 3)];
        let dom = cpu_dominator_sets(4, 0, edges);
        assert_eq!(dom[0], vec![0]);
        assert_eq!(dom[1], vec![0, 1]);
        assert_eq!(dom[2], vec![0, 2]);
        assert_eq!(dom[3], vec![0, 3]);
    }

    #[test]
    fn cpu_dominator_sets_while_loop() {
        // 0 -> 1, 1 -> 2, 2 -> 1, 1 -> 3
        let edges = &[(0, 1), (1, 2), (2, 1), (1, 3)];
        let dom = cpu_dominator_sets(4, 0, edges);
        assert_eq!(dom[0], vec![0]);
        assert_eq!(dom[1], vec![0, 1]);
        assert_eq!(dom[2], vec![0, 1, 2]);
        assert_eq!(dom[3], vec![0, 1, 3]);
    }

    #[test]
    fn dominance_predecessor_step_reaches_immediate_ancestors() {
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
        assert_eq!(out[0], 0b0110, "backward from 3 must reach 1 and 2");
    }

    #[test]
    fn dominance_predecessors_emit_frontier_buffers() {
        let p = dominance_predecessors(ProgramGraphShape::new(4, 4), "fin", "fout");
        let names: Vec<&str> = p.buffers().iter().map(|b| b.name()).collect();
        assert!(names.contains(&"fin"));
        assert!(names.contains(&"fout"));
    }

    #[test]
    fn dominance_predecessors_use_precise_operation_identity() {
        use vyre_foundation::ir::Node;
        let p = dominance_predecessors(ProgramGraphShape::new(2, 1), "fin", "fout");
        let [Node::Region { generator, .. }] = p.entry() else {
            panic!("dominance_predecessors must emit one wrapped region");
        };
        assert_eq!(generator.as_str(), OP_ID);
    }

    #[test]
    fn dominance_predecessors_do_not_claim_strict_dominance() {
        let p = dominance_predecessors(ProgramGraphShape::new(4, 4), "fin", "fout");
        let to_bytes = vyre_primitives::wire::pack_u32_slice;
        let inputs = vec![
            to_bytes(&[0, 0, 0, 0]),    // pg_nodes
            to_bytes(&[0, 2, 3, 4, 4]), // pg_edge_offsets
            to_bytes(&[1, 2, 3, 3]),    // pg_edge_targets
            to_bytes(&[
                edge_kind::DOMINANCE,
                edge_kind::DOMINANCE,
                edge_kind::DOMINANCE,
                edge_kind::DOMINANCE,
            ]),
            to_bytes(&[0, 0, 0, 0]), // pg_node_tags
            to_bytes(&[0b1000]),     // fin = {3}
            to_bytes(&[0b1000]),     // fout seed = {3}
        ];
        let values: Vec<vyre_reference::value::Value> = inputs
            .into_iter()
            .map(vyre_reference::value::Value::from)
            .collect();
        let outputs = vyre_reference::reference_eval(&p, &values).unwrap();
        let gpu_out = u32::from_le_bytes(outputs[0].to_bytes()[0..4].try_into().unwrap());

        let dom = cpu_dominator_sets(4, 0, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
        let true_dom_bitset: u32 = dom[3].iter().map(|&n| 1u32 << n).sum();

        // Adversarial test for the documented soundness gap.
        //
        // This operation is one `csr_backward_traverse` step over matching
        // edges. Starting from `{3}` yields the seed plus immediate
        // predecessors `{1, 2}` and never reaches `{0}` two hops away.
        let one_hop_predecessors_of_3: u32 = 0b1110; // {1, 2, 3}: self + immediate DOMINANCE preds
        assert_eq!(
            gpu_out, one_hop_predecessors_of_3,
            "dominance_predecessors returned {gpu_out:b}; expected \
             {one_hop_predecessors_of_3:b} (seed plus immediate predecessors). \
             Strict dominators are {true_dom_bitset:b}."
        );
        assert_ne!(
            gpu_out, true_dom_bitset,
            "one-step dominance predecessors must differ from strict dominators \
             on the diamond"
        );
    }

    #[test]
    #[should_panic(expected = "node_count must be positive")]
    fn dominance_predecessors_reject_zero_node_count() {
        let _ = dominance_predecessors(ProgramGraphShape::new(0, 0), "fin", "fout");
    }

    #[test]
    #[should_panic(expected = "empty buffer name")]
    fn dominance_predecessors_reject_empty_buffer_name() {
        let _ = dominance_predecessors(ProgramGraphShape::new(4, 4), "", "fout");
    }
}
