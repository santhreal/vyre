//! `taint_flow`  -  the `flows_to` forward-reach predicate under a second
//! op id, so the conformance harness covers the API-facing `taint_flow` /
//! `taint_flow_unsanitized` names as well.
//!
//! Downstream analyzer's predicate lowering routes both `taint_flow` and
//! `flows_to` through `BinaryGraphKind::FlowsToForward`; there is no semantic
//! difference. Both build from
//! `crate::security::flow_composition::security_flow_program` with the same
//! [`FLOWS_TO_MASK`] predicate, so the only thing that can differ is the op id
//! the region carries.

use vyre_foundation::ir::Program;
use crate::graph::program_graph::ProgramGraphShape;

use crate::security::flow_composition::{
    forward_reach_fixture_expected, forward_reach_fixture_inputs, security_flow_program,
    FlowPredicate, SecurityFlowOptions, FLOW_MAX_ITERATIONS,
};
use crate::security::flows_to::FLOWS_TO_MASK;

pub(crate) const OP_ID: &str = "vyre-libs::security::taint_flow";

/// Build one forward-traversal step over DATAFLOW edges only.
#[must_use]
pub fn taint_flow(shape: ProgramGraphShape, frontier_in: &str, frontier_out: &str) -> Program {
    security_flow_program(SecurityFlowOptions::reach(
        OP_ID,
        shape,
        FlowPredicate::forward(FLOWS_TO_MASK),
        frontier_in,
        frontier_out,
    ))
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || taint_flow(ProgramGraphShape::new(4, 3), "fin", "fout"),
        Some(forward_reach_fixture_inputs),
        Some(forward_reach_fixture_expected),
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
    use crate::predicate::edge_kind;

    #[test]
    fn taint_flow_uses_restricted_dataflow_mask() {
        assert_eq!(FLOWS_TO_MASK & edge_kind::CONTROL, 0);
        assert_eq!(FLOWS_TO_MASK & edge_kind::DOMINANCE, 0);
        assert_ne!(FLOWS_TO_MASK & edge_kind::ASSIGNMENT, 0);
    }

    #[test]
    fn taint_flow_program_emits_frontier_buffers() {
        let p = taint_flow(ProgramGraphShape::new(4, 3), "fin", "fout");
        let names: Vec<&str> = p.buffers.iter().map(|b| b.name()).collect();
        assert!(names.contains(&"fin"));
        assert!(names.contains(&"fout"));
    }

    #[test]
    fn taint_flow_program_uses_non_degenerate_shape() {
        let shape = ProgramGraphShape::new(64, 128);
        let p = taint_flow(shape, "fin", "fout");
        let fin_buf = p
            .buffers
            .iter()
            .find(|b| b.name() == "fin")
            .expect("Fix: fin buffer");
        assert!(
            fin_buf.count >= 2,
            "bitset_words(64) = 2; count {} suggests degenerate shape",
            fin_buf.count
        );
    }

    #[test]
    fn taint_flow_delegation_preserves_distinct_ir_identity() {
        let p_flows =
            crate::security::flows_to::flows_to(ProgramGraphShape::new(4, 3), "fin", "fout");
        let p_taint = taint_flow(ProgramGraphShape::new(4, 3), "fin", "fout");
        assert_ne!(
            p_flows.fingerprint(),
            p_taint.fingerprint(),
            "distinct operation identities must produce distinct canonical IR fingerprints"
        );
    }

    #[test]
    #[should_panic(expected = "node_count must be positive")]
    fn taint_flow_zero_node_count_should_panic() {
        let _ = taint_flow(ProgramGraphShape::new(0, 0), "fin", "fout");
    }

    #[test]
    #[should_panic(expected = "empty buffer name")]
    fn taint_flow_empty_buffer_name_should_panic() {
        let _ = taint_flow(ProgramGraphShape::new(4, 3), "", "fout");
    }
}
