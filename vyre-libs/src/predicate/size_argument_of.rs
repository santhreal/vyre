//! `size_argument_of`  -  reverse CallArg traversal for size argument
//! candidates.
//!
//! The primitive marks argument nodes whose callee is in the input
//! frontier. Rule-level predicates own any additional node-kind
//! filtering.

use vyre_foundation::ir::Program;

use crate::graph::program_graph::ProgramGraphShape;
use crate::predicate::edge_kind;
use crate::predicate::node_kind;
use crate::predicate::traversal::backward_edge_program;

/// Canonical op id.
pub const OP_ID: &str = "vyre-libs::predicate::size_argument_of";

/// Build a Program that reverse-traverses CallArg edges and marks
/// argument nodes whose callees are in `frontier_in`.
///
/// Downstream analyzer rules own any additional node-kind predicates at the rule
/// layer. This primitive deliberately avoids a baked-in Literal filter:
/// allocator size arguments are often computed expressions, and
/// filtering here would erase realistic vulnerability witnesses before
/// rule-specific predicates can inspect them.
#[must_use]
pub fn size_argument_of(
    shape: ProgramGraphShape,
    frontier_in: &str,
    frontier_out: &str,
) -> Program {
    backward_edge_program(OP_ID, shape, frontier_in, frontier_out, edge_kind::CALL_ARG)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_foundation::ir::Node;

    #[test]
    fn preserves_wrapper_op_id() {
        let program = size_argument_of(ProgramGraphShape::new(4, 2), "fin", "fout");
        let generator = match &program.entry[0] {
            Node::Region { generator, .. } => generator.to_string(),
            other => panic!("Fix: size_argument_of must build a Region entry, got {other:?}."),
        };
        assert_eq!(generator, OP_ID);
    }
}

const EXPECTED_SIZE_ARGUMENT_OF_OUTPUT_BYTES: [u8; 4] = [5, 0, 0, 0];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || size_argument_of(ProgramGraphShape::new(4, 4), "fin", "fout"),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![vec![
                to_bytes(&[node_kind::LITERAL, node_kind::CALL, node_kind::LITERAL, node_kind::CALL]),
                to_bytes(&[0, 1, 2, 3, 4]),
                to_bytes(&[1, 2, 3, 0]),
                to_bytes(&[edge_kind::CALL_ARG, 0, edge_kind::CALL_ARG, 0]),
                to_bytes(&[0, 0, 0, 0]),
                to_bytes(&[0b1010]),
                to_bytes(&[0]),
            ]]
        }),
        Some(|| {
            vec![vec![EXPECTED_SIZE_ARGUMENT_OF_OUTPUT_BYTES.to_vec()]]
        }),
    )
}
