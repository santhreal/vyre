//! `size_argument_of`  -  reverse CallArg traversal for size argument
//! candidates.
//!
//! The primitive marks argument nodes whose callee is in the input
//! frontier. Rule-level predicates own any additional node-kind
//! filtering.

use vyre_foundation::composition::tag_program;
use vyre_foundation::ir::Program;

use crate::graph::program_graph::ProgramGraphShape;
use crate::predicate::arg_of::arg_of;

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
    tag_program(OP_ID, arg_of(shape, frontier_in, frontier_out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predicate::traversal::assert_region_op_id;

    #[test]
    fn size_argument_of_preserves_region_generator() {
        let program = size_argument_of(ProgramGraphShape::new(4, 2), "fin", "fout");
        assert_region_op_id(&program, OP_ID, "size_argument_of");
    }
}

const EXPECTED_SIZE_ARGUMENT_OF_OUTPUT_BYTES: [u8; 4] = [5, 0, 0, 0];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || size_argument_of(ProgramGraphShape::new(4, 4), "fin", "fout"),
        Some(|| {
            use crate::predicate::{edge_kind, node_kind};
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
    .with_laws(&["complement"])
}
