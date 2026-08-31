//! `literal_of`  -  `NodeSet = { v : nodes[v] == Literal AND
//!                                  literal_values[v] == probe }`.
//!
//! The IR-level primitive filters by NodeKind only; a external analyzer's
//! type-inference ensures `literal_of(probe)` is only lowered against
//! literal-typed frontiers. A runtime match on the literal value can
//! be composed by re-filtering with a dedicated literal-payload
//! comparison primitive in this crate.

use vyre_foundation::composition::tag_program;
use vyre_foundation::ir::Program;

use crate::predicate::node_kind;
use crate::predicate::node_kind_eq::node_kind_eq;

/// Canonical op id.
pub const OP_ID: &str = "vyre-libs::predicate::literal_of";

/// Build a Program that emits every node whose kind is Literal.
#[must_use]
pub fn literal_of(nodes: &str, nodeset_out: &str, node_count: u32) -> Program {
    tag_program(
        OP_ID,
        node_kind_eq(nodes, nodeset_out, node_count, node_kind::LITERAL),
    )
}

const EXPECTED_LITERAL_OF_OUTPUT_BYTES: [u8; 4] = [8, 0, 0, 0];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID,
        || literal_of("nodes", "nodeset", 4),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![vec![
                to_bytes(&[1, 2, 1, 4]), // nodes: VARIABLE, CALL, VARIABLE, LITERAL
                to_bytes(&[0]),          // nodeset_out
            ]]
        }),
        Some(|| {
            // node 3 (LITERAL)
            vec![vec![EXPECTED_LITERAL_OF_OUTPUT_BYTES.to_vec()]]
        }),
    )
    .with_laws(&["absorbing"])
}
