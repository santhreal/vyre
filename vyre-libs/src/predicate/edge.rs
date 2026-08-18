//! `edge`  -  raw forward traversal with caller-supplied edge mask.
//!
//! Primitive escape hatch for rules that match on non-canonical
//! edge-kind combinations. Downstream analyzer lowers arbitrary
//! `edge(frontier, kind_mask)` expressions directly through this.

use vyre_foundation::ir::Program;

use crate::graph::program_graph::ProgramGraphShape;
use crate::predicate::traversal::forward_edge_program;

/// Canonical op id.
pub const OP_ID: &str = "vyre-libs::predicate::edge";

/// Build a Program. The body is a `Region { generator: edge::OP_ID }`
/// wrapping the underlying `csr_forward_traverse` so callers (the
/// external analyzer motif lowerer in particular) can locate the edge dispatch
/// by its own op id rather than the delegate's.
#[must_use]
pub fn edge(
    shape: ProgramGraphShape,
    frontier_in: &str,
    frontier_out: &str,
    edge_kind_mask: u32,
) -> Program {
    forward_edge_program(OP_ID, shape, frontier_in, frontier_out, edge_kind_mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::csr_forward_traverse::cpu_ref_into;
    use crate::predicate::traversal::assert_region_op_id;

    #[test]
    fn preserves_wrapper_op_id() {
        let program = edge(ProgramGraphShape::new(4, 2), "fin", "fout", 0xFFFF_FFFF);
        assert_region_op_id(&program, OP_ID, "edge");
    }

    #[test]
    fn cpu_ref_into_reuses_forward_edge_nodeset() {
        let mut out = Vec::with_capacity(4);
        cpu_ref_into(
            4,
            &[0, 1, 2, 2, 2],
            &[1, 2],
            &[1, 1],
            &[0b0001],
            0xFFFF_FFFF,
            &mut out,
        );
        assert_eq!(out, vec![0b0010]);
    }
}

const EXPECTED_EDGE_OUTPUT_BYTES: [u8; 4] = [2, 0, 0, 0];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || edge(ProgramGraphShape::new(4, 2), "fin", "fout", 0xFFFF_FFFF),
        Some(|| {
            use super::inventory_u32_le_bytes as b;
            vec![vec![
                b(&[2, 1, 1, 1]),       // pg_nodes
                b(&[0, 1, 2, 2, 2]),    // pg_edge_offsets
                b(&[1, 2]),              // pg_edge_targets
                b(&[1, 1]),              // pg_edge_kind_mask (all edges)
                b(&[0, 0, 0, 0]),       // pg_node_tags
                b(&[0b0001]),            // frontier_in = {0}
                b(&[0]),                 // frontier_out
            ]]
        }),
        Some(|| {
            // {1} reached via any edge
            vec![vec![EXPECTED_EDGE_OUTPUT_BYTES.to_vec()]]
        }),
    )
}
