//! `edge`  -  raw forward traversal with caller-supplied edge mask.
//!
//! Primitive escape hatch for rules that match on non-canonical
//! edge-kind combinations. Downstream analyzer lowers arbitrary
//! `edge(frontier, kind_mask)` expressions directly through this.

use vyre_foundation::ir::Program;

#[cfg(any(test, feature = "cpu-parity"))]
use crate::graph::csr_frontier_step::{define_csr_frontier_step_cpu_ref, CsrFrontierStepKind};
use crate::graph::program_graph::ProgramGraphShape;
use crate::predicate::traversal::forward_edge_program;

/// Canonical op id.
pub const OP_ID: &str = "vyre-primitives::predicate::edge";

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

define_csr_frontier_step_cpu_ref! {
    direction: CsrFrontierStepKind::Forward,
    label: "predicate::edge",
    /// CPU reference for the `edge` predicate.
    ///
    /// AUDIT_2026-04-24 F-PE-01: the inventory fixture used to ship
    /// without a `cpu_ref`, leaving the conform harness unable to
    /// byte-compare GPU output against a reference. `edge` is a thin
    /// alias for forward traversal, so publishing the same forward step
    /// under this op id is the exact semantic match.
    pub fn cpu_ref,
    /// CPU reference using caller-owned output storage.
    pub fn cpu_ref_into,
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_foundation::ir::Node;

    #[test]
    fn preserves_wrapper_op_id() {
        let program = edge(ProgramGraphShape::new(4, 2), "fin", "fout", 0xFFFF_FFFF);
        let generator = match &program.entry[0] {
            Node::Region { generator, .. } => generator.to_string(),
            other => panic!("Fix: edge must build a Region entry, got {other:?}."),
        };
        assert_eq!(generator, OP_ID);
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

#[cfg(feature = "inventory-registry")]
inventory::submit! {
    vyre_foundation::operation::OperationRegistration::primitive(
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
            use super::inventory_u32_le_bytes as b;
            vec![vec![b(&[0b0010])]]   // {1} reached via any edge
        }),
    )
}
