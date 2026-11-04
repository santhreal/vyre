//! Kernel body assembly: capacity estimation, literal and child-body pushes,
//! and the trap scan over an assembled body.

use crate::descriptor::{KernelBody, KernelOpKind, LiteralValue};
use crate::error::LowerError;
use vyre_foundation::ir::{Node, Program};

pub(super) fn body_contains_trap(body: &KernelBody) -> bool {
    body.ops
        .iter()
        .any(|op| matches!(op.kind, KernelOpKind::Trap { .. }))
        || body.child_bodies.iter().any(body_contains_trap)
}

pub(super) fn opaque_extension_id(extension: &dyn vyre_foundation::ir::ExprNode) -> u32 {
    u32::from_le_bytes(
        extension.stable_fingerprint()[0..4]
            .try_into()
            .unwrap_or_else(|_| unreachable!("slice length is fixed")),
    )
}

pub(super) fn empty_body_for_nodes(nodes: &[Node]) -> KernelBody {
    empty_body_with_capacity(estimated_node_slice_op_capacity(nodes))
}

pub(super) fn empty_body_with_capacity(op_capacity: usize) -> KernelBody {
    KernelBody {
        ops: Vec::with_capacity(op_capacity),
        child_bodies: Vec::with_capacity(estimated_child_body_capacity(op_capacity)),
        literals: Vec::with_capacity(op_capacity / 3),
    }
}

pub(super) fn estimated_root_op_capacity(program: &Program) -> usize {
    let stats = program.stats();
    stats
        .instruction_count
        .saturating_add(stats.node_count as u64)
        .saturating_add(4)
        .min(usize::MAX as u64) as usize
}

pub(super) fn estimated_node_slice_op_capacity(nodes: &[Node]) -> usize {
    nodes
        .len()
        .saturating_mul(2)
        .saturating_add(estimated_child_body_capacity(nodes.len()))
}

pub(super) fn estimated_child_body_capacity(parent_ops: usize) -> usize {
    parent_ops.min(16)
}

pub(super) fn push_literal(
    body: &mut KernelBody,
    literal: LiteralValue,
) -> Result<u32, LowerError> {
    let index = u32::try_from(body.literals.len()).map_err(|_| LowerError::OperandIdOverflow)?;
    body.literals.push(literal);
    Ok(index)
}

pub(super) fn push_child(body: &mut KernelBody, child: KernelBody) -> Result<u32, LowerError> {
    let index =
        u32::try_from(body.child_bodies.len()).map_err(|_| LowerError::OperandIdOverflow)?;
    body.child_bodies.push(child);
    Ok(index)
}
