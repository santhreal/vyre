//! Backend support validation before dispatch.

use super::capability::Backend;
use std::sync::{Arc, LazyLock};
pub use vyre_foundation::ir::node_op_id;
use vyre_foundation::ir::Node;
use vyre_foundation::ir::{OpId, Program};
use vyre_foundation::validate::ValidationError;
use vyre_foundation::visit::child_bodies;

const CORE_SUPPORTED_OP_IDS: &[&str] = &[
    "vyre.node.let",
    "vyre.node.assign",
    "vyre.node.store",
    "vyre.node.if",
    "vyre.node.loop",
    "vyre.node.return",
    "vyre.node.block",
    "vyre.node.barrier",
    "vyre.node.indirect_dispatch",
    "vyre.node.async_load",
    "vyre.node.async_store",
    "vyre.node.async_wait",
    "vyre.node.resume",
    "vyre.node.tile_decl",
    "vyre.node.tile_load",
    "vyre.node.tile_store",
    "vyre.node.tile_matmul",
    "vyre.node.tile_reduce",
    "vyre.node.tile_elementwise",
    "vyre.node.region",
    "vyre.lit_u32",
    "vyre.lit_i32",
    "vyre.lit_f32",
    "vyre.lit_bool",
    "vyre.var",
    "vyre.bin_op",
    "vyre.un_op",
    "vyre.load",
    "vyre.store",
];

/// Validate that `backend` supports every operation in `program`.
///
/// # Errors
///
/// Returns the first node, in depth-first source order, whose operation the
/// backend does not declare support for.
pub fn validate_program(program: &Program, backend: &dyn Backend) -> Result<(), ValidationError> {
    validate_nodes(program.entry(), backend.id(), backend.supported_ops())
}

/// Default core operation support set for legacy backends.
pub fn default_supported_ops() -> &'static std::collections::HashSet<OpId> {
    static OPS: LazyLock<std::collections::HashSet<OpId>> = LazyLock::new(|| {
        let mut ops = std::collections::HashSet::with_capacity(CORE_SUPPORTED_OP_IDS.len());
        ops.extend(CORE_SUPPORTED_OP_IDS.iter().copied().map(Arc::<str>::from));
        ops
    });
    &OPS
}

/// Default core operation set plus `Node::Trap`.
///
/// `Trap` is a structural control-flow node, not a concrete-driver extension:
/// backends that lower it as lane termination should use this shared set
/// instead of carrying a backend-local static and literal allocation.
pub fn default_supported_ops_with_trap() -> &'static std::collections::HashSet<OpId> {
    static OPS: LazyLock<std::collections::HashSet<OpId>> = LazyLock::new(|| {
        let base = default_supported_ops();
        let mut ops = std::collections::HashSet::with_capacity(base.len().saturating_add(1));
        ops.extend(base.iter().cloned());
        ops.insert(Arc::<str>::from("vyre.node.trap"));
        ops
    });
    &OPS
}

/// The first node whose operation `supported` does not contain, paired with
/// its position in the body that holds it.
///
/// Child bodies come from [`child_bodies`], the shared-read descent owner in
/// `vyre-foundation`, so this crate does not restate which `Node` variants
/// nest. The hand-written match this replaces ended in a catch-all arm, and
/// `Node` is `#[non_exhaustive]`: a variant added upstream landed there as a
/// transparent leaf, so an unsupported operation buried in its body validated
/// clean and reached the backend anyway.
///
/// Dispatch admission and the target-facet join both ask this question, so
/// they ask it here. Two walks would let a backend advertise a facet for an
/// operation its own dispatch path refuses.
pub(crate) fn first_unsupported_node_op<'a>(
    nodes: &'a [Node],
    supported: &std::collections::HashSet<OpId>,
) -> Option<(&'static str, usize)> {
    let mut stack: Vec<(&'a Node, usize)> = Vec::with_capacity(nodes.len());
    stack.extend(
        nodes
            .iter()
            .enumerate()
            .rev()
            .map(|(index, node)| (node, index)),
    );
    while let Some((node, index)) = stack.pop() {
        let op = node_op_id(node);
        if !supported.contains(op) {
            return Some((op, index));
        }
        // Groups in reverse, each reversed, so `then` pops before `otherwise`
        // and both in source order: the same visit order as the recursion.
        for body in child_bodies(node).into_iter().rev() {
            stack.extend(
                body.iter()
                    .enumerate()
                    .rev()
                    .map(|(index, node)| (node, index)),
            );
        }
    }
    None
}

/// Check every node in `nodes` and in every body nested under it.
///
/// `index` is the node's position in the body that holds it, matching the
/// error the recursive walk reported.
fn validate_nodes(
    nodes: &[Node],
    backend: &'static str,
    supported: &std::collections::HashSet<OpId>,
) -> Result<(), ValidationError> {
    match first_unsupported_node_op(nodes, supported) {
        Some((op, index)) => Err(ValidationError::unsupported_op(
            backend,
            &Arc::<str>::from(op),
            index,
        )),
        None => Ok(()),
    }
}
