//! The one owner of "may a node standing between two accesses to `buffer`
//! disturb the answer".
//!
//! `dead_store_elim` and `store_to_load_forward` are the same shape: find two
//! accesses to one buffer in one sibling list, prove nothing in the gap gets in
//! the way, rewrite. Only the second step is analysis, and it was written twice.
//! The two copies had already drifted on two questions, and the drift was
//! invisible because each copy was exhaustive, tested, and internally coherent:
//!
//! - a foreign `Expr::Atomic` compare-exchange is a lock acquisition, so a
//!   concurrent invocation may write `buffer` across it. `dead_store_elim`
//!   refused to look past one; `store_to_load_forward` forwarded a stale value
//!   through it.
//! - `Node::Trap` runs a host effect handler that may touch any buffer and
//!   `Node::IndirectDispatch` launches a grid that may touch any buffer. Both
//!   copies inspected only the one buffer each names, while both pass module
//!   docs promised the node blocks outright.
//!
//! What is genuinely per-pass is a single bit, [`Interference`]: whether a write
//! to the buffer interferes, or only a read.

use crate::ir::{Expr, Ident, Node};
use crate::visit::{any_descendant, node_buffer_refs, node_operands};

/// Which accesses to a buffer a memory pass cannot look past.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Interference {
    /// Only a read interferes. A `Node::Store` to the buffer whose index and
    /// value never read it is a blind overwrite: it cannot observe the bytes
    /// already there, so it cannot keep an earlier write to the same lane
    /// alive. `dead_store_elim` asks this question.
    Reads,
    /// A read or a write interferes. `store_to_load_forward` asks this one: two
    /// indices that are only structurally distinct may name the same lane at
    /// run time, so a blind overwrite can still clobber the forwarded bytes.
    ReadsAndWrites,
}

/// True when `node` or anything nested in it interferes with `buffer`.
pub(super) fn node_interferes(node: &Node, buffer: &Ident, question: Interference) -> bool {
    any_descendant(node, &mut |current| {
        direct_interference(current, buffer, question)
    })
}

/// True when any node in `nodes`, at any depth, interferes with `buffer`.
pub(super) fn any_node_interferes(nodes: &[Node], buffer: &Ident, question: Interference) -> bool {
    nodes
        .iter()
        .any(|node| node_interferes(node, buffer, question))
}

/// True when `expr` reads, writes, or measures `buffer`, at any depth.
///
/// Operand positions come from `visit::expr_children`, so a new
/// operand-carrying `Expr` variant cannot hide an access from the proofs that
/// call this. `Expr::Opaque` answers `true` because its buffer effect is
/// unnameable. A compare-exchange against ANY buffer answers `true` because it
/// is how a lock is taken: past a successful one, another invocation's writes
/// to `buffer` are visible even though the atomic never names it.
pub(super) fn expr_touches_buffer(expr: &Expr, buffer: &Ident) -> bool {
    use crate::ir::AtomicOp;
    crate::visit::any_subexpr(expr, &mut |candidate| match candidate {
        Expr::Load { buffer: other, .. }
        | Expr::BufLen { buffer: other }
        | Expr::BufferRef { buffer: other } => other == buffer,
        Expr::Atomic {
            buffer: other, op, ..
        } => {
            other == buffer
                || matches!(
                    op,
                    AtomicOp::CompareExchange | AtomicOp::CompareExchangeWeak
                )
        }
        Expr::Opaque(_) => true,
        _ => false,
    })
}

/// True when `node` itself interferes, ignoring its nested bodies.
fn direct_interference(node: &Node, buffer: &Ident, question: Interference) -> bool {
    if has_unnameable_effect(node) {
        return true;
    }
    let refs = node_buffer_refs(node);
    if refs
        .reads
        .into_iter()
        .flatten()
        .any(|named| named == buffer)
    {
        return true;
    }
    // A `Store` is the one write whose extent is a single lane named by an
    // operand, which is what lets `Interference::Reads` look past it. Every
    // other write reaching here copies or reduces over an extent no operand
    // states, so no pass may look past it.
    let write_is_a_blind_lane =
        question == Interference::Reads && matches!(node, Node::Store { .. });
    if !write_is_a_blind_lane
        && refs
            .writes
            .into_iter()
            .flatten()
            .any(|named| named == buffer)
    {
        return true;
    }
    node_operands(node)
        .into_iter()
        .flatten()
        .any(|operand| expr_touches_buffer(operand, buffer))
}

/// True when the node's effect on `buffer` is not expressible as a read or a
/// write it names, so no per-buffer proof may look past it.
///
/// The match has no catch-all arm: a new `Node` variant fails to compile here
/// rather than defaulting to "harmless", which is the direction that
/// miscompiles.
fn has_unnameable_effect(node: &Node) -> bool {
    match node {
        // Publishes every prior write and admits every concurrent one.
        Node::Barrier { .. }
        // A grid synchronization point, so it is at least a barrier.
        | Node::AllReduce { .. }
        | Node::AllGather { .. }
        | Node::ReduceScatter { .. }
        | Node::Broadcast { .. }
        // Completes an async copy whose destination this node does not name.
        | Node::AsyncWait { .. }
        // Launches a grid that may touch any buffer.
        | Node::IndirectDispatch { .. }
        // Enters and returns from a host effect handler that may touch any
        // buffer; the address operand names where to trap, not what it reads.
        | Node::Trap { .. }
        | Node::Resume { .. }
        // Control leaves the body: what follows executes in a different run.
        | Node::Return
        // A backend extension whose buffer effect core cannot enumerate.
        | Node::Opaque(_) => true,
        // The rest name every buffer they touch: `node_buffer_refs` for the
        // async copies and the store, `node_operands` for the rest.
        Node::Store { .. }
        | Node::AsyncLoad { .. }
        | Node::AsyncStore { .. }
        | Node::Let { .. }
        | Node::Assign { .. }
        | Node::If { .. }
        | Node::Loop { .. }
        | Node::Block(_)
        | Node::Region { .. } => false,
    }
}
