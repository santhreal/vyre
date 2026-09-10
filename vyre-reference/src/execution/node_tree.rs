//! Shared recursive inspection helpers for statement IR nodes.

use vyre_foundation::ir::Node;
use vyre_foundation::visit::any_descendant;

/// Whether any statement in `nodes` may reach a `Barrier`, at any depth.
pub(crate) fn contains_barrier(nodes: &[Node]) -> bool {
    nodes.iter().any(node_contains_barrier)
}

/// True when `node` or anything under it is a barrier.
///
/// Child enumeration comes from
/// `vyre_foundation::visit::child_bodies`, the one exhaustive owner
/// of which `Node` variants contain other nodes. This function used to name the
/// nesting variants itself, its doc comment claimed the match was exhaustive,
/// and it was not: `Node::Region` fell through to `_ => false`. Since
/// `Program::wrapped` puts the whole entry sequence inside a Region, a barrier
/// reached only through a Region body read as ABSENT.
fn node_contains_barrier(node: &Node) -> bool {
    any_descendant(node, &mut |candidate| {
        matches!(
            candidate,
            Node::Barrier { .. } | Node::LogicalBarrier { .. }
        )
    })
}

/// Stable per-process identifier for a borrowed `Node`.
pub(crate) fn node_id(node: &Node) -> usize {
    std::ptr::from_ref(node).addr()
}

/// Whether `node` evaluates a subgroup collective in its OWN operands.
///
/// Child bodies are excluded on purpose: a nested statement is stepped as its
/// own node and rendezvouses then, so counting a collective under an `If` or a
/// `Loop` would hold every lane at the enclosing statement instead of at the
/// collective itself.
#[cfg(feature = "subgroup-ops")]
pub(crate) fn node_reads_peer_lanes(node: &Node) -> bool {
    use vyre_foundation::ir::Expr;
    use vyre_foundation::visit::{any_subexpr, node_operands, node_variadic_operands};

    let mut is_collective = |candidate: &Expr| {
        matches!(
            candidate,
            Expr::SubgroupShuffle { .. }
                | Expr::SubgroupBallot { .. }
                | Expr::SubgroupReduce { .. }
        )
    };
    node_operands(node)
        .into_iter()
        .flatten()
        .chain(node_variadic_operands(node))
        .any(|operand| any_subexpr(operand, &mut is_collective))
}
