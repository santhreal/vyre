//! Shared recursive inspection helpers for statement IR nodes.

use vyre_foundation::ir::Node;
use vyre_foundation::visit::any_descendant;

/// What kind of workgroup rendezvous a branch body can reach.
///
/// A lane that enters a branch containing one of these waits for peers that a
/// diverged lane never sends. The two are one defect class and differ only in
/// what the diagnostic names.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RendezvousKind {
    /// `Node::Barrier` or `Node::LogicalBarrier`.
    Barrier,
    /// A statement whose own operands read peer lanes.
    ///
    /// Only reachable with `subgroup-ops`, which is what defines the
    /// collective expressions in the first place.
    #[cfg(feature = "subgroup-ops")]
    SubgroupCollective,
}

impl RendezvousKind {
    /// The construct as it is named in a diagnostic.
    pub(crate) const fn describe(self) -> &'static str {
        match self {
            Self::Barrier => "Barrier",
            #[cfg(feature = "subgroup-ops")]
            Self::SubgroupCollective => "a subgroup collective",
        }
    }
}

/// Which rendezvous a branch body can reach, at any depth, or `None`.
///
/// A barrier is reported ahead of a collective when a body contains both,
/// because the two carry the same rule and only the wording differs.
pub(crate) fn contains_rendezvous(nodes: &[Node]) -> Option<RendezvousKind> {
    if nodes.iter().any(node_contains_barrier) {
        return Some(RendezvousKind::Barrier);
    }
    #[cfg(feature = "subgroup-ops")]
    if nodes.iter().any(|node| {
        any_descendant(node, &mut |candidate: &Node| {
            node_reads_peer_lanes(candidate)
        })
    }) {
        return Some(RendezvousKind::SubgroupCollective);
    }
    None
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
