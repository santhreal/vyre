//! Child body access and mapping for `Node` variants.

use std::borrow::Cow;
use std::sync::Arc;

use smallvec::SmallVec;

use crate::ir_inner::model::expr::Expr;
use crate::ir_inner::model::node::Node;
use crate::transform::rewrite_walk;

/// Every child node body of `node`, in source order.
///
/// This is the ONE owner of the question "which node variants contain other
/// nodes". Adding a `Node` variant fails to compile here, and that failure is
/// the mechanism that keeps every traversal in the workspace correct.
///
/// A traversal that re-derives this with its own `match node` ending in
/// `_ => false` silently classifies a new nesting variant as a leaf. In
/// `validate::barrier` that is a correctness bug rather than a missed
/// optimization: a barrier hidden inside an unrecognised variant makes an exit
/// look ordered when it is not.
///
/// Leaves return two empty slices, so a caller can flatten unconditionally.
/// Only `Node::If` uses both groups.
#[inline]
#[must_use]
pub fn child_bodies(node: &Node) -> [&[Node]; 2] {
    match node {
        Node::If {
            then, otherwise, ..
        } => [then, otherwise],
        Node::Loop { body, .. } => [body, &[]],
        Node::Block(nodes) => [nodes, &[]],
        Node::Region { body, .. } => [body.as_slice(), &[]],
        Node::TileElementwise { body, .. } => [body.as_slice(), &[]],
        Node::Let { .. }
        | Node::Assign { .. }
        | Node::Store { .. }
        | Node::Return
        | Node::Barrier { .. }
        | Node::LogicalBarrier { .. }
        | Node::IndirectDispatch { .. }
        | Node::AllReduce { .. }
        | Node::AllGather { .. }
        | Node::ReduceScatter { .. }
        | Node::Broadcast { .. }
        | Node::AsyncLoad { .. }
        | Node::AsyncStore { .. }
        | Node::AsyncWait { .. }
        | Node::Trap { .. }
        | Node::Resume { .. }
        | Node::TileLoad { .. }
        | Node::TileStore { .. }
        | Node::TileMatmul { .. }
        | Node::TileReduce { .. }
        | Node::TileDecl { .. }
        | Node::Opaque(_) => [&[], &[]],
    }
}

/// Every child node body of `node`, in source order, as slots a caller can
/// take a body out of or write a new one into.
///
/// The unique-reference owner of "which body slots does this variant have".
/// [`child_bodies`] answers the same question for a read-only scan and
/// [`rewrite_walk::rewrite_node`] answers it for a rebuild that must preserve
/// an unchanged borrow. Neither can hand back a slot to MOVE a body out of, so
/// an owning map has to re-derive the slot list, and
/// [`node_map::map_body`](crate::visit::node_map::map_body) did exactly that
/// with a list ending in `other => other`. A body-bearing variant that list had
/// not been told about came back unchanged, so a pass composed on it was a
/// silent no-op for that variant instead of an error.
///
/// Only the slots the variant really has are returned, so a one-slot variant is
/// never handed the empty padding [`child_bodies`] adds to its answer. A
/// `Node::Region` body is shared through an `Arc` and is cloned here only when
/// another owner still holds it.
///
/// Exhaustive with no catch-all arm, deliberately, for the same reason
/// [`child_bodies`] is.
#[must_use]
pub fn child_bodies_mut(node: &mut Node) -> SmallVec<[&mut Vec<Node>; 2]> {
    let mut slots: SmallVec<[&mut Vec<Node>; 2]> = SmallVec::new();
    match node {
        Node::If {
            then, otherwise, ..
        } => {
            slots.push(then);
            slots.push(otherwise);
        }
        Node::Loop { body, .. } | Node::Block(body) | Node::TileElementwise { body, .. } => {
            slots.push(body)
        }
        Node::Region { body, .. } => slots.push(Arc::make_mut(body)),
        Node::Let { .. }
        | Node::Assign { .. }
        | Node::Store { .. }
        | Node::Return
        | Node::Barrier { .. }
        | Node::LogicalBarrier { .. }
        | Node::IndirectDispatch { .. }
        | Node::AllReduce { .. }
        | Node::AllGather { .. }
        | Node::ReduceScatter { .. }
        | Node::Broadcast { .. }
        | Node::AsyncLoad { .. }
        | Node::AsyncStore { .. }
        | Node::AsyncWait { .. }
        | Node::Trap { .. }
        | Node::Resume { .. }
        | Node::TileLoad { .. }
        | Node::TileStore { .. }
        | Node::TileMatmul { .. }
        | Node::TileReduce { .. }
        | Node::TileDecl { .. }
        | Node::Opaque(_) => {}
    }
    slots
}

/// `node` with each body slot replaced by `map`'s result for that slot.
///
/// The change-reporting rebuild, for a caller holding a borrow that must not
/// pay for a clone when nothing changed.
///
/// Which slots the variant has, and how the node is put back together from new
/// ones, is [`rewrite_walk::rewrite_node`]'s decision: it offers exactly the
/// real slots, in source order, and clones the variant's own operands rather
/// than its bodies, so an unchanged subtree costs nothing. Its hook cannot
/// carry the caller's borrow, so the slice handed to `map` comes from
/// [`child_bodies`] instead and the two are matched by position. A one-slot
/// variant therefore never sees the empty second slice `child_bodies` pads its
/// answer with: a rule that rewrites a whole body would otherwise be handed a
/// body that does not exist.
#[must_use]
pub(crate) fn map_bodies_cow<'a>(
    node: &'a Node,
    map: &mut impl FnMut(&'a [Node]) -> Cow<'a, [Node]>,
) -> Cow<'a, Node> {
    struct MapBodies<'a, 'm, M> {
        slots: [&'a [Node]; 2],
        next: usize,
        map: &'m mut M,
    }

    impl<'a, M> rewrite_walk::NodeRewrite for MapBodies<'a, '_, M>
    where
        M: FnMut(&'a [Node]) -> Cow<'a, [Node]>,
    {
        fn operand(&mut self, _expr: &Expr) -> Option<Expr> {
            None
        }

        fn body(&mut self, _parent: &Node, body: &[Node]) -> Option<Vec<Node>> {
            let slot = self.slots[self.next];
            debug_assert!(
                std::ptr::eq(slot.as_ptr(), body.as_ptr()) && slot.len() == body.len(),
                "rewrite_node offered body slot {} that child_bodies does not report there",
                self.next
            );
            self.next += 1;
            match (self.map)(slot) {
                Cow::Borrowed(_) => None,
                Cow::Owned(rewritten) => Some(rewritten),
            }
        }
    }

    let mut mapper = MapBodies {
        slots: child_bodies(node),
        next: 0,
        map,
    };
    rewrite_walk::rewrite_node(node, &mut mapper).map_or(Cow::Borrowed(node), Cow::Owned)
}
