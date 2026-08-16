//! What a `Node` variant carries.
//!
//! The per-variant decisions a traversal needs and cannot re-derive safely:
//! which bodies a variant nests, which scalar name it binds and what it does to
//! that name, which operand expressions it evaluates, and which buffers it
//! names by direction. Every match here is exhaustive with no catch-all arm, so
//! adding a `Node` variant is a compile error in this file rather than a silent
//! leaf classification somewhere downstream.

use crate::ir_inner::model::expr::Expr;
use crate::ir_inner::model::expr::Ident;
use crate::ir_inner::model::node::Node;
use crate::transform::rewrite_walk;
use smallvec::SmallVec;
use std::borrow::Cow;
use std::sync::Arc;

/// What a [`Node`] variant carries, and therefore what a traversal owes it.
///
/// This is the RECORDED DECISION for every variant, and it exists because
/// `Node` is `#[non_exhaustive]`: no crate other than this one can write an
/// exhaustive match, so every traversal downstream ends in a catch-all arm and
/// a new variant lands in that arm without anybody choosing it. The match in
/// [`node_shape`] is the one place where adding a variant is a compile error,
/// so the decision has to be made once, here, in the same patch that adds it.
///
/// The run-time half of the same property is
/// [`NODE_VARIANT_NAMES`](crate::ir::NODE_VARIANT_NAMES): a test can enumerate
/// every declared variant, ask for its shape, and refuse to pass until a
/// fixture and a traversal decision exist for each one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeShape {
    /// The variant nests child node bodies, enumerated by [`child_bodies`].
    ///
    /// A recursive traversal MUST descend. Treating one of these as a leaf
    /// skips a whole subtree, and a barrier, store, or early exit inside that
    /// subtree then reads as absent rather than as unknown.
    pub nests_nodes: bool,
    /// The variant owns operand expressions reachable from [`super::walk_exprs`].
    ///
    /// An analysis that collects buffer reads, variable uses, or literal
    /// operands must visit them or it under-reports the node's effects.
    pub carries_operands: bool,
    /// The payload is an out-of-tree extension whose contents core cannot
    /// enumerate, so an analysis must treat it as unknown rather than as empty.
    pub opaque_payload: bool,
}

impl NodeShape {
    const INERT: Self = Self {
        nests_nodes: false,
        carries_operands: false,
        opaque_payload: false,
    };
    const OPERANDS: Self = Self {
        nests_nodes: false,
        carries_operands: true,
        opaque_payload: false,
    };
    const BODIES: Self = Self {
        nests_nodes: true,
        carries_operands: false,
        opaque_payload: false,
    };
    const BODIES_AND_OPERANDS: Self = Self {
        nests_nodes: true,
        carries_operands: true,
        opaque_payload: false,
    };
    const OPAQUE: Self = Self {
        nests_nodes: false,
        carries_operands: false,
        opaque_payload: true,
    };
}

/// The recorded traversal decision for the variant `node` holds.
///
/// Exhaustive with no catch-all arm, deliberately. Adding a `Node` variant
/// fails to compile here, and that failure is the point: it forces the author
/// to say whether the new variant nests bodies, owns operands, or is opaque,
/// before any traversal downstream can silently classify it as a leaf.
#[inline]
#[must_use]
pub fn node_shape(node: &Node) -> NodeShape {
    match node {
        Node::If { .. } | Node::Loop { .. } => NodeShape::BODIES_AND_OPERANDS,
        Node::Block(_) | Node::Region { .. } => NodeShape::BODIES,
        Node::Let { .. }
        | Node::Assign { .. }
        | Node::Store { .. }
        | Node::AsyncLoad { .. }
        | Node::AsyncStore { .. }
        | Node::Trap { .. } => NodeShape::OPERANDS,
        Node::Return
        | Node::Barrier { .. }
        | Node::IndirectDispatch { .. }
        | Node::AsyncWait { .. }
        | Node::Resume { .. }
        | Node::AllReduce { .. }
        | Node::AllGather { .. }
        | Node::ReduceScatter { .. }
        | Node::Broadcast { .. } => NodeShape::INERT,
        Node::Opaque(_) => NodeShape::OPAQUE,
    }
}

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
        Node::Region { body, .. } => [body, &[]],
        Node::Let { .. }
        | Node::Assign { .. }
        | Node::Store { .. }
        | Node::Return
        | Node::Barrier { .. }
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
        Node::Loop { body, .. } | Node::Block(body) => slots.push(body),
        Node::Region { body, .. } => slots.push(Arc::make_mut(body)),
        Node::Let { .. }
        | Node::Assign { .. }
        | Node::Store { .. }
        | Node::Return
        | Node::Barrier { .. }
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

/// What a statement does to the scalar name it carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NameBinding {
    /// Introduces the name into the enclosing scope, bound to the value in the
    /// first operand slot. `Node::Let`.
    Declare,
    /// Rebinds a name the enclosing scope already declares, to the value in the
    /// first operand slot. `Node::Assign`.
    Reassign,
    /// Introduces a loop induction variable: a counter the loop itself drives,
    /// with no value operand. `Node::Loop`.
    Induction,
}

/// Everything `node` carries in the scalar namespace: the name it binds, what
/// it does to that name, and the operand expressions it evaluates.
///
/// One record rather than two answers because both come from the same
/// per-variant decision. A scope pass that asks only "which name" and a rewrite
/// that asks only "which operands" used to read two separate enumerations of
/// the same enum, and two enumerations are two chances to forget a variant.
#[derive(Debug, Clone, Copy)]
pub struct NodeScalars<'a> {
    /// The name the statement binds, and what it does to it.
    ///
    /// `Node::AsyncLoad` and the collectives name buffers, which is a different
    /// namespace answered by [`node_buffer_refs`].
    pub binding: Option<(NameBinding, &'a Ident)>,
    /// Operand expressions in source order, padded with `None`, so a caller can
    /// flatten unconditionally. The widest variants carry exactly two: `Store`
    /// (index, value), `Loop` (from, to), and the async copies (offset, size).
    pub operands: [Option<&'a Expr>; 2],
}

impl<'a> NodeScalars<'a> {
    const NONE: Self = Self {
        binding: None,
        operands: [None, None],
    };

    const fn operands_only(operands: [Option<&'a Expr>; 2]) -> Self {
        Self {
            binding: None,
            operands,
        }
    }
}

/// The scalar name and operand positions of `node`.
///
/// This is the ONE owner of both questions, and the match has no catch-all arm.
/// Adding a `Node` variant fails to compile here, so a variant that gains an
/// operand position cannot be skipped by a scan or a rewrite in silence, and a
/// variant that gains a binding position cannot let a pass hoist, fuse, or
/// inline across a live rebinding while every existing test still passes. The
/// hand-written versions this replaces each ended in a `_` arm reporting
/// "carries nothing".
#[inline]
#[must_use]
pub fn node_scalars(node: &Node) -> NodeScalars<'_> {
    match node {
        Node::Let { name, value } => NodeScalars {
            binding: Some((NameBinding::Declare, name)),
            operands: [Some(value), None],
        },
        Node::Assign { name, value } => NodeScalars {
            binding: Some((NameBinding::Reassign, name)),
            operands: [Some(value), None],
        },
        Node::Loop { var, from, to, .. } => NodeScalars {
            binding: Some((NameBinding::Induction, var)),
            operands: [Some(from), Some(to)],
        },
        Node::Store { index, value, .. } => NodeScalars::operands_only([Some(index), Some(value)]),
        Node::If { cond, .. } => NodeScalars::operands_only([Some(cond), None]),
        Node::AsyncLoad { offset, size, .. } | Node::AsyncStore { offset, size, .. } => {
            NodeScalars::operands_only([Some(offset), Some(size)])
        }
        Node::Trap { address, .. } => NodeScalars::operands_only([Some(address), None]),
        Node::Block(_)
        | Node::Region { .. }
        | Node::Return
        | Node::Barrier { .. }
        | Node::IndirectDispatch { .. }
        | Node::AllReduce { .. }
        | Node::AllGather { .. }
        | Node::ReduceScatter { .. }
        | Node::Broadcast { .. }
        | Node::AsyncWait { .. }
        | Node::Resume { .. }
        | Node::Opaque(_) => NodeScalars::NONE,
    }
}

/// Every operand expression `node` carries directly, in source order.
///
/// The operand half of [`node_scalars`], for the many callers that do not care
/// which name the statement binds. Leaves return two `None`s.
#[inline]
#[must_use]
pub fn node_operands(node: &Node) -> [Option<&Expr>; 2] {
    node_scalars(node).operands
}

/// The scalar name a statement binds, or `None` when it binds nothing.
///
/// The name half of [`node_scalars`], for a caller that needs the name but not
/// what the statement does to it. A caller that must tell a fresh declaration
/// from a rebinding reads [`NodeScalars::binding`] instead: `visit::bound_names`
/// collects only the declaring forms, because a `Node::Assign` writes a name the
/// enclosing scope already declares and counting it as a second declaration
/// makes a scope-extension pass refuse a legal rewrite.
#[inline]
#[must_use]
pub fn node_bound_name(node: &Node) -> Option<&Ident> {
    node_scalars(node).binding.map(|(_, name)| name)
}

/// The stream tag a node names, or `None` when it names none.
///
/// The ONE owner of the tag namespace, which is neither the value namespace
/// [`node_scalars`] answers nor the buffer namespace [`node_buffer_refs`]
/// answers. A tag names an in-flight asynchronous transfer: the start that
/// opens it and the wait that closes it carry the same tag, and that pairing
/// is what `validate::async_pipeline` reads. A pass that renames a value must
/// therefore leave tags alone, and a pass that renames a tag must rewrite both
/// ends of the pair.
///
/// Exhaustive with no catch-all arm, deliberately, for the same reason
/// [`node_scalars`] is: a variant that gains a tag position cannot be left out
/// of a rename in silence, which would separate a start from its wait and make
/// the pipeline analysis read a transfer nothing waits for.
#[inline]
#[must_use]
pub fn node_tag(node: &Node) -> Option<&Ident> {
    match node {
        Node::AsyncLoad { tag, .. }
        | Node::AsyncStore { tag, .. }
        | Node::AsyncWait { tag }
        | Node::Trap { tag, .. }
        | Node::Resume { tag } => Some(tag),
        Node::Let { .. }
        | Node::Assign { .. }
        | Node::Store { .. }
        | Node::If { .. }
        | Node::Loop { .. }
        | Node::Block(_)
        | Node::Region { .. }
        | Node::Return
        | Node::Barrier { .. }
        | Node::IndirectDispatch { .. }
        | Node::AllReduce { .. }
        | Node::AllGather { .. }
        | Node::ReduceScatter { .. }
        | Node::Broadcast { .. }
        | Node::Opaque(_) => None,
    }
}

/// The buffers a node names directly, split by direction.
#[derive(Debug, Clone, Copy)]
pub struct BufferRefs<'a> {
    /// Buffers read by name, in source order.
    pub reads: [Option<&'a Ident>; 2],
    /// Buffers written by name, in source order.
    pub writes: [Option<&'a Ident>; 2],
    /// False when the node carries an opaque payload whose buffer references
    /// core cannot enumerate, which makes the two arrays a LOWER BOUND. A
    /// caller whose answer has to be sound must then treat the node as touching
    /// every buffer rather than none.
    pub complete: bool,
}

impl<'a> BufferRefs<'a> {
    const NONE: Self = Self {
        reads: [None, None],
        writes: [None, None],
        complete: true,
    };

    const fn read(buffer: &'a Ident) -> Self {
        Self {
            reads: [Some(buffer), None],
            ..Self::NONE
        }
    }

    const fn write(buffer: &'a Ident) -> Self {
        Self {
            writes: [Some(buffer), None],
            ..Self::NONE
        }
    }

    const fn read_write(read: &'a Ident, write: &'a Ident) -> Self {
        Self {
            reads: [Some(read), None],
            writes: [Some(write), None],
            complete: true,
        }
    }
}

/// Which buffers `node` names, and in which direction.
///
/// This is the ONE owner of "what does this statement do to a buffer BY NAME".
/// A buffer reached through an operand expression is not here: that is
/// [`node_operands`] followed by [`super::expr_buffer_ref`], and the two answers
/// compose. Adding a `Node` variant fails to compile here.
///
/// The four collective variants are the reason this exists. They name their
/// operands as buffers and carry no operand expression at all, so every
/// dependency walk that answered this question with a per-variant match ending
/// in `_ => {}` reported that an `AllReduce` touches nothing.
#[must_use]
pub fn node_buffer_refs(node: &Node) -> BufferRefs<'_> {
    match node {
        Node::Store { buffer, .. } => BufferRefs::write(buffer),
        Node::AsyncLoad {
            source,
            destination,
            ..
        }
        | Node::AsyncStore {
            source,
            destination,
            ..
        } => BufferRefs::read_write(source, destination),
        Node::IndirectDispatch { count_buffer, .. } => BufferRefs::read(count_buffer),
        // In place on every rank: each contributes its own copy of `buffer` and
        // receives the combined one. `Broadcast` reads it on the root rank and
        // writes it on the others, which is the same pair of names.
        Node::AllReduce { buffer, .. } | Node::Broadcast { buffer, .. } => {
            BufferRefs::read_write(buffer, buffer)
        }
        Node::AllGather { input, output, .. } | Node::ReduceScatter { input, output, .. } => {
            BufferRefs::read_write(input, output)
        }
        Node::Let { .. }
        | Node::Assign { .. }
        | Node::If { .. }
        | Node::Loop { .. }
        | Node::Trap { .. }
        | Node::AsyncWait { .. }
        | Node::Resume { .. }
        | Node::Return
        | Node::Barrier { .. }
        | Node::Block(_)
        | Node::Region { .. } => BufferRefs::NONE,
        Node::Opaque(_) => BufferRefs {
            complete: false,
            ..BufferRefs::NONE
        },
    }
}
