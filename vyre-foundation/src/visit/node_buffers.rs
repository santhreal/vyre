//! Which buffers a `Node` names, by direction.
//!
//! Split from the rest of the per-variant decisions because the read view and
//! the rename view are the same exhaustive match written twice, once to observe
//! and once to substitute. Both are exhaustive with no catch-all arm, so a new
//! `Node` variant that names a buffer is a compile error here rather than a
//! buffer no rename reaches.

use crate::ir_inner::model::expr::Ident;
use crate::ir_inner::model::node::Node;

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
/// [`super::node_operands`] followed by [`super::expr_buffer_ref`], and the two
/// answers
/// compose. Adding a `Node` variant fails to compile here.
///
/// The four collective variants are the reason this exists. They name their
/// operands as buffers and carry no operand expression at all, so every
/// dependency walk that answered this question with a per-variant match ending
/// in `_ => {}` reported that an `AllReduce` touches nothing.
#[must_use]
pub fn node_buffer_refs(node: &Node) -> BufferRefs<'_> {
    match node {
        Node::Store { buffer, .. } | Node::TileStore { buffer, .. } => BufferRefs::write(buffer),
        Node::TileLoad { buffer, .. } => BufferRefs::read(buffer),
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
        | Node::LogicalBarrier { .. }
        | Node::Block(_)
        | Node::Region { .. }
        | Node::TileMatmul { .. }
        | Node::TileReduce { .. }
        | Node::TileElementwise { .. }
        | Node::TileDecl { .. } => BufferRefs::NONE,
        Node::Opaque(_) => BufferRefs {
            complete: false,
            ..BufferRefs::NONE
        },
    }
}

/// The buffer names a node holds directly, borrowed for replacement.
///
/// Direction is dropped on purpose. A rename replaces a name wherever it
/// appears, and a node that reads and writes one buffer holds that name in a
/// single field, so splitting the answer by direction would hand out two
/// mutable borrows of the same field.
#[derive(Debug)]
pub struct BufferNamesMut<'a> {
    names: [Option<&'a mut Ident>; 2],
    complete: bool,
}

impl<'a> BufferNamesMut<'a> {
    const fn none() -> Self {
        Self {
            names: [None, None],
            complete: true,
        }
    }

    const fn one(buffer: &'a mut Ident) -> Self {
        Self {
            names: [Some(buffer), None],
            complete: true,
        }
    }

    const fn two(first: &'a mut Ident, second: &'a mut Ident) -> Self {
        Self {
            names: [Some(first), Some(second)],
            complete: true,
        }
    }

    /// Every buffer-name position the node holds, in source order.
    pub fn into_names(self) -> impl Iterator<Item = &'a mut Ident> {
        self.names.into_iter().flatten()
    }

    /// False when the node carries an opaque payload whose buffer references
    /// core cannot enumerate, which makes the positions a LOWER BOUND.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        self.complete
    }
}

/// The buffer names `node` holds, borrowed for replacement.
///
/// The write direction of [`node_buffer_refs`], exhaustive for the same
/// reason: a variant that names a buffer and reports none here keeps a
/// reference to a name a rename has already retired.
#[must_use]
pub fn node_buffer_names_mut(node: &mut Node) -> BufferNamesMut<'_> {
    match node {
        Node::Store { buffer, .. }
        | Node::TileStore { buffer, .. }
        | Node::TileLoad { buffer, .. }
        | Node::IndirectDispatch {
            count_buffer: buffer,
            ..
        }
        | Node::AllReduce { buffer, .. }
        | Node::Broadcast { buffer, .. } => BufferNamesMut::one(buffer),
        Node::AsyncLoad {
            source,
            destination,
            ..
        }
        | Node::AsyncStore {
            source,
            destination,
            ..
        } => BufferNamesMut::two(source, destination),
        Node::AllGather { input, output, .. } | Node::ReduceScatter { input, output, .. } => {
            BufferNamesMut::two(input, output)
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
        | Node::LogicalBarrier { .. }
        | Node::Block(_)
        | Node::Region { .. }
        | Node::TileMatmul { .. }
        | Node::TileReduce { .. }
        | Node::TileElementwise { .. }
        | Node::TileDecl { .. } => BufferNamesMut::none(),
        Node::Opaque(_) => BufferNamesMut {
            complete: false,
            ..BufferNamesMut::none()
        },
    }
}
