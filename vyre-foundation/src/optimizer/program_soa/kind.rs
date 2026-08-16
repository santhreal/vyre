//! The tags every fact column is keyed by.
//!
//! A node's kind and a buffer reference's kind are one byte each, and the
//! preorder index is what every column and every query hands back. They are
//! read by the build walk and by every query, so they sit under neither.

use crate::ir::AtomicOp;

/// Stable preorder index into the `ProgramFacts` columns. Distinct
/// programs (or rebuilt fact tables for the same program) generally
/// hash to distinct sequences of indices; do not persist these
/// across `Program::with_rewritten_entry` calls.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct NodeIndex(pub u32);

/// Compact 1-byte tag mirroring every `Node` variant. The
/// discriminant order matches the order in
/// `vyre-foundation/src/ir_inner/model/generated.rs::Node`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum NodeKind {
    /// `Node::Let { name, value }`.
    Let,
    /// `Node::Assign { name, value }`.
    Assign,
    /// `Node::Store { buffer, index, value }`.
    Store,
    /// `Node::If { cond, then, otherwise }`.
    If,
    /// `Node::Loop { var, from, to, body }`.
    Loop,
    /// `Node::IndirectDispatch { count_buffer, .. }`.
    IndirectDispatch,
    /// `Node::AsyncLoad { source, destination, .. }`.
    AsyncLoad,
    /// `Node::AsyncStore { source, destination, .. }`.
    AsyncStore,
    /// `Node::AsyncWait { tag }`.
    AsyncWait,
    /// `Node::Trap { address, tag }`.
    Trap,
    /// `Node::Resume { tag }`.
    Resume,
    /// `Node::Return`.
    Return,
    /// `Node::Barrier { ordering }`.
    Barrier,
    /// `Node::Block(body)`.
    Block,
    /// `Node::Region { generator, source_region, body }`.
    Region,
    /// `Node::AllReduce { .. }`.
    AllReduce,
    /// `Node::AllGather { .. }`.
    AllGather,
    /// `Node::ReduceScatter { .. }`.
    ReduceScatter,
    /// `Node::Broadcast { .. }`.
    Broadcast,
    /// `Node::Opaque(extension)`.
    Opaque,
}

/// How a buffer was touched at a given node. Drives alias-aware
/// queries that need to distinguish reads from writes from atomics.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum BufferRefKind {
    /// `Expr::Load { buffer, .. }`, `Expr::BufLen { buffer }`, or
    /// any read-side reference inside another expression.
    Read,
    /// `Node::Store { buffer, .. }`, `Node::AsyncStore.destination`,
    /// or any write-side reference.
    Write,
    /// `Expr::Atomic { buffer, op, .. }`  -  both a read and a write
    /// in one operation, with explicit memory ordering.
    Atomic(AtomicOp),
    /// `Node::AsyncLoad.destination`  -  the destination of an async
    /// copy is treated as a write target.
    AsyncDestination,
    /// `Node::AsyncLoad.source` / `Node::AsyncStore.source`  -  async
    /// copy sources are read targets.
    AsyncSource,
    /// `Node::IndirectDispatch.count_buffer`  -  read-side reference
    /// to a dispatch-grid buffer.
    IndirectCount,
}

/// Bit position of a `NodeKind` inside the `ProgramFacts::kinds_present`
/// bitset. Returned as a `u32` so callers can `1 << kind_bit(k)` directly.
#[must_use]
#[inline]
pub const fn kind_bit(kind: NodeKind) -> u32 {
    kind as u32
}

/// `1 << kind_bit(k)` mask for one [`NodeKind`].
#[must_use]
#[inline]
pub const fn kind_mask(kind: NodeKind) -> u32 {
    1u32 << (kind as u32)
}
