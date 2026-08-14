//! Visitor for IR traversal.
//!
//! Optimization passes, lowering, and analysis use these utilities to walk
//! the IR tree without manually matching every variant. All traversals are
//! implemented with an explicit stack rather than recursion. This is a
//! critical design choice: it prevents stack overflows when processing deep
//! ASTs (e.g., highly nested `If` or `Block` nodes) during adversarial or
//! extreme workloads.

use crate::ir_inner::model::expr::Expr;
use crate::ir_inner::model::expr::Ident;
use crate::ir_inner::model::node::Node;
use crate::ir_inner::model::program::Program;
use smallvec::SmallVec;
use std::borrow::Cow;
use std::collections::HashSet;
use std::ops::ControlFlow;
use std::sync::Arc;

/// Visitor called for each [`Node`] during [`walk_nodes_and_exprs`].
pub trait NodeVisitor {
    /// Invoked once for every node in the program, in the same order as
    /// [`walk_nodes`].
    fn visit_node(&mut self, node: &Node);
}

/// Visitor called for each [`Expr`] during [`walk_nodes_and_exprs`].
pub trait ExprVisitor {
    /// Invoked once for every expression in the program, in the same order
    /// as [`walk_exprs`].
    fn visit_expr(&mut self, expr: &Expr);
}

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
    /// The variant owns operand expressions reachable from [`walk_exprs`].
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

/// `node` with each body slot replaced by `map`'s result for that slot.
///
/// This is the borrowed, change-reporting counterpart of
/// [`node_map::map_body`](crate::visit::node_map::map_body), and the ONE owner
/// of "which body slots does this variant have" in the rebuild direction.
/// `child_bodies` answers the same question for a read-only scan, but a scan
/// cannot say what to do with a slot it changed, so a rebuild that re-derives
/// the slot list can descend into a body and then drop it.
///
/// A variant with no body is returned borrowed without calling `map`, and a
/// variant with one body has `map` called once: a one-slot variant must not see
/// the empty second slice `child_bodies` pads its answer with, because a rule
/// that rewrites a whole body would then be handed a body that does not exist.
///
/// The node is rebuilt only when a slot came back owned, and the rebuild clones
/// the variant's own operands rather than its bodies, so an unchanged subtree
/// costs nothing.
#[must_use]
pub(crate) fn map_bodies_cow<'a>(
    node: &'a Node,
    map: &mut impl FnMut(&'a [Node]) -> Cow<'a, [Node]>,
) -> Cow<'a, Node> {
    match node {
        Node::If {
            cond,
            then,
            otherwise,
        } => {
            let then_body = map(then);
            let otherwise_body = map(otherwise);
            if matches!(then_body, Cow::Borrowed(_)) && matches!(otherwise_body, Cow::Borrowed(_)) {
                return Cow::Borrowed(node);
            }
            Cow::Owned(Node::If {
                cond: cond.clone(),
                then: then_body.into_owned(),
                otherwise: otherwise_body.into_owned(),
            })
        }
        Node::Loop {
            var,
            from,
            to,
            body,
        } => match map(body) {
            Cow::Borrowed(_) => Cow::Borrowed(node),
            Cow::Owned(body) => Cow::Owned(Node::Loop {
                var: var.clone(),
                from: from.clone(),
                to: to.clone(),
                body,
            }),
        },
        Node::Block(body) => match map(body) {
            Cow::Borrowed(_) => Cow::Borrowed(node),
            Cow::Owned(body) => Cow::Owned(Node::Block(body)),
        },
        Node::Region {
            generator,
            source_region,
            body,
        } => match map(body) {
            Cow::Borrowed(_) => Cow::Borrowed(node),
            Cow::Owned(body) => Cow::Owned(Node::Region {
                generator: generator.clone(),
                source_region: source_region.clone(),
                body: Arc::new(body),
            }),
        },
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
        | Node::Opaque(_) => Cow::Borrowed(node),
    }
}

/// Every operand expression `node` carries directly, in source order.
///
/// This is the ONE owner of the question "which node variants carry
/// expressions", the operand counterpart of [`child_bodies`]. Adding a `Node`
/// variant fails to compile here, so a variant that gains an expression
/// position cannot be skipped by a scan or a rewrite in silence.
///
/// Leaves return two `None`s, so a caller can flatten unconditionally. The
/// widest variants carry exactly two operands: `Store` (index, value), `Loop`
/// (from, to), and the async copies (offset, size).
#[inline]
#[must_use]
pub fn node_operands(node: &Node) -> [Option<&Expr>; 2] {
    match node {
        Node::Let { value, .. } | Node::Assign { value, .. } => [Some(value), None],
        Node::Store { index, value, .. } => [Some(index), Some(value)],
        Node::If { cond, .. } => [Some(cond), None],
        Node::Loop { from, to, .. } => [Some(from), Some(to)],
        Node::AsyncLoad { offset, size, .. } | Node::AsyncStore { offset, size, .. } => {
            [Some(offset), Some(size)]
        }
        Node::Trap { address, .. } => [Some(address), None],
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
        | Node::Opaque(_) => [None, None],
    }
}

/// Every sub-expression of every node in `nodes` and in every nested body.
///
/// The slice-taking companion of [`for_each_node`], for an analysis outside this
/// crate that holds a body rather than a whole program. Node descent comes from
/// [`child_bodies`] and operand positions from [`node_operands`], so a caller
/// asking "does anything in this body mention X" cannot miss a position by
/// naming variants itself.
#[inline]
pub fn for_each_expr<'a>(nodes: &'a [Node], mut f: impl FnMut(&'a Expr)) {
    for_each_node(nodes, |node| {
        for operand in node_operands(node).into_iter().flatten() {
            for_each_subexpr(operand, &mut f);
        }
    });
}

/// Every sub-expression of every node in `nodes`, in source pre-order,
/// stopping at the first `Break`.
///
/// The short-circuiting form of [`for_each_expr`], for a guard that reports the
/// FIRST expression matching a predicate instead of every one. Node descent is
/// [`try_for_each_node`], operand positions are [`node_operands`], and
/// sub-expressions are [`expr_children`], so a guard cannot answer "no such
/// expression anywhere" for a position it never named. A guard that hand-rolls
/// the three enumerations instead reports "clean" for the positions it forgot,
/// which is the failure mode of a fail-closed check: it fails open.
///
/// Both walks are explicit worklists, so an adversarially deep program costs
/// heap rather than native stack.
pub fn try_for_each_expr<B>(
    nodes: &[Node],
    mut f: impl FnMut(&Expr) -> ControlFlow<B>,
) -> ControlFlow<B> {
    let mut stopped: Option<B> = None;
    try_for_each_node(nodes, |node| {
        for operand in node_operands(node).into_iter().flatten() {
            let hit = any_subexpr(operand, &mut |expr| match f(expr) {
                ControlFlow::Continue(()) => false,
                ControlFlow::Break(value) => {
                    stopped = Some(value);
                    true
                }
            });
            if hit {
                return ControlFlow::Break(());
            }
        }
        ControlFlow::Continue(())
    });
    stopped.map_or(ControlFlow::Continue(()), ControlFlow::Break)
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
/// [`node_operands`] followed by [`expr_buffer_ref`], and the two answers
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

/// What an expression does to the buffer it names.
#[derive(Debug, Clone, Copy)]
pub enum ExprBufferRef<'a> {
    /// Names no buffer.
    None,
    /// Reads the named buffer, or reads its metadata.
    Read(&'a Ident),
    /// Reads and writes the named buffer: an atomic read-modify-write.
    ReadWrite(&'a Ident),
    /// An out-of-tree extension, whose buffer references core cannot enumerate.
    /// A caller whose answer has to be sound must treat it as touching every
    /// buffer.
    Unknown,
}

/// The buffer `expr` names, and what it does to it.
///
/// The expression half of [`node_buffer_refs`]. `Expr::Atomic` is the case every
/// buffer-set walk in this crate had recorded as a pure read, which is the
/// direction that loses: a dependency walk that believes an atomic only reads
/// sees no conflict with a store to the same buffer.
#[must_use]
pub fn expr_buffer_ref(expr: &Expr) -> ExprBufferRef<'_> {
    match expr {
        Expr::Atomic { buffer, .. } => ExprBufferRef::ReadWrite(buffer),
        Expr::Load { buffer, .. } | Expr::BufLen { buffer } | Expr::BufferRef { buffer } => {
            ExprBufferRef::Read(buffer)
        }
        Expr::LitU32(_)
        | Expr::LitI32(_)
        | Expr::LitF32(_)
        | Expr::LitBool(_)
        | Expr::Var(_)
        | Expr::InvocationId { .. }
        | Expr::WorkgroupId { .. }
        | Expr::LocalId { .. }
        | Expr::BinOp { .. }
        | Expr::UnOp { .. }
        | Expr::Call { .. }
        | Expr::Select { .. }
        | Expr::Cast { .. }
        | Expr::Fma { .. }
        | Expr::SubgroupBallot { .. }
        | Expr::SubgroupShuffle { .. }
        | Expr::SubgroupReduce { .. }
        | Expr::SubgroupLocalId
        | Expr::SubgroupSize => ExprBufferRef::None,
        Expr::Opaque(_) => ExprBufferRef::Unknown,
    }
}

/// Every operand expression of `expr`, in source order.
///
/// This is the ONE owner of the question "which expression variants contain
/// other expressions", the [`child_bodies`] of the value namespace. Adding an
/// `Expr` variant fails to compile in [`expr_children`], and that failure is
/// the mechanism that keeps every expression walk in the crate correct.
///
/// At most three operands are held inline and the argument list of an
/// [`Expr::Call`] is borrowed as a slice, so enumerating children allocates
/// nothing. The whole record is `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct ExprChildren<'a> {
    /// Fixed operand positions, in source order. `None` is an absent optional
    /// operand (`Expr::Atomic::expected`) and is skipped by [`Self::iter`].
    direct: [Option<&'a Expr>; 3],
    /// Call arguments, in source order. Empty for every other variant.
    args: &'a [Expr],
}

impl<'a> ExprChildren<'a> {
    const NONE: Self = Self {
        direct: [None, None, None],
        args: &[],
    };

    const fn one(first: &'a Expr) -> Self {
        Self {
            direct: [Some(first), None, None],
            args: &[],
        }
    }

    const fn two(first: &'a Expr, second: &'a Expr) -> Self {
        Self {
            direct: [Some(first), Some(second), None],
            args: &[],
        }
    }

    const fn three(first: &'a Expr, second: &'a Expr, third: &'a Expr) -> Self {
        Self {
            direct: [Some(first), Some(second), Some(third)],
            args: &[],
        }
    }

    /// The operands in source order.
    ///
    /// The iterator is double-ended, so a stack-based walk that wants children
    /// popped in source order pushes `iter().rev()`.
    pub fn iter(self) -> impl DoubleEndedIterator<Item = &'a Expr> + Clone {
        self.direct.into_iter().flatten().chain(self.args.iter())
    }
}

/// The operands of `expr`, in source order.
///
/// Exhaustive with no catch-all arm, deliberately. Adding an `Expr` variant
/// fails to compile here, and that failure is the point: it forces the author
/// to say which of the new variant's positions a walk owes a visit. A walk that
/// re-derives this with its own `match expr` ending in `_ => {}` classifies a
/// new variant as a leaf, which is how an operand stops being renamed,
/// substituted, counted as a live use, or folded.
#[inline]
#[must_use]
pub fn expr_children(expr: &Expr) -> ExprChildren<'_> {
    match expr {
        Expr::LitU32(_)
        | Expr::LitI32(_)
        | Expr::LitF32(_)
        | Expr::LitBool(_)
        | Expr::Var(_)
        | Expr::BufferRef { .. }
        | Expr::BufLen { .. }
        | Expr::InvocationId { .. }
        | Expr::WorkgroupId { .. }
        | Expr::LocalId { .. }
        | Expr::SubgroupLocalId
        | Expr::SubgroupSize
        | Expr::Opaque(_) => ExprChildren::NONE,
        Expr::Load { index, .. }
        | Expr::UnOp { operand: index, .. }
        | Expr::Cast { value: index, .. }
        | Expr::SubgroupBallot { cond: index }
        | Expr::SubgroupReduce { value: index, .. } => ExprChildren::one(index),
        Expr::BinOp { left, right, .. } => ExprChildren::two(left, right),
        Expr::SubgroupShuffle { value, lane } => ExprChildren::two(value, lane),
        Expr::Select {
            cond,
            true_val,
            false_val,
        } => ExprChildren::three(cond, true_val, false_val),
        Expr::Fma { a, b, c } => ExprChildren::three(a, b, c),
        Expr::Atomic {
            index,
            expected,
            value,
            ..
        } => ExprChildren {
            direct: [Some(index), expected.as_deref(), Some(value)],
            args: &[],
        },
        Expr::Call { args, .. } => ExprChildren {
            direct: [None, None, None],
            args,
        },
    }
}

/// True when `expr` or any sub-expression satisfies `pred`.
///
/// Children come from [`expr_children`], so a new operand-carrying variant is
/// covered without touching this function. The walk is an explicit worklist,
/// short-circuiting on the first match, so an adversarially deep expression
/// cannot overflow the native stack.
#[must_use]
pub fn any_subexpr(expr: &Expr, pred: &mut impl FnMut(&Expr) -> bool) -> bool {
    let mut stack: SmallVec<[&Expr; 32]> = SmallVec::new();
    stack.push(expr);
    while let Some(current) = stack.pop() {
        if pred(current) {
            return true;
        }
        stack.extend(expr_children(current).iter().rev());
    }
    false
}

/// Visit `expr` and every sub-expression below it, in source pre-order.
///
/// This is the collector counterpart of [`any_subexpr`]: it visits every node
/// rather than stopping at the first match, so a collector cannot accidentally
/// be written on an early-exit search and lose the operands after the first
/// hit. Children come from [`expr_children`], so a new operand-carrying variant
/// is covered without touching this function, and the walk is an explicit
/// worklist so an adversarially deep expression cannot overflow the native
/// stack.
pub fn for_each_subexpr<'a>(expr: &'a Expr, visit: &mut impl FnMut(&'a Expr)) {
    let mut stack: SmallVec<[&'a Expr; 32]> = SmallVec::new();
    stack.push(expr);
    while let Some(current) = stack.pop() {
        visit(current);
        stack.extend(expr_children(current).iter().rev());
    }
}

/// True when `node` or any descendant satisfies `pred`.
///
/// Children come from [`child_bodies`], so a new nesting variant is covered
/// without touching this function.
///
/// The walk uses an explicit worklist, not the native stack: this is the
/// short-circuiting scan every barrier, fence, and effect detector in the
/// workspace calls, and a recursive version overflows on an adversarially deep
/// tree. The 64-slot inline `SmallVec` covers ordinary programs without
/// allocating.
#[must_use]
pub fn any_descendant(node: &Node, pred: &mut impl FnMut(&Node) -> bool) -> bool {
    let mut stack: SmallVec<[&Node; 64]> = SmallVec::new();
    stack.push(node);
    while let Some(current) = stack.pop() {
        if pred(current) {
            return true;
        }
        for body in child_bodies(current).into_iter().rev() {
            stack.extend(body.iter().rev());
        }
    }
    false
}

/// Call `f` on `node` and on every node nested under it.
///
/// The visiting counterpart of [`any_descendant`], which four call sites in this
/// crate obtained by handing `any_descendant` a predicate that always returned
/// `false` and discarding the result. That spelling works only for as long as
/// nobody makes the predicate answer `true`: the search stops on the first
/// match, so a visitor written that way turns into a visitor of a prefix, with
/// no diagnostic. The same defect once cost this crate a liveness bug.
pub(crate) fn for_each_descendant(node: &Node, f: &mut impl FnMut(&Node)) {
    // The predicate never reports a match, so the scan runs to exhaustion.
    let _: bool = any_descendant(node, &mut |current| {
        f(current);
        false
    });
}

/// True when any expression anywhere under `nodes` satisfies `pred`.
///
/// This is the scan a pass runs to decide whether it has anything to do. Node
/// nesting comes from [`child_bodies`], operand positions from
/// [`node_operands`], and sub-expressions from [`expr_children`], so a new
/// variant in either namespace reaches the scan without an edit at the call
/// site. A pass that re-derives the three enumerations instead answers "no
/// candidate" for a position it forgot, and a skipped pass leaves no trace.
///
/// The scan short-circuits on the first match.
#[must_use]
pub(crate) fn any_expr_in(nodes: &[Node], pred: &mut impl FnMut(&Expr) -> bool) -> bool {
    let mut stack: SmallVec<[&Node; 64]> = SmallVec::new();
    stack.extend(nodes.iter().rev());
    while let Some(current) = stack.pop() {
        for operand in node_operands(current).into_iter().flatten() {
            if any_subexpr(operand, pred) {
                return true;
            }
        }
        for body in child_bodies(current).into_iter().rev() {
            stack.extend(body.iter().rev());
        }
    }
    false
}

/// True when `pred` holds for `nodes` itself or for any body nested under it.
///
/// A pass whose candidate is a RELATION between siblings, rather than a
/// property of one node, cannot use [`any_descendant`]: two adjacent loops are
/// invisible from either loop alone. Such a pass needs the enclosing body, and
/// the bodies come from [`child_bodies`] so a new nesting variant reaches the
/// scan. `loop_fusion` re-derived the nesting list here and ended it in
/// `_ => return false`, which would have reported "no fusable pair" for a
/// variant that later gains a body.
///
/// The scan short-circuits on the first match.
#[must_use]
pub(crate) fn any_body(nodes: &[Node], pred: &mut impl FnMut(&[Node]) -> bool) -> bool {
    if pred(nodes) {
        return true;
    }
    let mut stack: SmallVec<[&Node; 64]> = SmallVec::new();
    stack.extend(nodes.iter().rev());
    while let Some(current) = stack.pop() {
        for body in child_bodies(current) {
            if !body.is_empty() && pred(body) {
                return true;
            }
        }
        for body in child_bodies(current).into_iter().rev() {
            stack.extend(body.iter().rev());
        }
    }
    false
}

/// Walk all nodes in a program, calling `f` on each.
///
/// The traversal is depth-first and visits every statement node in the
/// program's entry block, including nested `If`, `Loop`, and `Block`
/// bodies. Because the walk is iterative, it can handle arbitrarily deep
/// nesting without growing the native call stack.
///
/// Child bodies come from [`child_bodies`], the single exhaustive owner, so
/// this function does not restate which variants nest.
///
/// # Examples
///
/// ```
/// use vyre::ir::Program;
/// use vyre_foundation::transform::visit::walk_nodes;
///
/// let program = Program::empty();
/// walk_nodes(&program, |_node| {
///     // process node
/// });
/// ```
#[inline]
pub fn walk_nodes(program: &Program, f: impl FnMut(&Node)) {
    for_each_node(program.entry(), f);
}

/// Every node in `nodes` and in every nested body, depth first, in source
/// order.
///
/// This is the slice-taking form of [`walk_nodes`], for a caller holding a body
/// rather than a whole program: an analysis that answers a question about one
/// `Loop`'s body, or a rule that scans a branch. Both descend through
/// [`child_bodies`], the single exhaustive owner, so neither restates which
/// variants nest and a new nesting variant cannot be walked by one and skipped
/// by the other.
///
/// The walk is iterative, so nesting depth costs heap rather than native stack.
#[inline]
pub fn for_each_node<'a>(nodes: &'a [Node], mut f: impl FnMut(&'a Node)) {
    let _: ControlFlow<()> = try_for_each_node(nodes, |node| {
        f(node);
        ControlFlow::Continue(())
    });
}

/// Every node in `nodes` and in every nested body, depth first, in source
/// order, stopping at the first `Break`.
///
/// The short-circuiting form of [`for_each_node`], and the descent every
/// fallible scan outside this crate should use. Before it existed, a scan that
/// wanted to stop early had to implement `NodeVisitor`, which is
/// abstract-by-default: answering a question about two variants meant writing a
/// no-op body, with its full signature, for the other fifteen. Four scanners in
/// this workspace restated that same block of stubs, and the one that refused to
/// hand-rolled its own recursive descent ending in `_ => {}` instead, which
/// classified every nesting variant it had not been told about as containing
/// nothing.
///
/// Descent is [`child_bodies`], the single exhaustive owner, so a new nesting
/// variant is a compile error there rather than a silently empty answer here.
/// A caller that also needs the expressions a node carries takes them from
/// [`node_operands`], and the buffers it names from [`node_buffer_refs`]; both
/// are exhaustive for the same reason.
///
/// The walk is iterative, so nesting depth costs heap rather than native stack.
#[inline]
pub fn try_for_each_node<'a, B>(
    nodes: &'a [Node],
    mut f: impl FnMut(&'a Node) -> ControlFlow<B>,
) -> ControlFlow<B> {
    let mut stack: SmallVec<[&'a Node; 128]> = SmallVec::new();
    stack.reserve(nodes.len());
    for node in nodes.iter().rev() {
        stack.push(node);
    }

    while let Some(node) = stack.pop() {
        f(node)?;
        // Groups in reverse, each reversed: `then` pops before `otherwise`,
        // and both in source order. Same visit order as the hand-written match
        // this replaces.
        for body in child_bodies(node).into_iter().rev() {
            for n in body.iter().rev() {
                stack.push(n);
            }
        }
    }
    ControlFlow::Continue(())
}

fn push_node_children_and_exprs<'a>(
    node: &'a Node,
    node_stack: &mut SmallVec<[&'a Node; 128]>,
    expr_stack: &mut SmallVec<[&'a Expr; 128]>,
) {
    // Child bodies come from the single exhaustive owner. The two stacks are
    // independent, so pushing bodies before expressions preserves the order
    // within each one.
    for body in child_bodies(node).into_iter().rev() {
        for n in body.iter().rev() {
            node_stack.push(n);
        }
    }

    // Operand positions come from the single exhaustive owner. Pushed in
    // reverse so `drain_expr_stack` pops them in source order.
    expr_stack.extend(node_operands(node).into_iter().rev().flatten());
}

/// Visit every expression on `expr_stack` and everything below it.
///
/// Children come from [`expr_children`]. The hand-written enumeration this
/// replaces classified `SubgroupBallot`, `SubgroupShuffle`, and
/// `SubgroupReduce` as leaves, so `walk_exprs` and `walk_nodes_and_exprs`
/// never saw a subgroup operand: a `Load` inside `subgroup_add(load(b, i))`
/// did not count as a buffer reference, and a `Call` inside a shuffle lane was
/// invisible to `collect_call_op_ids`, which is how the inliner decides which
/// operations a program depends on.
fn drain_expr_stack<'a>(
    expr_stack: &mut SmallVec<[&'a Expr; 128]>,
    mut visit: impl FnMut(&'a Expr),
) {
    while let Some(expr) = expr_stack.pop() {
        visit(expr);
        expr_stack.extend(expr_children(expr).iter().rev());
    }
}

/// Walk all expressions in a program, calling `f` on each.
///
/// The traversal visits every `Expr` nested inside every node, again using
/// an explicit stack. This is the primary way to inspect or transform the
/// value-producing parts of a program.
///
/// # Examples
///
/// ```
/// use vyre::ir::Program;
/// use vyre_foundation::transform::visit::walk_exprs;
///
/// let program = Program::empty();
/// walk_exprs(&program, |_expr| {
///     // process expression
/// });
/// ```
#[inline]
pub fn walk_exprs(program: &Program, mut f: impl FnMut(&Expr)) {
    let mut node_stack: SmallVec<[&Node; 128]> = SmallVec::new();
    node_stack.reserve(program.entry().len());
    for node in program.entry().iter().rev() {
        node_stack.push(node);
    }

    let mut expr_stack: SmallVec<[&Expr; 128]> = SmallVec::new();
    expr_stack.reserve(program.entry().len().saturating_mul(2));

    while let Some(node) = node_stack.pop() {
        push_node_children_and_exprs(node, &mut node_stack, &mut expr_stack);
        drain_expr_stack(&mut expr_stack, &mut f);
    }
}

/// Mutably walk all nodes, allowing in-place transformation.
///
/// This is the mutable counterpart to [`walk_nodes`]. Callers can rewrite
/// nodes in place, for example to specialize control flow or inject
/// instrumentation. The explicit-stack invariant is preserved.
///
/// # Examples
///
/// ```
/// use vyre::ir::Program;
/// use vyre_foundation::transform::visit::walk_nodes_mut;
///
/// let mut program = Program::empty();
/// walk_nodes_mut(&mut program, |_node| {
///     // modify node
/// });
/// ```
#[inline]
pub fn walk_nodes_mut(program: &mut Program, mut f: impl FnMut(&mut Node)) {
    let mut stack: SmallVec<[&mut Node; 128]> = SmallVec::new();
    stack.reserve(program.entry().len());
    for node in program.entry_mut().iter_mut().rev() {
        stack.push(node);
    }

    while let Some(node) = stack.pop() {
        f(&mut *node);
        match node {
            Node::If {
                then, otherwise, ..
            } => {
                for n in otherwise.iter_mut().rev() {
                    stack.push(n);
                }
                for n in then.iter_mut().rev() {
                    stack.push(n);
                }
            }
            Node::Loop { body, .. } => {
                for n in body.iter_mut().rev() {
                    stack.push(n);
                }
            }
            Node::Block(inner) => {
                for n in inner.iter_mut().rev() {
                    stack.push(n);
                }
            }
            Node::Region { body, .. } => {
                for n in std::sync::Arc::make_mut(body).iter_mut().rev() {
                    stack.push(n);
                }
            }
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
    }
}

/// Walk all nodes and expressions in a program in a single traversal.
///
/// For each node, `visitor.visit_node(node)` is called, then all
/// expressions owned by that node (and their sub-expressions) are
/// visited via `visitor.visit_expr(expr)`.  Child nodes are pushed
/// onto the same explicit stack so the walk is iterative and safe
/// for arbitrarily deep ASTs.
///
/// The relative order of node visits matches [`walk_nodes`] and the
/// relative order of expression visits matches [`walk_exprs`].
///
/// # Examples
///
/// ```
/// use vyre::ir::Program;
/// use vyre_foundation::transform::visit::{walk_nodes_and_exprs, NodeVisitor, ExprVisitor};
///
/// struct CountAll;
///
/// impl NodeVisitor for CountAll {
///     fn visit_node(&mut self, _node: &vyre::ir::Node) {}
/// }
///
/// impl ExprVisitor for CountAll {
///     fn visit_expr(&mut self, _expr: &vyre::ir::Expr) {}
/// }
///
/// let program = Program::empty();
/// walk_nodes_and_exprs(&program, &mut CountAll);
/// ```
#[inline]
pub fn walk_nodes_and_exprs<V: NodeVisitor + ExprVisitor>(program: &Program, visitor: &mut V) {
    let mut node_stack: SmallVec<[&Node; 128]> = SmallVec::new();
    node_stack.reserve(program.entry().len());
    for node in program.entry().iter().rev() {
        node_stack.push(node);
    }

    let mut expr_stack: SmallVec<[&Expr; 128]> = SmallVec::new();
    expr_stack.reserve(program.entry().len().saturating_mul(2));

    while let Some(node) = node_stack.pop() {
        visitor.visit_node(node);
        push_node_children_and_exprs(node, &mut node_stack, &mut expr_stack);
        drain_expr_stack(&mut expr_stack, |expr| visitor.visit_expr(expr));
    }
}

/// This is a convenience wrapper around the visitor that extracts the set
/// of buffer identifiers actually used by the program. It is used by
/// validation and lowering to check that every declared buffer is
/// referenced and that no undeclared buffer is accessed.
///
/// The implementation uses a single combined traversal ([`walk_nodes_and_exprs`])
/// instead of the previous two-pass approach.
///
/// # Examples
///
/// ```
/// use vyre::ir::Program;
/// use vyre_foundation::transform::visit::referenced_buffers;
///
/// let program = Program::empty();
/// let buffers = referenced_buffers(&program);
/// assert!(buffers.is_empty());
/// ```
#[must_use]
#[inline]
pub fn referenced_buffers(program: &Program) -> HashSet<Ident> {
    // ProgramFacts::buffer_refs already enumerates every buffer-touching
    // node and expression in the program (Store/IndirectDispatch/AsyncLoad/
    // AsyncStore plus Load/BufLen/Atomic via the same SoA walk). Reuse the
    // OnceLock-cached facts instead of re-walking the entire tree with a
    // dedicated NodeVisitor + ExprVisitor pair.
    let facts = crate::optimizer::program_soa::ProgramFacts::build_cached(program);
    let mut names = HashSet::with_capacity(program.buffers().len());
    for (_, name, _) in facts.buffer_refs() {
        names.insert(name.clone());
    }
    names
}

/// Collect operation IDs from every [`Expr::Call`] in traversal order.
///
/// This helper is used by the inliner and the conform gate to discover
/// which operations a program depends on. The returned vector preserves
/// the order of first appearance.
///
/// # Examples
///
/// ```
/// use vyre::ir::{Expr, Node, Program};
/// use vyre_foundation::transform::visit::collect_call_op_ids;
///
/// let program = Program::wrapped(
///     Vec::new(),
///     [1, 1, 1],
///     vec![Node::let_bind("x", Expr::call("primitive.math.add", vec![Expr::u32(1)]))],
/// );
/// assert_eq!(
///     collect_call_op_ids(&program)
///         .into_iter()
///         .map(|id| id.to_string())
///         .collect::<Vec<_>>(),
///     vec!["primitive.math.add".to_string()]
/// );
/// ```
#[must_use]
#[inline]
pub fn collect_call_op_ids(program: &Program) -> Vec<Arc<str>> {
    // Cached call_count is the exact number of Expr::Call sites in
    // the program. When it is zero, skip the entire expression walk.
    // When non-zero, pre-size the output to the exact count so we
    // never resize during the walk.
    let stats = program.stats();
    let call_count = stats.call_count as usize;
    if call_count == 0 {
        return Vec::new();
    }
    let mut op_ids = Vec::with_capacity(call_count);
    walk_exprs(program, |expr| {
        if let Expr::Call { op_id, .. } = expr {
            op_ids.push(op_id.shared_text());
        }
    });
    op_ids
}

/// Shared IR-shape generators for the traversal proptests.
///
/// The public visitor and the validator are exercised over the same corpus,
/// so the corpus has exactly one owner: this module. `crate::validate` drives
/// it through [`arb_program`].
///
/// The corpus is the union of what both callers need. In particular
/// [`arb_expr`] emits argument-less `Expr::Call` leaves: the validator needs
/// them to reach `validate_call`, and the traversal proptests are strictly
/// better covered for having them, since `Expr::Call` is a node the walk
/// descends into.
#[cfg(test)]
pub(crate) mod fixtures {
    use crate::ir::{AtomicOp, BinOp, BufferDecl, DataType, Expr, Node, Program, UnOp};
    use crate::MemoryOrdering;
    use proptest::prelude::*;

    pub(crate) fn arb_ident() -> BoxedStrategy<String> {
        prop::sample::select(&["x", "y", "idx", "i", "acc"][..])
            .prop_map(str::to_string)
            .boxed()
    }

    pub(crate) fn arb_buffer_name() -> BoxedStrategy<String> {
        prop::sample::select(&["out", "input", "rw", "counts", "scratch"][..])
            .prop_map(str::to_string)
            .boxed()
    }

    pub(crate) fn arb_call_op() -> BoxedStrategy<String> {
        prop::sample::select(
            &[
                "test.noop",
                "test.add.u32",
                "test.mul.f32",
                "test.unknown_op",
            ][..],
        )
        .prop_map(str::to_string)
        .boxed()
    }

    pub(crate) fn arb_expr() -> BoxedStrategy<Expr> {
        let leaf = prop_oneof![
            any::<u32>().prop_map(Expr::LitU32),
            any::<i32>().prop_map(Expr::LitI32),
            any::<bool>().prop_map(Expr::LitBool),
            arb_ident().prop_map(Expr::var),
            arb_buffer_name().prop_map(Expr::buf_len),
            arb_call_op().prop_map(|op| Expr::call(op, vec![])),
        ];

        leaf.prop_recursive(3, 48, 3, |inner| {
            prop_oneof![
                (arb_buffer_name(), inner.clone()).prop_map(|(buffer, index)| Expr::Load {
                    buffer: buffer.into(),
                    index: Box::new(index),
                }),
                (inner.clone(), inner.clone()).prop_map(|(left, right)| Expr::BinOp {
                    op: BinOp::Add,
                    left: Box::new(left),
                    right: Box::new(right),
                }),
                (inner.clone(), inner.clone()).prop_map(|(left, right)| Expr::BinOp {
                    op: BinOp::Sub,
                    left: Box::new(left),
                    right: Box::new(right),
                }),
                inner.clone().prop_map(|operand| Expr::UnOp {
                    op: UnOp::Negate,
                    operand: Box::new(operand),
                }),
                (inner.clone(), inner.clone(), inner.clone()).prop_map(
                    |(cond, true_val, false_val)| Expr::Select {
                        cond: Box::new(cond),
                        true_val: Box::new(true_val),
                        false_val: Box::new(false_val),
                    }
                ),
                inner.clone().prop_map(|value| Expr::Cast {
                    target: DataType::U32,
                    value: Box::new(value),
                }),
                (
                    arb_buffer_name(),
                    inner.clone(),
                    proptest::option::of(inner.clone()),
                    inner.clone(),
                )
                    .prop_map(|(buffer, index, expected, value)| Expr::Atomic {
                        op: AtomicOp::Add,
                        buffer: buffer.into(),
                        index: Box::new(index),
                        expected: expected.map(Box::new),
                        value: Box::new(value),
                        ordering: MemoryOrdering::SeqCst,
                    }),
            ]
        })
        .boxed()
    }

    pub(crate) fn arb_node() -> BoxedStrategy<Node> {
        arb_node_with_depth(3)
    }

    pub(crate) fn arb_node_with_depth(depth: u32) -> BoxedStrategy<Node> {
        let leaf = prop_oneof![
            (arb_ident(), arb_expr()).prop_map(|(name, value)| Node::Let {
                name: name.into(),
                value,
            }),
            (arb_ident(), arb_expr()).prop_map(|(name, value)| Node::Assign {
                name: name.into(),
                value,
            }),
            (arb_buffer_name(), arb_expr(), arb_expr()).prop_map(|(buffer, index, value)| {
                Node::Store {
                    buffer: buffer.into(),
                    index,
                    value,
                }
            }),
            Just(Node::Return),
            Just(Node::barrier()),
        ];

        if depth == 0 {
            return leaf.boxed();
        }

        leaf.prop_recursive(2, 32, 2, move |inner| {
            prop_oneof![
                (
                    arb_expr(),
                    prop::collection::vec(inner.clone(), 0..=3),
                    prop::collection::vec(inner.clone(), 0..=3),
                )
                    .prop_map(|(cond, then, otherwise)| Node::If {
                        cond,
                        then,
                        otherwise,
                    }),
                (
                    arb_ident(),
                    arb_expr(),
                    arb_expr(),
                    prop::collection::vec(inner.clone(), 0..=3),
                )
                    .prop_map(|(var, from, to, body)| Node::Loop {
                        var: var.into(),
                        from,
                        to,
                        body,
                    }),
                prop::collection::vec(inner, 0..=3).prop_map(Node::Block),
            ]
        })
        .boxed()
    }

    pub(crate) fn arb_program() -> BoxedStrategy<Program> {
        prop::collection::vec(arb_node(), 0..=8)
            .prop_map(|entry| {
                Program::wrapped(
                    vec![
                        BufferDecl::output("out", 0, DataType::U32)
                            .with_count(8)
                            .with_output_byte_range(0..16),
                        BufferDecl::read("input", 1, DataType::U32).with_count(8),
                        BufferDecl::read_write("rw", 2, DataType::U32).with_count(8),
                        BufferDecl::read("counts", 3, DataType::U32).with_count(8),
                        BufferDecl::workgroup("scratch", 4, DataType::U32),
                    ],
                    [1, 1, 1],
                    entry,
                )
            })
            .boxed()
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::arb_program;
    use super::*;
    use crate::ir::{AtomicOp, BufferDecl, DataType, Expr, Node, Program};
    use proptest::prelude::*;

    /// Legacy double-walk implementation for equivalence verification.
    /// Mirrors every buffer-touching site that ProgramFacts::buffer_refs
    /// records so the equivalence proptest stays sound even when arb_node
    /// is extended with Async / IndirectDispatch variants.
    fn referenced_buffers_legacy(program: &Program) -> HashSet<Ident> {
        let mut names = HashSet::new();
        walk_exprs(program, |expr| match expr {
            Expr::Load { buffer, .. }
            | Expr::BufLen { buffer }
            | Expr::BufferRef { buffer }
            | Expr::Atomic { buffer, .. } => {
                names.insert(buffer.clone());
            }
            _ => {}
        });
        walk_nodes(program, |node| match node {
            Node::Store { buffer, .. } => {
                names.insert(buffer.clone());
            }
            Node::IndirectDispatch { count_buffer, .. } => {
                names.insert(count_buffer.clone());
            }
            Node::AsyncLoad {
                source,
                destination,
                ..
            }
            | Node::AsyncStore {
                source,
                destination,
                ..
            } => {
                names.insert(source.clone());
                names.insert(destination.clone());
            }
            _ => {}
        });
        names
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 256,
            ..ProptestConfig::default()
        })]

        #[test]
        fn combined_walker_referenced_buffers_eq_legacy(program in arb_program()) {
            let combined = referenced_buffers(&program);
            let legacy = referenced_buffers_legacy(&program);
            prop_assert_eq!(combined, legacy);
        }
    }

    #[test]
    fn referenced_buffers_collects_from_store_and_load() {
        let program = Program::wrapped(
            vec![
                BufferDecl::read("input", 0, DataType::U32).with_count(8),
                BufferDecl::output("out", 1, DataType::U32).with_count(8),
            ],
            [1, 1, 1],
            vec![
                Node::let_bind("x", Expr::load("input", Expr::u32(0))),
                Node::store("out", Expr::u32(0), Expr::var("x")),
                Node::Return,
            ],
        );

        let buffers = referenced_buffers(&program);
        assert!(buffers.contains(&Ident::from("input")));
        assert!(buffers.contains(&Ident::from("out")));
        assert_eq!(buffers.len(), 2);
    }

    #[test]
    fn referenced_buffers_collects_from_atomic_and_indirect_dispatch() {
        let program = Program::wrapped(
            vec![
                BufferDecl::read_write("rw", 0, DataType::U32).with_count(8),
                BufferDecl::read("counts", 1, DataType::U32).with_count(8),
            ],
            [1, 1, 1],
            vec![
                Node::let_bind(
                    "x",
                    Expr::Atomic {
                        op: AtomicOp::Add,
                        buffer: "rw".into(),
                        index: Box::new(Expr::u32(0)),
                        expected: None,
                        value: Box::new(Expr::u32(1)),
                        ordering: crate::MemoryOrdering::SeqCst,
                    },
                ),
                Node::IndirectDispatch {
                    count_buffer: "counts".into(),
                    count_offset: 0,
                },
                Node::Return,
            ],
        );

        let buffers = referenced_buffers(&program);
        assert!(buffers.contains(&Ident::from("rw")));
        assert!(buffers.contains(&Ident::from("counts")));
        assert_eq!(buffers.len(), 2);
    }
}
