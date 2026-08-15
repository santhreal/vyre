//! Traversals over nodes and expressions.
//!
//! Every walk here is an explicit worklist rather than recursion, so an
//! adversarially deep program costs heap instead of native stack. None of them
//! restates which variants nest or which positions they carry: node descent
//! comes from [`child_bodies`](super::node::child_bodies), operand positions
//! from [`node_operands`](super::node::node_operands), and sub-expressions from
//! [`expr_children`](super::expr::expr_children), so a new variant reaches
//! every walk in the workspace from one exhaustive match.

use super::expr::{any_subexpr, expr_children, for_each_subexpr};
use super::node::{child_bodies, child_bodies_mut, node_operands};
use crate::ir_inner::model::expr::Expr;
use crate::ir_inner::model::expr::Ident;
use crate::ir_inner::model::node::Node;
use crate::ir_inner::model::program::Program;
use smallvec::SmallVec;
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
    // The inner walk's own `ControlFlow` is a stop signal with no payload: the
    // caller's `B` cannot cross the `FnMut(&Node) -> ControlFlow<()>` boundary,
    // so it is parked in `stopped` instead. The two must agree, or a break was
    // reported without a value and this would answer `Continue` after stopping.
    let signal = try_for_each_node(nodes, |node| {
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
    debug_assert_eq!(
        signal.is_break(),
        stopped.is_some(),
        "the node walk stopped without a value to break with, or carried one without stopping"
    );
    stopped.map_or(ControlFlow::Continue(()), ControlFlow::Break)
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
pub fn any_expr_in(nodes: &[Node], pred: &mut impl FnMut(&Expr) -> bool) -> bool {
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
/// Descent is [`child_bodies_mut`], the single exhaustive owner of the body
/// slots in the unique-reference direction, so this function does not restate
/// which variants nest and cannot disagree with [`walk_nodes`] about which
/// subtrees a rewrite reaches.
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
        for body in child_bodies_mut(node).into_iter().rev() {
            stack.extend(IntoIterator::into_iter(body).rev());
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
