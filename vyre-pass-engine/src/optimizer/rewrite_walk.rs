//! Encoder-order IR rewriting for the passes driven by a GPU analysis kernel.
//!
//! Every pass that consumes a per-Expr verdict from a GPU analysis kernel has
//! to revisit the IR in exactly the order `expr_arena` encoded it, because the
//! verdict is indexed by the encoder's post-order Expr id. That walk is
//! identical for const-fold, canonicalize, pattern-match, and the fused
//! resident decode; only the decision taken at each id differs. This module
//! supplies the encoder-specific part of that walk and each pass supplies the
//! decision.
//!
//! Which positions of a `Node` a rewrite must visit is not decided here. That
//! is IR structure, owned by [`vyre_foundation::transform::rewrite_walk`], and
//! this module is a [`NodeRewrite`] policy over it. Two things are genuinely
//! the encoder's and stay here: the post-order Expr counter, which must advance
//! exactly as `expr_arena::encode_expr` did, and the scope truncation at the
//! first `Return`, which is what the encoder emitted a node for.

use vyre_foundation::ir::{Expr, Node, Program};
use vyre_foundation::transform::rewrite_walk::{self, NodeRewrite};

/// The nodes of one scope the encoder emitted ids for.
///
/// Nodes after the first `Return` were never encoded, so no verdict is indexed
/// for them. Every walk over a scope truncates here, and until this was one
/// function four of them repeated the truncation beside their own loop.
pub(super) fn reachable_prefix(body: &[Node]) -> &[Node] {
    &body[..super::encode::reachable_prefix_len(body)]
}

/// Drive `rewrite` over one scope for its effects, discarding the rebuilt nodes.
///
/// A counting pass reads its answer out of the policy, not out of the tree, so
/// it must still visit exactly the positions the rewriting pass will visit.
pub(super) fn visit_scope<R: NodeRewrite>(body: &[Node], rewrite: &mut R) {
    for node in reachable_prefix(body) {
        rewrite_walk::rewrite_node(node, rewrite);
    }
}

/// Append one rewritten scope onto `out`.
///
/// A node the policy reports unchanged is cloned rather than rebuilt, which is
/// what keeps an untouched scope from being deep-copied.
pub(super) fn extend_with_rewritten_scope<R: NodeRewrite>(
    body: &[Node],
    rewrite: &mut R,
    out: &mut Vec<Node>,
) {
    let reachable = reachable_prefix(body);
    out.reserve(reachable.len());
    for node in reachable {
        out.push(rewrite_walk::rewrite_node(node, rewrite).unwrap_or_else(|| node.clone()));
    }
}

/// Rewrite one scope, reporting `None` when nothing in it changed.
///
/// The shared node walk is borrow-preserving: a node whose positions all report
/// no change returns `None` rather than a rebuilt clone. A scope walk that
/// discards that answer and rebuilds anyway deep-copies the whole subtree on
/// every pass, including the passes that rewrote nothing. Truncation counts as
/// a change, because the nodes past the first `Return` were never encoded and
/// must not survive.
pub(super) fn rewrite_scope_opt<R: NodeRewrite>(
    body: &[Node],
    rewrite: &mut R,
) -> Option<Vec<Node>> {
    let reachable = reachable_prefix(body);
    let mut out: Option<Vec<Node>> = None;
    for (index, node) in reachable.iter().enumerate() {
        match rewrite_walk::rewrite_node(node, rewrite) {
            None => {
                if let Some(out) = out.as_mut() {
                    out.push(node.clone());
                }
            }
            Some(rewritten) => {
                out.get_or_insert_with(|| {
                    let mut sink = Vec::with_capacity(reachable.len());
                    sink.extend_from_slice(&reachable[..index]);
                    sink
                })
                .push(rewritten);
            }
        }
    }
    if out.is_none() && reachable.len() != body.len() {
        return Some(reachable.to_vec());
    }
    out
}

/// Rewrite one scope into a fresh body.
pub(super) fn rewrite_scope<R: NodeRewrite>(body: &[Node], rewrite: &mut R) -> Vec<Node> {
    rewrite_scope_opt(body, rewrite).unwrap_or_else(|| reachable_prefix(body).to_vec())
}

/// Rewrite every Expr in `program` in encoder order.
///
/// `rewrite_expr` receives each Expr and the running post-order counter, and is
/// responsible for advancing that counter exactly as the encoder did. Use
/// [`rewrite_simple_expr_postorder`] inside it to get that for free.
pub(super) fn rewrite_program_with_expr_rewriter<F>(program: &Program, rewrite_expr: F) -> Program
where
    F: FnMut(&Expr, &mut u32) -> Expr,
{
    let mut walk = EncodedOrder {
        rewrite_expr,
        counter: 0,
    };
    super::rewrite_program_entry(program, |body| rewrite_scope(body, &mut walk))
}

/// Rebuild `expr` bottom-up, then hand the rebuilt Expr and its arena id to
/// `transform`.
///
/// The counter advances once per encoded Expr, after its children, which is the
/// encoder's own post-order numbering. Expr variants the encoder rejects return
/// unchanged without consuming an id.
pub(super) fn rewrite_simple_expr_postorder<F>(
    expr: &Expr,
    counter: &mut u32,
    transform: &mut F,
) -> Expr
where
    F: FnMut(Expr, u32) -> Expr,
{
    let rebuilt = match expr {
        Expr::LitU32(_)
        | Expr::LitI32(_)
        | Expr::LitF32(_)
        | Expr::LitBool(_)
        | Expr::Var(_)
        | Expr::BufLen { .. }
        | Expr::InvocationId { .. }
        | Expr::WorkgroupId { .. }
        | Expr::LocalId { .. }
        | Expr::SubgroupLocalId
        | Expr::SubgroupSize => expr.clone(),
        Expr::Load { buffer, index } => Expr::Load {
            buffer: buffer.clone(),
            index: Box::new(rewrite_simple_expr_postorder(index, counter, transform)),
        },
        Expr::BinOp { op, left, right } => Expr::BinOp {
            op: *op,
            left: Box::new(rewrite_simple_expr_postorder(left, counter, transform)),
            right: Box::new(rewrite_simple_expr_postorder(right, counter, transform)),
        },
        Expr::UnOp { op, operand } => Expr::UnOp {
            op: op.clone(),
            operand: Box::new(rewrite_simple_expr_postorder(operand, counter, transform)),
        },
        Expr::Select {
            cond,
            true_val,
            false_val,
        } => Expr::Select {
            cond: Box::new(rewrite_simple_expr_postorder(cond, counter, transform)),
            true_val: Box::new(rewrite_simple_expr_postorder(true_val, counter, transform)),
            false_val: Box::new(rewrite_simple_expr_postorder(false_val, counter, transform)),
        },
        Expr::Fma { a, b, c } => Expr::Fma {
            a: Box::new(rewrite_simple_expr_postorder(a, counter, transform)),
            b: Box::new(rewrite_simple_expr_postorder(b, counter, transform)),
            c: Box::new(rewrite_simple_expr_postorder(c, counter, transform)),
        },
        _ => return expr.clone(),
    };
    let id = *counter;
    *counter += 1;
    transform(rebuilt, id)
}

/// Drives the shared node walk in the encoder's order.
struct EncodedOrder<F> {
    rewrite_expr: F,
    counter: u32,
}

impl<F> NodeRewrite for EncodedOrder<F>
where
    F: FnMut(&Expr, &mut u32) -> Expr,
{
    /// The shared walk offers operands in source order before child bodies,
    /// which is the order `expr_arena::encode_node` allocated ids in, so the
    /// counter stays aligned with the verdict buffer.
    fn operand(&mut self, expr: &Expr) -> Option<Expr> {
        let rewritten = (self.rewrite_expr)(expr, &mut self.counter);
        (rewritten != *expr).then_some(rewritten)
    }

    fn body(&mut self, _parent: &Node, body: &[Node]) -> Option<Vec<Node>> {
        rewrite_scope_opt(body, self)
    }
}
