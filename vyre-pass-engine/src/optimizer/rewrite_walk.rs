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
//! exactly as `expr_arena::encode_expr` did. Scope truncation at the first
//! `Return` is IR structure too and lives beside the node walk.

use vyre_foundation::ir::{Expr, Node, Program};
use vyre_foundation::transform::rewrite_walk::{
    rewrite_program_entry, rewrite_scope, rewrite_scope_opt, NodeRewrite,
};

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
    rewrite_program_entry(program, |body| rewrite_scope(body, &mut walk))
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
