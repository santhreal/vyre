//! The one encoded-order IR rewrite walk.
//!
//! Every pass that consumes a per-Expr verdict from a GPU analysis kernel has
//! to revisit the IR in exactly the order `expr_arena` encoded it, because the
//! verdict is indexed by the encoder's post-order Expr id. That walk is
//! identical for const-fold, canonicalize, pattern-match, and the fused
//! resident decode; only the decision taken at each id differs. The walk is
//! here and each pass supplies the decision.

use std::sync::Arc;

use vyre_foundation::ir::{Expr, Node, Program};

/// Rewrite every Expr in `program` in encoder order.
///
/// `rewrite_expr` receives each Expr and the running post-order counter, and is
/// responsible for advancing that counter exactly as the encoder did. Use
/// [`rewrite_simple_expr_postorder`] inside it to get that for free.
pub(super) fn rewrite_program_with_expr_rewriter<F>(
    program: &Program,
    mut rewrite_expr: F,
) -> Program
where
    F: FnMut(&Expr, &mut u32) -> Expr,
{
    let mut counter = 0u32;
    super::rewrite_program_entry(program, |body| {
        rewrite_scope(body, &mut rewrite_expr, &mut counter)
    })
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

fn rewrite_scope<F>(body: &[Node], rewrite_expr: &mut F, counter: &mut u32) -> Vec<Node>
where
    F: FnMut(&Expr, &mut u32) -> Expr,
{
    let prefix_len = super::encode::reachable_prefix_len(body);
    let mut out = Vec::with_capacity(prefix_len);
    for node in &body[..prefix_len] {
        out.push(rewrite_node(node, rewrite_expr, counter));
    }
    out
}

fn rewrite_node<F>(node: &Node, rewrite_expr: &mut F, counter: &mut u32) -> Node
where
    F: FnMut(&Expr, &mut u32) -> Expr,
{
    match node {
        Node::Let { name, value } => Node::let_bind(name.clone(), rewrite_expr(value, counter)),
        Node::Assign { name, value } => Node::assign(name.clone(), rewrite_expr(value, counter)),
        Node::Store {
            buffer,
            index,
            value,
        } => Node::store(
            buffer.clone(),
            rewrite_expr(index, counter),
            rewrite_expr(value, counter),
        ),
        Node::If {
            cond,
            then,
            otherwise,
        } => Node::if_then_else(
            rewrite_expr(cond, counter),
            rewrite_scope(then, rewrite_expr, counter),
            rewrite_scope(otherwise, rewrite_expr, counter),
        ),
        Node::Loop {
            var,
            from,
            to,
            body,
        } => Node::loop_for(
            var.clone(),
            rewrite_expr(from, counter),
            rewrite_expr(to, counter),
            rewrite_scope(body, rewrite_expr, counter),
        ),
        Node::AsyncLoad {
            source,
            destination,
            offset,
            size,
            tag,
        } => Node::AsyncLoad {
            source: source.clone(),
            destination: destination.clone(),
            offset: Box::new(rewrite_expr(offset, counter)),
            size: Box::new(rewrite_expr(size, counter)),
            tag: tag.clone(),
        },
        Node::AsyncStore {
            source,
            destination,
            offset,
            size,
            tag,
        } => Node::AsyncStore {
            source: source.clone(),
            destination: destination.clone(),
            offset: Box::new(rewrite_expr(offset, counter)),
            size: Box::new(rewrite_expr(size, counter)),
            tag: tag.clone(),
        },
        Node::Trap { address, tag } => Node::Trap {
            address: Box::new(rewrite_expr(address, counter)),
            tag: tag.clone(),
        },
        Node::Block(body) => Node::Block(rewrite_scope(body, rewrite_expr, counter)),
        Node::Region {
            generator,
            source_region,
            body,
        } => Node::Region {
            generator: generator.clone(),
            source_region: source_region.clone(),
            body: Arc::new(rewrite_scope(body.as_slice(), rewrite_expr, counter)),
        },
        Node::Return
        | Node::Barrier { .. }
        | Node::IndirectDispatch { .. }
        | Node::AsyncWait { .. }
        | Node::Resume { .. }
        | Node::Opaque(_) => node.clone(),
        _ => node.clone(),
    }
}
