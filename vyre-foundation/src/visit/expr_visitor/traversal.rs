use super::ExprVisitor;
use crate::ir_inner::model::expr::Expr;
use crate::visit::expr_children;
use crate::visit::VisitOrder;
use smallvec::SmallVec;
use std::ops::ControlFlow;

/// Visit an expression tree in pre-order.
///
/// This is the historical default entry point for expression traversal.
pub fn visit_expr<V: ExprVisitor>(visitor: &mut V, expr: &Expr) -> ControlFlow<V::Break> {
    visit_preorder(visitor, expr)
}

/// Visit an expression tree in pre-order.
pub fn visit_preorder<V: ExprVisitor>(visitor: &mut V, expr: &Expr) -> ControlFlow<V::Break> {
    let mut stack = SmallVec::<[&Expr; 32]>::new();
    stack.push(expr);
    while let Some(current) = stack.pop() {
        dispatch_expr(visitor, current)?;
        push_expr_children_reverse(&mut stack, current);
    }
    ControlFlow::Continue(())
}

/// Visit an expression tree in post-order.
pub fn visit_postorder<V: ExprVisitor>(visitor: &mut V, expr: &Expr) -> ControlFlow<V::Break> {
    let mut stack = SmallVec::<[ExprVisitTask<'_>; 32]>::new();
    stack.push(ExprVisitTask::Visit(expr));
    while let Some(task) = stack.pop() {
        match task {
            ExprVisitTask::Visit(current) => {
                stack.push(ExprVisitTask::Dispatch(current));
                push_expr_child_tasks_reverse(&mut stack, current);
            }
            ExprVisitTask::Dispatch(current) => dispatch_expr(visitor, current)?,
        }
    }
    ControlFlow::Continue(())
}

/// Walk only the children of `expr`, leaving the current node to the caller.
///
/// Operand positions come from [`expr_children`], the one exhaustive owner, so
/// this function does not restate which variants carry operands.
pub fn walk_expr_children_default<V: ExprVisitor>(
    visitor: &mut V,
    expr: &Expr,
    order: VisitOrder,
) -> ControlFlow<V::Break> {
    for child in expr_children(expr).iter() {
        visit_with_order(visitor, child, order)?;
    }
    ControlFlow::Continue(())
}

fn visit_with_order<V: ExprVisitor>(
    visitor: &mut V,
    expr: &Expr,
    order: VisitOrder,
) -> ControlFlow<V::Break> {
    match order {
        VisitOrder::Preorder => visit_preorder(visitor, expr),
        VisitOrder::Postorder => visit_postorder(visitor, expr),
    }
}

fn push_expr_children_reverse<'a>(stack: &mut SmallVec<[&'a Expr; 32]>, expr: &'a Expr) {
    stack.extend(expr_children(expr).iter().rev());
}

fn push_expr_child_tasks_reverse<'a>(
    stack: &mut SmallVec<[ExprVisitTask<'a>; 32]>,
    expr: &'a Expr,
) {
    stack.extend(expr_children(expr).iter().rev().map(ExprVisitTask::Visit));
}

enum ExprVisitTask<'a> {
    Visit(&'a Expr),
    Dispatch(&'a Expr),
}

fn dispatch_expr<V: ExprVisitor>(visitor: &mut V, expr: &Expr) -> ControlFlow<V::Break> {
    match expr {
        Expr::LitU32(value) => visitor.visit_lit_u32(expr, *value),
        Expr::LitI32(value) => visitor.visit_lit_i32(expr, *value),
        Expr::LitF32(value) => visitor.visit_lit_f32(expr, *value),
        Expr::LitBool(value) => visitor.visit_lit_bool(expr, *value),
        Expr::Var(name) => visitor.visit_var(expr, name),
        Expr::Load { buffer, index } => visitor.visit_load(expr, buffer, index),
        Expr::BufLen { buffer } => visitor.visit_buf_len(expr, buffer),
        Expr::BufferRef { buffer } => visitor.visit_buffer_ref(expr, buffer),
        Expr::InvocationId { axis } => visitor.visit_invocation_id(expr, (*axis).into()),
        Expr::WorkgroupId { axis } => visitor.visit_workgroup_id(expr, (*axis).into()),
        Expr::LocalId { axis } => visitor.visit_local_id(expr, (*axis).into()),
        Expr::LogicalIndex { axis } => visitor.visit_logical_index(expr, (*axis).into()),
        Expr::LogicalTileId { axis } => visitor.visit_logical_tile_id(expr, (*axis).into()),
        Expr::LogicalWithinTileId { axis } => {
            visitor.visit_logical_within_tile_id(expr, (*axis).into())
        }
        Expr::BinOp { op, left, right } => visitor.visit_bin_op(expr, op, left, right),
        Expr::UnOp { op, operand } => visitor.visit_un_op(expr, op, operand),
        Expr::Call { op_id, args } => visitor.visit_call(expr, op_id, args),
        Expr::Fma { a, b, c } => visitor.visit_fma(expr, a, b, c),
        Expr::Select {
            cond,
            true_val,
            false_val,
        } => visitor.visit_select(expr, cond, true_val, false_val),
        Expr::Cast { target, value } => visitor.visit_cast(expr, target, value),
        Expr::Atomic {
            op,
            buffer,
            index,
            expected,
            value,
            ordering: _,
        } => visitor.visit_atomic(expr, op, buffer, index, expected.as_deref(), value),
        Expr::SubgroupBallot { cond } => visitor.visit_subgroup_ballot(expr, cond),
        Expr::SubgroupShuffle { value, lane } => visitor.visit_subgroup_shuffle(expr, value, lane),
        Expr::SubgroupReduce { value, .. } => visitor.visit_subgroup_add(expr, value),
        Expr::SubgroupLocalId => visitor.visit_subgroup_local_id(expr),
        Expr::SubgroupSize => visitor.visit_subgroup_size(expr),
        Expr::Opaque(extension) => visitor.visit_opaque_expr(expr, extension.as_ref()),
    }
}
