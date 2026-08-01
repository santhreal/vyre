use super::ExprVisitor;
use crate::ir_inner::model::expr::Expr;
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
pub fn walk_expr_children_default<V: ExprVisitor>(
    visitor: &mut V,
    expr: &Expr,
    order: VisitOrder,
) -> ControlFlow<V::Break> {
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
        | Expr::Opaque(_) => ControlFlow::Continue(()),
        Expr::Load { index, .. } | Expr::UnOp { operand: index, .. } => {
            visit_with_order(visitor, index, order)
        }
        Expr::BinOp { left, right, .. } => {
            visit_with_order(visitor, left, order)?;
            visit_with_order(visitor, right, order)
        }
        Expr::Call { args, .. } => {
            for arg in args {
                visit_with_order(visitor, arg, order)?;
            }
            ControlFlow::Continue(())
        }
        Expr::Select {
            cond,
            true_val,
            false_val,
        } => {
            visit_with_order(visitor, cond, order)?;
            visit_with_order(visitor, true_val, order)?;
            visit_with_order(visitor, false_val, order)
        }
        Expr::Cast { value, .. }
        | Expr::SubgroupBallot { cond: value }
        | Expr::SubgroupReduce { value, .. } => visit_with_order(visitor, value, order),
        Expr::Fma { a, b, c } => {
            visit_with_order(visitor, a, order)?;
            visit_with_order(visitor, b, order)?;
            visit_with_order(visitor, c, order)
        }
        Expr::Atomic {
            index,
            expected,
            value,
            ..
        } => {
            visit_with_order(visitor, index, order)?;
            if let Some(expected) = expected.as_deref() {
                visit_with_order(visitor, expected, order)?;
            }
            visit_with_order(visitor, value, order)
        }
        Expr::SubgroupShuffle { value, lane } => {
            visit_with_order(visitor, value, order)?;
            visit_with_order(visitor, lane, order)
        }
    }
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
        | Expr::Opaque(_) => {}
        Expr::Load { index, .. }
        | Expr::UnOp { operand: index, .. }
        | Expr::Cast { value: index, .. }
        | Expr::SubgroupBallot { cond: index }
        | Expr::SubgroupReduce { value: index, .. } => stack.push(index),
        Expr::BinOp { left, right, .. } => {
            stack.push(right);
            stack.push(left);
        }
        Expr::Call { args, .. } => {
            for arg in args.iter().rev() {
                stack.push(arg);
            }
        }
        Expr::Fma { a, b, c } => {
            stack.push(c);
            stack.push(b);
            stack.push(a);
        }
        Expr::Select {
            cond,
            true_val,
            false_val,
        } => {
            stack.push(false_val);
            stack.push(true_val);
            stack.push(cond);
        }
        Expr::Atomic {
            index,
            expected,
            value,
            ..
        } => {
            stack.push(value);
            if let Some(expected) = expected.as_deref() {
                stack.push(expected);
            }
            stack.push(index);
        }
        Expr::SubgroupShuffle { value, lane } => {
            stack.push(lane);
            stack.push(value);
        }
    }
}

fn push_expr_child_tasks_reverse<'a>(
    stack: &mut SmallVec<[ExprVisitTask<'a>; 32]>,
    expr: &'a Expr,
) {
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
        | Expr::Opaque(_) => {}
        Expr::Load { index, .. }
        | Expr::UnOp { operand: index, .. }
        | Expr::Cast { value: index, .. }
        | Expr::SubgroupBallot { cond: index }
        | Expr::SubgroupReduce { value: index, .. } => stack.push(ExprVisitTask::Visit(index)),
        Expr::BinOp { left, right, .. } => {
            stack.push(ExprVisitTask::Visit(right));
            stack.push(ExprVisitTask::Visit(left));
        }
        Expr::Call { args, .. } => {
            for arg in args.iter().rev() {
                stack.push(ExprVisitTask::Visit(arg));
            }
        }
        Expr::Fma { a, b, c } => {
            stack.push(ExprVisitTask::Visit(c));
            stack.push(ExprVisitTask::Visit(b));
            stack.push(ExprVisitTask::Visit(a));
        }
        Expr::Select {
            cond,
            true_val,
            false_val,
        } => {
            stack.push(ExprVisitTask::Visit(false_val));
            stack.push(ExprVisitTask::Visit(true_val));
            stack.push(ExprVisitTask::Visit(cond));
        }
        Expr::Atomic {
            index,
            expected,
            value,
            ..
        } => {
            stack.push(ExprVisitTask::Visit(value));
            if let Some(expected) = expected.as_deref() {
                stack.push(ExprVisitTask::Visit(expected));
            }
            stack.push(ExprVisitTask::Visit(index));
        }
        Expr::SubgroupShuffle { value, lane } => {
            stack.push(ExprVisitTask::Visit(lane));
            stack.push(ExprVisitTask::Visit(value));
        }
    }
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
