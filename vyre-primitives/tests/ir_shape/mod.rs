//! Structural queries over a built `Program`, shared by the suites that pin a
//! builder's emitted shape rather than its values.
//!
//! A shape assertion is only as trustworthy as the walker behind it: a walker
//! that forgets an `Expr` arm silently answers "no" and turns the assertion
//! green. Keeping one walker per question means a new IR node is fixed in one
//! place for every suite that asks.

/// Does the program contain any loop, at any nesting depth?
///
/// A builder that lowers to a serial loop where the contract promises a
/// parallel multi-block chain answers `true` here.
pub(crate) fn contains_loop(program: &vyre_foundation::ir::Program) -> bool {
    program.entry().iter().any(node_contains_loop)
}

fn node_contains_loop(node: &vyre_foundation::ir::Node) -> bool {
    use vyre_foundation::ir::Node;
    match node {
        Node::Loop { .. } => true,
        Node::Block(children) => children.iter().any(node_contains_loop),
        Node::If {
            then, otherwise, ..
        } => then.iter().any(node_contains_loop) || otherwise.iter().any(node_contains_loop),
        Node::Region { body, .. } => body.iter().any(node_contains_loop),
        _ => false,
    }
}

/// Does the program gate work behind `invocation_id.x == 0`?
///
/// That gate serializes a dispatch onto one lane, so a builder that claims to
/// expose parallel work must not contain one.
pub(crate) fn contains_invocation_zero_gate(program: &vyre_foundation::ir::Program) -> bool {
    program
        .entry()
        .iter()
        .any(node_contains_invocation_zero_gate)
}

fn node_contains_invocation_zero_gate(node: &vyre_foundation::ir::Node) -> bool {
    use vyre_foundation::ir::Node;
    match node {
        Node::If {
            cond,
            then,
            otherwise,
        } => {
            expr_is_invocation_zero(cond)
                || then.iter().any(node_contains_invocation_zero_gate)
                || otherwise.iter().any(node_contains_invocation_zero_gate)
        }
        Node::Block(children) => children.iter().any(node_contains_invocation_zero_gate),
        Node::Loop { body, .. } => body.iter().any(node_contains_invocation_zero_gate),
        Node::Region { body, .. } => body.iter().any(node_contains_invocation_zero_gate),
        _ => false,
    }
}

fn expr_is_invocation_zero(expr: &vyre_foundation::ir::Expr) -> bool {
    use vyre_foundation::ir::{BinOp, Expr};
    match expr {
        Expr::BinOp { op, left, right } if *op == BinOp::Eq => matches!(
            (&**left, &**right),
            (Expr::InvocationId { axis: 0 }, Expr::LitU32(0))
                | (Expr::LitU32(0), Expr::InvocationId { axis: 0 })
        ),
        Expr::BinOp { left, right, .. } => {
            expr_is_invocation_zero(left) || expr_is_invocation_zero(right)
        }
        Expr::UnOp { operand, .. } | Expr::Cast { value: operand, .. } => {
            expr_is_invocation_zero(operand)
        }
        Expr::Load { index, .. } => expr_is_invocation_zero(index),
        Expr::Select {
            cond,
            true_val,
            false_val,
        } => {
            expr_is_invocation_zero(cond)
                || expr_is_invocation_zero(true_val)
                || expr_is_invocation_zero(false_val)
        }
        Expr::Atomic {
            index,
            value,
            expected,
            ..
        } => {
            expr_is_invocation_zero(index)
                || expr_is_invocation_zero(value)
                || expected
                    .as_ref()
                    .is_some_and(|expr| expr_is_invocation_zero(expr))
        }
        Expr::Fma { a, b, c } => {
            expr_is_invocation_zero(a) || expr_is_invocation_zero(b) || expr_is_invocation_zero(c)
        }
        Expr::Call { args, .. } => args.iter().any(expr_is_invocation_zero),
        _ => false,
    }
}

/// Does the program read `invocation_id` anywhere?
///
/// A parallel builder must; a builder that lost its lane indexing reads none
/// and would otherwise still pass a value comparison on a one-element input.
pub(crate) fn contains_invocation_id(program: &vyre_foundation::ir::Program) -> bool {
    program.entry().iter().any(node_contains_invocation_id)
}

fn node_contains_invocation_id(node: &vyre_foundation::ir::Node) -> bool {
    use vyre_foundation::ir::Node;
    match node {
        Node::Let { value, .. } | Node::Assign { value, .. } => expr_contains_invocation_id(value),
        Node::Store { index, value, .. } => {
            expr_contains_invocation_id(index) || expr_contains_invocation_id(value)
        }
        Node::If {
            cond,
            then,
            otherwise,
        } => {
            expr_contains_invocation_id(cond)
                || then.iter().any(node_contains_invocation_id)
                || otherwise.iter().any(node_contains_invocation_id)
        }
        Node::Loop { from, to, body, .. } => {
            expr_contains_invocation_id(from)
                || expr_contains_invocation_id(to)
                || body.iter().any(node_contains_invocation_id)
        }
        Node::Block(children) => children.iter().any(node_contains_invocation_id),
        Node::Region { body, .. } => body.iter().any(node_contains_invocation_id),
        _ => false,
    }
}

fn expr_contains_invocation_id(expr: &vyre_foundation::ir::Expr) -> bool {
    use vyre_foundation::ir::Expr;
    match expr {
        Expr::InvocationId { .. } => true,
        Expr::Load { index, .. } | Expr::UnOp { operand: index, .. } => {
            expr_contains_invocation_id(index)
        }
        Expr::BinOp { left, right, .. } => {
            expr_contains_invocation_id(left) || expr_contains_invocation_id(right)
        }
        Expr::Call { args, .. } => args.iter().any(expr_contains_invocation_id),
        Expr::Select {
            cond,
            true_val,
            false_val,
        } => {
            expr_contains_invocation_id(cond)
                || expr_contains_invocation_id(true_val)
                || expr_contains_invocation_id(false_val)
        }
        Expr::Atomic {
            index,
            value,
            expected,
            ..
        } => {
            expr_contains_invocation_id(index)
                || expr_contains_invocation_id(value)
                || expected
                    .as_ref()
                    .is_some_and(|expr| expr_contains_invocation_id(expr))
        }
        Expr::Cast { value, .. } => expr_contains_invocation_id(value),
        Expr::Fma { a, b, c } => {
            expr_contains_invocation_id(a)
                || expr_contains_invocation_id(b)
                || expr_contains_invocation_id(c)
        }
        _ => false,
    }
}

/// How many grid-wide barriers does the program contain?
///
/// The multi-block scan chain is exactly Pass-A / Pass-B / Pass-C, so it needs
/// exactly two. A dropped barrier reads as a lost cross-block dependency.
pub(crate) fn grid_sync_barrier_count(program: &vyre_foundation::ir::Program) -> usize {
    program
        .entry()
        .iter()
        .map(node_grid_sync_barrier_count)
        .sum()
}

fn node_grid_sync_barrier_count(node: &vyre_foundation::ir::Node) -> usize {
    use vyre_foundation::ir::MemoryOrdering;
    use vyre_foundation::ir::Node;
    match node {
        Node::Barrier {
            ordering: MemoryOrdering::GridSync,
        } => 1,
        Node::Block(children) => children.iter().map(node_grid_sync_barrier_count).sum(),
        Node::If {
            then, otherwise, ..
        } => {
            then.iter().map(node_grid_sync_barrier_count).sum::<usize>()
                + otherwise
                    .iter()
                    .map(node_grid_sync_barrier_count)
                    .sum::<usize>()
        }
        Node::Loop { body, .. } => body.iter().map(node_grid_sync_barrier_count).sum(),
        Node::Region { body, .. } => body.iter().map(node_grid_sync_barrier_count).sum(),
        _ => 0,
    }
}
