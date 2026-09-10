//! Structural queries over a built `Program`, shared by the suites that pin a
//! builder's emitted shape rather than its values.
//!
//! A shape assertion is only as trustworthy as the walker behind it: a walker
//! that forgets an `Expr` arm silently answers "no" and turns the assertion
//! green. Keeping one walker per question means a new IR node is fixed in one
//! place for every suite that asks.
//!
//! Every question is answered through [`shape_of`], one record per program.
//! A suite that asked one question directly left the other walkers with no
//! caller in that binary, so the shape is read whole and each suite states the
//! facts it pins.

use std::fmt;
use vyre_foundation::visit::{any_descendant, for_each_node};

/// The structural facts a suite pins about a built program.
pub(crate) struct ProgramShape {
    /// Contains a loop at any nesting depth.
    ///
    /// A builder that lowers to a serial loop where the contract promises a
    /// parallel multi-block chain reports `true`.
    pub(crate) loops: bool,
    /// Reads `invocation_id` anywhere.
    ///
    /// A parallel builder must; a builder that lost its lane indexing reads
    /// none and would otherwise still pass a value comparison on a one-element
    /// input.
    pub(crate) reads_invocation_id: bool,
    /// Gates work behind `invocation_id.x == 0`.
    ///
    /// That gate serializes a dispatch onto one lane, so a builder that claims
    /// to expose parallel work must not contain one.
    pub(crate) gates_on_invocation_zero: bool,
    /// Number of grid-wide barriers.
    ///
    /// The multi-block scan chain is exactly Pass-A / Pass-B / Pass-C, so it
    /// needs exactly two. A dropped barrier reads as a lost cross-block
    /// dependency.
    pub(crate) grid_sync_barriers: usize,
}

impl fmt::Display for ProgramShape {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "loops={} invocation_id={} invocation_zero_gate={} grid_sync_barriers={}",
            self.loops,
            self.reads_invocation_id,
            self.gates_on_invocation_zero,
            self.grid_sync_barriers
        )
    }
}

/// Read every structural fact of `program` in one walk per question.
pub(crate) fn shape_of(program: &vyre_foundation::ir::Program) -> ProgramShape {
    ProgramShape {
        loops: program.entry().iter().any(node_contains_loop),
        reads_invocation_id: program.entry().iter().any(node_contains_invocation_id),
        gates_on_invocation_zero: program
            .entry()
            .iter()
            .any(node_contains_invocation_zero_gate),
        grid_sync_barriers: program
            .entry()
            .iter()
            .map(node_grid_sync_barrier_count)
            .sum(),
    }
}

fn node_contains_loop(node: &vyre_foundation::ir::Node) -> bool {
    use vyre_foundation::ir::Node;
    any_descendant(node, &mut |current| {
        matches!(current, Node::Loop { .. })
    })
}

fn node_contains_invocation_zero_gate(node: &vyre_foundation::ir::Node) -> bool {
    use vyre_foundation::ir::Node;
    any_descendant(node, &mut |current| {
        matches!(current, Node::If { cond, .. } if expr_is_invocation_zero(cond))
    })
}

fn expr_is_invocation_zero(expr: &vyre_foundation::ir::Expr) -> bool {
    use vyre_foundation::ir::{BinOp, Expr};
    match expr {
        Expr::BinOp { op, left, right } if *op == BinOp::Eq => matches!(
            (&**left, &**right),
            (Expr::LogicalIndex { axis: 0 }, Expr::LitU32(0))
                | (Expr::LitU32(0), Expr::LogicalIndex { axis: 0 })
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

fn node_contains_invocation_id(node: &vyre_foundation::ir::Node) -> bool {
    use vyre_foundation::ir::Node;
    any_descendant(node, &mut |current| match current {
        Node::Let { value, .. } | Node::Assign { value, .. } => expr_contains_invocation_id(value),
        Node::Store { index, value, .. } => {
            expr_contains_invocation_id(index) || expr_contains_invocation_id(value)
        }
        Node::If { cond, .. } => expr_contains_invocation_id(cond),
        Node::Loop { from, to, .. } => {
            expr_contains_invocation_id(from) || expr_contains_invocation_id(to)
        }
        _ => false,
    })
}

fn expr_contains_invocation_id(expr: &vyre_foundation::ir::Expr) -> bool {
    use vyre_foundation::ir::Expr;
    match expr {
        Expr::LogicalIndex { .. } => true,
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

fn node_grid_sync_barrier_count(node: &vyre_foundation::ir::Node) -> usize {
    use vyre_foundation::ir::MemoryOrdering;
    use vyre_foundation::ir::Node;
    let mut count = 0;
    for_each_node(std::slice::from_ref(node), |current| {
        if matches!(
            current,
            Node::LogicalBarrier {
                ordering: MemoryOrdering::GridSync
            }
        ) {
            count += 1;
        }
    });
    count
}
