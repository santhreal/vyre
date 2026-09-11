//! Applying the action column back onto the expression tree.
//!
//! The walk visits Exprs in the same order the arena encoder numbered them, so
//! the counter and the action index stay in step. An action the decoder does
//! not know leaves the Expr alone.
//!
//! Every discriminant in `rewrite_action` is answered here. The match is
//! exhaustive over the shapes a rule can fire on, and the arms that reach into
//! a child rebuild the node when that child no longer has the shape the kernel
//! saw: a nested rule that already collapsed the child retracts the premise of
//! the outer rewrite, and the outer node has to survive intact rather than
//! yield an operand that is no longer there.

use vyre_foundation::ir::{Expr, Program};

use super::rewrite_action;
use crate::optimizer::rewrite_walk;

pub(super) fn rewrite_program_with_actions(program: Program, actions: &[u32]) -> Program {
    rewrite_walk::rewrite_program_with_expr_rewriter(&program, |expr, counter| {
        rewrite_expr(expr, actions, counter)
    })
}

fn rewrite_expr(expr: &Expr, actions: &[u32], counter: &mut u32) -> Expr {
    rewrite_walk::rewrite_simple_expr_postorder(expr, counter, &mut |rewritten, id| {
        let action = actions
            .get(id as usize)
            .copied()
            .unwrap_or(rewrite_action::NONE);
        apply_action(action, rewritten)
    })
}

/// The rewritten Expr `action` selects, or `rewritten` when the action does not
/// apply to the shape the post-order walk produced.
fn apply_action(action: u32, rewritten: Expr) -> Expr {
    match (action, rewritten) {
        (rewrite_action::REPLACE_WITH_LEFT, Expr::BinOp { left, .. }) => *left,
        (rewrite_action::REPLACE_WITH_RIGHT, Expr::BinOp { right, .. }) => *right,
        (rewrite_action::REPLACE_WITH_LIT_ZERO, Expr::BinOp { .. }) => Expr::LitU32(0),
        (rewrite_action::REPLACE_WITH_LIT_TRUE, Expr::BinOp { .. }) => Expr::LitBool(true),
        (rewrite_action::REPLACE_WITH_LIT_FALSE, Expr::BinOp { .. }) => Expr::LitBool(false),
        (rewrite_action::REPLACE_WITH_GRAND_OPERAND, Expr::UnOp { op, operand }) => {
            match *operand {
                Expr::UnOp { operand: grand, .. } => *grand,
                collapsed => Expr::UnOp {
                    op,
                    operand: Box::new(collapsed),
                },
            }
        }
        (rewrite_action::REPLACE_WITH_LEFT_INNER_LEFT, Expr::BinOp { op, left, right }) => {
            match *left {
                Expr::BinOp {
                    left: inner_left, ..
                } => *inner_left,
                collapsed => Expr::BinOp {
                    op,
                    left: Box::new(collapsed),
                    right,
                },
            }
        }
        (rewrite_action::REPLACE_WITH_LEFT_INNER_RIGHT, Expr::BinOp { op, left, right }) => {
            match *left {
                Expr::BinOp {
                    right: inner_right, ..
                } => *inner_right,
                collapsed => Expr::BinOp {
                    op,
                    left: Box::new(collapsed),
                    right,
                },
            }
        }
        (_, other) => other,
    }
}
