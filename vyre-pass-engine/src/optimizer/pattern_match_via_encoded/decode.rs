//! Applying the action column back onto the expression tree.
//!
//! The walk visits Exprs in the same order the arena encoder numbered them, so
//! the counter and the action index stay in step. An action the decoder does
//! not know leaves the Expr alone.

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
        match (action, rewritten) {
            (rewrite_action::REPLACE_WITH_LEFT, Expr::BinOp { left, .. }) => *left,
            (rewrite_action::REPLACE_WITH_RIGHT, Expr::BinOp { right, .. }) => *right,
            (rewrite_action::REPLACE_WITH_LIT_ZERO, Expr::BinOp { .. }) => Expr::LitU32(0),
            (_, other) => other,
        }
    })
}
