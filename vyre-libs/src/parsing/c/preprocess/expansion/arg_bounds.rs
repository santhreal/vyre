//! Reads and writes of the per-argument token bound table.

use vyre_foundation::ir::{Expr, Node};

pub(super) fn selected_arg_bound(arg_bounds: &str, param: Expr) -> Expr {
    Expr::load(arg_bounds, param)
}

pub(super) fn assign_arg_bound(
    arg_bounds: &str,
    arg_index: Expr,
    value: Expr,
    num_tokens: Expr,
    overflow_trap: &'static str,
) -> Vec<Node> {
    vec![Node::if_then_else(
        Expr::lt(arg_index.clone(), num_tokens.clone()),
        vec![Node::store(arg_bounds, arg_index.clone(), value)],
        vec![Node::trap(arg_index, overflow_trap)],
    )]
}
