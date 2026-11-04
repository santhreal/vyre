//! Loop matching and bound analysis helpers for loop passes.

use crate::ir::{Expr, Ident, Node};

/// Borrowed components of a `Node::Loop`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LoopRef<'a> {
    pub var: &'a Ident,
    pub from: &'a Expr,
    pub to: &'a Expr,
    pub body: &'a [Node],
}

/// Match a `Node::Loop`, borrowing its fields.
#[must_use]
pub(crate) fn match_loop(node: &Node) -> Option<LoopRef<'_>> {
    match node {
        Node::Loop {
            var,
            from,
            to,
            body,
        } => Some(LoopRef {
            var,
            from,
            to,
            body,
        }),
        _ => None,
    }
}

/// Extract constant bounds `(from, to)` if both are u32 or i32 literals.
#[must_use]
pub(crate) fn literal_bounds(from: &Expr, to: &Expr) -> Option<(u32, u32)> {
    let from = literal_u32(from)?;
    let to = literal_u32(to)?;
    Some((from, to))
}

/// Extract a `u32` value from a `LitU32` or non-negative `LitI32`.
#[must_use]
pub(crate) fn literal_u32(expr: &Expr) -> Option<u32> {
    match expr {
        Expr::LitU32(value) => Some(*value),
        Expr::LitI32(value) => u32::try_from(*value).ok(),
        _ => None,
    }
}
