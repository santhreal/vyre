//! Index math for one VAST node row inside the flat `vast_nodes` buffer.
//!
//! Every row is `VAST_NODE_STRIDE_U32` words wide; field 0 is the node kind
//! and field 1 the parent index. Builders address rows through these
//! accessors so the stride and field order are stated once.

use super::*;

pub(crate) fn vast_row_base_expr(idx: Expr) -> Expr {
    Expr::mul(idx, Expr::u32(VAST_NODE_STRIDE_U32))
}

pub(crate) fn vast_row_field_from_base_expr(vast_nodes: &str, base: Expr, field: u32) -> Expr {
    let offset = if field == 0 {
        base
    } else {
        Expr::add(base, Expr::u32(field))
    };
    Expr::load(vast_nodes, offset)
}

pub(crate) fn vast_row_field_expr(vast_nodes: &str, idx: Expr, field: u32) -> Expr {
    vast_row_field_from_base_expr(vast_nodes, vast_row_base_expr(idx), field)
}

pub(crate) fn vast_row_kind_expr(vast_nodes: &str, idx: Expr) -> Expr {
    vast_row_kind_from_base_expr(vast_nodes, vast_row_base_expr(idx))
}

pub(crate) fn vast_row_kind_from_base_expr(vast_nodes: &str, base: Expr) -> Expr {
    vast_row_field_from_base_expr(vast_nodes, base, 0)
}

pub(crate) fn vast_row_parent_from_base_expr(vast_nodes: &str, base: Expr) -> Expr {
    vast_row_field_from_base_expr(vast_nodes, base, 1)
}

/// Kind of the row `offset` positions before `idx`, or `SENTINEL` when `idx`
/// is within `offset` of the start of the table.
///
/// The index is clamped before the load. `Expr::Select` is a value select, so
/// both arms are evaluated: an unclamped `idx - offset` wraps to nearly
/// `u32::MAX` at the start of the table and the untaken arm then addresses a
/// row far past the end.
pub(crate) fn vast_prior_row_kind_expr(vast_nodes: &str, idx: Expr, offset: u32) -> Expr {
    let in_range = Expr::ge(idx.clone(), Expr::u32(offset));
    let prior = Expr::select(
        in_range.clone(),
        Expr::sub(idx, Expr::u32(offset)),
        Expr::u32(0),
    );
    Expr::select(
        in_range,
        vast_row_kind_expr(vast_nodes, prior),
        Expr::u32(SENTINEL),
    )
}

/// Kind of the row after `idx`, or `fallback` when `idx` is the last row of a
/// `num_nodes`-row table.
///
/// The forward index is clamped to `idx` for the same reason the backward one
/// is clamped: the untaken arm is still evaluated, and `idx + 1` at the last
/// row addresses a row that does not exist.
pub(crate) fn vast_next_row_kind_expr(
    vast_nodes: &str,
    idx: Expr,
    num_nodes: &Expr,
    fallback: Expr,
) -> Expr {
    let in_range = Expr::lt(Expr::add(idx.clone(), Expr::u32(1)), num_nodes.clone());
    let next = Expr::select(in_range.clone(), Expr::add(idx.clone(), Expr::u32(1)), idx);
    Expr::select(in_range, vast_row_kind_expr(vast_nodes, next), fallback)
}

pub(crate) fn vast_bounded_row_kind_expr(vast_nodes: &str, idx: Expr, fallback: Expr) -> Expr {
    Expr::select(
        Expr::lt(idx.clone(), Expr::var("annot_num_nodes")),
        vast_row_kind_expr(vast_nodes, idx),
        fallback,
    )
}
