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

pub(crate) fn vast_prior_row_kind_expr(vast_nodes: &str, idx: Expr, offset: u32) -> Expr {
    Expr::select(
        Expr::ge(idx.clone(), Expr::u32(offset)),
        vast_row_kind_expr(vast_nodes, Expr::sub(idx, Expr::u32(offset))),
        Expr::u32(SENTINEL),
    )
}

pub(crate) fn vast_bounded_row_kind_expr(vast_nodes: &str, idx: Expr, fallback: Expr) -> Expr {
    Expr::select(
        Expr::lt(idx.clone(), Expr::var("annot_num_nodes")),
        vast_row_kind_expr(vast_nodes, idx),
        fallback,
    )
}
