//! Encoding of the declaration-chain links the typedef-visibility scan walks.
//!
//! Only the encode/decode rules and the VAST fields that carry them live here.
//! Row and context index math belongs to `decl_context_row_access`, which
//! callers name directly rather than through a wrapper of the same shape.

use super::super::super::decl_context_row_access::{
    decl_context_base, load_decl_context_field, load_vast_node_field,
};
use super::super::super::{
    SENTINEL, VAST_DECL_CONTEXT_PREV_DECL_CHAIN_LEN_FIELD, VAST_DECL_CONTEXT_PREV_DECL_LINK_FIELD,
    VAST_TYPEDEF_FLAGS_FIELD, VAST_TYPEDEF_SCOPE_FIELD, VAST_TYPEDEF_SYMBOL_FIELD,
};
use vyre_foundation::ir::Expr;

/// Byte length of the identifier lexeme, field 6 of a VAST row.
pub(super) fn vast_len_from_base(vast_nodes: &str, base_var: &str) -> Expr {
    load_vast_node_field(vast_nodes, Expr::var(base_var), 6)
}

pub(super) fn vast_scope_from_base(vast_nodes: &str, base_var: &str) -> Expr {
    load_vast_node_field(vast_nodes, Expr::var(base_var), VAST_TYPEDEF_SCOPE_FIELD)
}

pub(super) fn vast_typedef_hash_from_base(vast_nodes: &str, base_var: &str) -> Expr {
    load_vast_node_field(vast_nodes, Expr::var(base_var), VAST_TYPEDEF_SYMBOL_FIELD)
}

pub(super) fn vast_typedef_flags_from_base(vast_nodes: &str, base_var: &str) -> Expr {
    load_vast_node_field(vast_nodes, Expr::var(base_var), VAST_TYPEDEF_FLAGS_FIELD)
}

pub(super) fn prev_decl_link_from_base(decl_contexts: &str, base_var: &str) -> Expr {
    load_decl_context_field(
        decl_contexts,
        Expr::var(base_var),
        VAST_DECL_CONTEXT_PREV_DECL_LINK_FIELD,
    )
}

pub(super) fn prev_decl_chain_len_from_base(decl_contexts: &str, base_var: &str) -> Expr {
    load_decl_context_field(
        decl_contexts,
        Expr::var(base_var),
        VAST_DECL_CONTEXT_PREV_DECL_CHAIN_LEN_FIELD,
    )
}

pub(super) fn prev_decl_link_for_index(decl_contexts: &str, idx: Expr) -> Expr {
    load_decl_context_field(
        decl_contexts,
        decl_context_base(idx),
        VAST_DECL_CONTEXT_PREV_DECL_LINK_FIELD,
    )
}

pub(super) fn prev_decl_chain_len_for_index(decl_contexts: &str, idx: Expr) -> Expr {
    load_decl_context_field(
        decl_contexts,
        decl_context_base(idx),
        VAST_DECL_CONTEXT_PREV_DECL_CHAIN_LEN_FIELD,
    )
}

pub(super) fn decode_prev_decl_link(raw: Expr) -> Expr {
    Expr::select(
        Expr::or(
            Expr::eq(raw.clone(), Expr::u32(0)),
            Expr::eq(raw.clone(), Expr::u32(SENTINEL)),
        ),
        Expr::u32(SENTINEL),
        Expr::sub(raw, Expr::u32(1)),
    )
}

pub(super) fn decode_prepared_prev_decl_link(raw: Expr, prepared: Expr) -> Expr {
    Expr::select(
        Expr::and(prepared, Expr::ne(raw.clone(), Expr::u32(SENTINEL))),
        Expr::sub(raw, Expr::u32(1)),
        Expr::u32(SENTINEL),
    )
}
