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

pub(crate) fn emit_declaration_kind_result_assignment(
    out_name: &str,
    is_identifier: Expr,
    declarator_follower: Expr,
    previous_token_allows_declarator: Expr,
    next_token_allows_declarator: Expr,
    contextual_declarator_allowed: Expr,
    has_typedef: Expr,
    has_type: Expr,
) -> Node {
    Node::if_then(
        Expr::and(
            is_identifier,
            Expr::and(
                declarator_follower,
                Expr::and(
                    previous_token_allows_declarator,
                    Expr::and(
                        next_token_allows_declarator,
                        Expr::and(
                            contextual_declarator_allowed,
                            Expr::or(has_typedef.clone(), has_type),
                        ),
                    ),
                ),
            ),
        ),
        vec![Node::assign(
            out_name,
            Expr::select(has_typedef, Expr::u32(1), Expr::u32(2)),
        )],
    )
}

pub(crate) fn emit_identifier_source_hash_for_index(
    vast_nodes: &str,
    haystack: &str,
    haystack_len: &Expr,
    idx: Expr,
    out_name: &str,
    prefix: &str,
    packed_haystack: bool,
) -> Vec<Node> {
    let base = format!("{prefix}_hash_base");
    let start = format!("{prefix}_hash_start");
    let len = format!("{prefix}_hash_len");
    let cursor = format!("{prefix}_hash_i");
    let byte = format!("{prefix}_hash_byte");

    let row = IdentifierRowHash {
        vast_nodes,
        haystack,
        haystack_len,
        row_base: Expr::var(&base),
        packed_haystack,
        names: IdentifierRowHashNames {
            start: &start,
            len: &len,
            hash: out_name,
            cursor: &cursor,
            byte: &byte,
        },
    };

    let mut nodes = vec![Node::let_bind(
        &base,
        Expr::mul(idx, Expr::u32(VAST_NODE_STRIDE_U32)),
    )];
    let guard = row.hash_is_unset();
    nodes.extend(row.nodes(guard));
    nodes
}
