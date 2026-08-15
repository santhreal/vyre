use super::*;
use crate::parsing::c::parse::vast::declaration_prefix_scan::{
    emit_declaration_prefix_back_scan, DeclarationPrefixScan,
};

pub(crate) fn emit_precomputed_declaration_kind_for_index(
    vast_nodes: &str,
    decl_contexts: &str,
    visible_type: &str,
    idx: Expr,
    out_name: &str,
    prefix: &str,
) -> Vec<Node> {
    let base = format!("{prefix}_base");
    let kind = format!("{prefix}_row_kind");
    let context_base = format!("{prefix}_context_base");
    let prefix_start = format!("{prefix}_prefix_start");
    let prefix_done = format!("{prefix}_prefix_done");
    let prefix_idx = format!("{prefix}_prefix_idx");
    let prefix_kind = format!("{prefix}_prefix_kind");
    let has_typedef = format!("{prefix}_has_typedef");
    let has_type = format!("{prefix}_has_type");
    let prev_kind = format!("{prefix}_prev_kind");
    let next_kind = format!("{prefix}_next_kind");
    let possible_declarator = format!("{prefix}_possible_declarator");

    let mut nodes = vec![
        Node::let_bind(out_name, Expr::u32(0)),
        Node::let_bind(&base, vast_row_base_expr(idx.clone())),
        Node::let_bind(&kind, Expr::load(vast_nodes, Expr::var(&base))),
        Node::let_bind(
            &context_base,
            Expr::mul(idx.clone(), Expr::u32(VAST_DECL_CONTEXT_STRIDE_U32)),
        ),
        Node::let_bind(
            &prefix_start,
            Expr::load(
                decl_contexts,
                Expr::add(
                    Expr::var(&context_base),
                    Expr::u32(VAST_DECL_CONTEXT_PREFIX_START_FIELD),
                ),
            ),
        ),
        Node::let_bind(&has_typedef, Expr::u32(0)),
        Node::let_bind(&has_type, Expr::u32(0)),
    ];
    nodes.extend(emit_declaration_prefix_back_scan(
        &DeclarationPrefixScan {
            vast_nodes,
            idx: idx.clone(),
            prefix_start: Expr::var(&prefix_start),
            prefix,
        },
        vec![
            Node::if_then(
                is_decl_prefix_reset_token(Expr::var(&prefix_kind)),
                vec![Node::assign(&prefix_done, Expr::u32(1))],
            ),
            Node::if_then(
                Expr::eq(Expr::var(&prefix_kind), Expr::u32(TOK_TYPEDEF)),
                vec![Node::assign(&has_typedef, Expr::u32(1))],
            ),
            Node::if_then(
                is_decl_prefix_token(Expr::var(&prefix_kind)),
                vec![Node::assign(&has_type, Expr::u32(1))],
            ),
            // A prefix identifier that resolves as a visible typedef-name (or a
            // GNU `typeof` keyword-hash) also counts as the declaration's TYPE.
            // The self-contained annotator resolves this inline per prefix row;
            // here it is the precomputed per-node bit from
            // `c11_precompute_vast_visible_type`.
            Node::if_then(
                Expr::eq(
                    Expr::load(visible_type, Expr::var(&prefix_idx)),
                    Expr::u32(1),
                ),
                vec![Node::assign(&has_type, Expr::u32(1))],
            ),
        ],
    ));
    nodes.extend([
        Node::let_bind(
            &prev_kind,
            vast_prior_row_kind_expr(vast_nodes, idx.clone(), 1),
        ),
        Node::let_bind(
            &next_kind,
            vast_next_row_kind_expr(
                vast_nodes,
                idx,
                &Expr::var("annot_num_nodes"),
                Expr::u32(SENTINEL),
            ),
        ),
        Node::let_bind(
            &possible_declarator,
            is_declarator_follower_token(Expr::var(&next_kind)),
        ),
        emit_declaration_kind_result_assignment(
            out_name,
            Expr::eq(Expr::var(&kind), Expr::u32(TOK_IDENTIFIER)),
            Expr::var(&possible_declarator),
            Expr::not(is_declaration_previous_disqualifier_token(Expr::var(
                &prev_kind,
            ))),
            Expr::ne(Expr::var(&next_kind), Expr::u32(TOK_COLON)),
            Expr::bool(true),
            Expr::eq(Expr::var(&has_typedef), Expr::u32(1)),
            Expr::eq(Expr::var(&has_type), Expr::u32(1)),
        ),
    ]);
    nodes
}
