use super::*;

mod chain;
mod precomputed_declaration;
mod precomputed_visibility;
mod visibility_match;

pub(crate) use precomputed_declaration::emit_precomputed_declaration_kind_for_index;
pub(crate) use precomputed_visibility::emit_typedef_visibility_scan_precomputed_context;

pub(in crate::parsing::c::parse::vast) const VISIBLE_NAME_FOR_ROW_OP_ID: &str =
    "vyre-libs::parsing::c11_typedef_visible_name_for_row";
pub(in crate::parsing::c::parse::vast) const VISIBLE_NAME_FOR_ROW_PACKED_OP_ID: &str =
    "vyre-libs::parsing::c11_typedef_visible_name_for_row_packed_haystack";

pub(crate) fn emit_visible_typedef_name_for_index(
    parent_op_id: &str,
    vast_nodes: &str,
    haystack: &str,
    decl_contexts: Option<&str>,
    haystack_len: &Expr,
    idx: Expr,
    out_name: &str,
    prefix: &str,
    packed_haystack: bool,
) -> Vec<Node> {
    let target_base = format!("{prefix}_target_base");
    let target_link_raw = format!("{prefix}_target_link_raw");
    let target_prepared = format!("{prefix}_target_prepared");
    let target_chain_len = format!("{prefix}_target_chain_len");
    let scan_limit = format!("{prefix}_scan_limit");
    let target_scope = format!("{prefix}_target_scope");
    let last_decl_kind = format!("{prefix}_last_decl_kind");
    let chain_cursor = format!("{prefix}_chain_cursor");
    let chain_raw = format!("{prefix}_chain_raw");
    let scan_valid = format!("{prefix}_scan_valid");
    let scan_offset = format!("{prefix}_scan_offset");
    let scan = format!("{prefix}_scan");
    let scan_safe = format!("{prefix}_scan_safe");
    let scan_base = format!("{prefix}_scan_base");
    let scan_kind = format!("{prefix}_scan_kind");
    let scan_scope = format!("{prefix}_scan_scope");
    let scan_decl_kind = format!("{prefix}_scan_decl_result_kind");
    let scope_walk = format!("{prefix}_scope_walk");
    let scope_walk_depth = format!("{prefix}_scope_walk_depth");
    let same_name = format!("{prefix}_same_name");
    let visible_scope = format!("{prefix}_visible_scope");
    let visible_function = format!("{prefix}_visible_function");

    let mut nodes = vec![
        Node::let_bind(out_name, Expr::u32(0)),
        Node::let_bind(&target_base, vast_row_base_expr(idx.clone())),
        Node::let_bind(
            &target_link_raw,
            if let Some(decl_contexts) = decl_contexts {
                chain::prev_decl_link_for_index(decl_contexts, idx.clone())
            } else {
                chain::vast_typedef_flags_from_base(vast_nodes, &target_base)
            },
        ),
        Node::let_bind(
            &target_prepared,
            Expr::ne(Expr::var(&target_link_raw), Expr::u32(0)),
        ),
        Node::let_bind(
            &target_chain_len,
            if let Some(decl_contexts) = decl_contexts {
                chain::prev_decl_chain_len_for_index(decl_contexts, idx.clone())
            } else {
                idx.clone()
            },
        ),
        Node::let_bind(
            &scan_limit,
            Expr::select(
                Expr::var(&target_prepared),
                Expr::var(&target_chain_len),
                idx.clone(),
            ),
        ),
    ];
    let mut lookup_body = Vec::new();
    lookup_body.extend(emit_identifier_hash_for_row(
        parent_op_id,
        vast_nodes,
        haystack,
        haystack_len,
        Expr::var(&target_base),
        &format!("{prefix}_target"),
        packed_haystack,
    ));
    lookup_body.push(Node::let_bind(
        &target_scope,
        chain::vast_scope_from_base(vast_nodes, &target_base),
    ));
    lookup_body.push(Node::if_then(
        Expr::not(Expr::var(&target_prepared)),
        vec![
            Node::assign(&target_scope, Expr::u32(SENTINEL)),
            emit_scope_open_scan_phase(
                parent_op_id,
                vast_nodes,
                idx.clone(),
                &target_scope,
                &format!("{prefix}_scope"),
            ),
        ],
    ));
    nodes.push(Node::let_bind(&last_decl_kind, Expr::u32(0)));
    nodes.push(Node::let_bind(
        &chain_cursor,
        chain::decode_prepared_prev_decl_link(
            Expr::var(&target_link_raw),
            Expr::var(&target_prepared),
        ),
    ));
    nodes.push(Node::if_then(
        Expr::or(
            Expr::not(Expr::var(&target_prepared)),
            Expr::ne(Expr::var(&chain_cursor), Expr::u32(SENTINEL)),
        ),
        {
            lookup_body.push(Node::loop_for(
                &scan_offset,
                Expr::u32(0),
                Expr::var(&scan_limit),
                vec![
                    Node::let_bind(
                        &scan,
                        Expr::select(
                            Expr::var(&target_prepared),
                            Expr::var(&chain_cursor),
                            Expr::sub(
                                Expr::sub(idx.clone(), Expr::u32(1)),
                                Expr::var(&scan_offset),
                            ),
                        ),
                    ),
                    Node::let_bind(&scan_valid, Expr::ne(Expr::var(&scan), Expr::u32(SENTINEL))),
                    Node::let_bind(
                        &scan_safe,
                        Expr::select(Expr::var(&scan_valid), Expr::var(&scan), Expr::u32(0)),
                    ),
                    Node::let_bind(&scan_base, vast_row_base_expr(Expr::var(&scan_safe))),
                    Node::let_bind(
                        &scan_kind,
                        Expr::select(
                            Expr::var(&scan_valid),
                            Expr::load(vast_nodes, Expr::var(&scan_base)),
                            Expr::u32(SENTINEL),
                        ),
                    ),
                    Node::if_then(
                        Expr::and(
                            Expr::var(&scan_valid),
                            Expr::and(
                                Expr::eq(Expr::var(&last_decl_kind), Expr::u32(0)),
                                Expr::eq(Expr::var(&scan_kind), Expr::u32(TOK_IDENTIFIER)),
                            ),
                        ),
                        {
                            let scan_hash_prefix = format!("{prefix}_scan_hash");
                            let target_hash = format!("{prefix}_target_hash");
                            let target_len = format!("{prefix}_target_len");
                            let scan_len = format!("{prefix}_scan_len");
                            let scan_next_kind = format!("{prefix}_scan_next_kind");
                            let scan_possible_declarator =
                                format!("{prefix}_scan_possible_declarator");
                            let mut body = emit_identifier_hash_for_row(
                                parent_op_id,
                                vast_nodes,
                                haystack,
                                haystack_len,
                                Expr::var(&scan_base),
                                &scan_hash_prefix,
                                packed_haystack,
                            );
                            body.push(Node::let_bind(
                                &same_name,
                                Expr::and(
                                    Expr::eq(
                                        Expr::var(format!("{scan_hash_prefix}_hash")),
                                        Expr::var(&target_hash),
                                    ),
                                    Expr::eq(
                                        Expr::var(format!("{scan_hash_prefix}_len")),
                                        Expr::var(&target_len),
                                    ),
                                ),
                            ));
                            let mut same_name_body = Vec::new();
                            same_name_body.push(Node::let_bind(
                                &scan_scope,
                                chain::vast_scope_from_base(vast_nodes, &scan_base),
                            ));
                            same_name_body.push(Node::if_then(
                                Expr::not(Expr::var(&target_prepared)),
                                vec![
                                    Node::assign(&scan_scope, Expr::u32(SENTINEL)),
                                    emit_scope_open_scan_phase(
                                        parent_op_id,
                                        vast_nodes,
                                        Expr::var(&scan),
                                        &scan_scope,
                                        &format!("{prefix}_scan_scope"),
                                    ),
                                ],
                            ));
                            same_name_body.extend(emit_builtin_declaration_kind_for_index(
                                parent_op_id,
                                vast_nodes,
                                Expr::var(&scan),
                                &scan_decl_kind,
                                &format!("{prefix}_scan_decl"),
                                decl_contexts,
                            ));
                            same_name_body
                                .push(Node::let_bind(&visible_function, Expr::bool(true)));
                            same_name_body.push(visibility_match::emit_function_visibility_gate(
                                parent_op_id,
                                vast_nodes,
                                idx.clone(),
                                Expr::var(&scan),
                                &scan_decl_kind,
                                &visible_function,
                                &format!("{prefix}_target_function"),
                                &format!("{prefix}_scan_function"),
                                &format!("{prefix}_function"),
                                &format!("{prefix}_scan_function"),
                            ));
                            same_name_body.push(Node::let_bind(
                                &visible_scope,
                                Expr::eq(Expr::var(&scan_scope), Expr::u32(SENTINEL)),
                            ));
                            same_name_body.extend(visibility_match::emit_scope_visibility_update(
                                vast_nodes,
                                &target_scope,
                                &scan_scope,
                                &visible_scope,
                                &visible_function,
                                &scan_decl_kind,
                                &last_decl_kind,
                                &scope_walk,
                                &scope_walk_depth,
                            ));
                            body.push(Node::if_then(Expr::var(&same_name), same_name_body));
                            vec![
                                Node::let_bind(
                                    &scan_len,
                                    chain::vast_len_from_base(vast_nodes, &scan_base),
                                ),
                                Node::let_bind(
                                    &scan_next_kind,
                                    vast_next_row_kind_expr(
                                        vast_nodes,
                                        Expr::var(&scan),
                                        &Expr::var("annot_num_nodes"),
                                        Expr::u32(SENTINEL),
                                    ),
                                ),
                                Node::let_bind(
                                    &scan_possible_declarator,
                                    is_typedef_symbol_link_follower_token(Expr::var(
                                        &scan_next_kind,
                                    )),
                                ),
                                Node::if_then(
                                    Expr::and(
                                        Expr::var(&scan_possible_declarator),
                                        Expr::eq(Expr::var(&scan_len), Expr::var(&target_len)),
                                    ),
                                    body,
                                ),
                            ]
                        },
                    ),
                    Node::if_then(
                        Expr::and(Expr::var(&target_prepared), Expr::var(&scan_valid)),
                        vec![
                            Node::let_bind(
                                &chain_raw,
                                chain::vast_typedef_flags_from_base(vast_nodes, &scan_base),
                            ),
                            Node::assign(
                                &chain_cursor,
                                chain::decode_prev_decl_link(Expr::var(&chain_raw)),
                            ),
                        ],
                    ),
                ],
            ));
            lookup_body
        },
    ));
    nodes.push(Node::if_then(
        Expr::eq(Expr::var(&last_decl_kind), Expr::u32(1)),
        vec![Node::assign(out_name, Expr::u32(1))],
    ));
    nodes
}

fn visible_name_phase_program(op_id: &str, packed_haystack: bool) -> Program {
    const VISIBLE: &str = "phase_visible_typedef_name";

    let haystack_len = phase_haystack_len();
    phase_program(
        op_id,
        PhaseInputs::RowWithHaystack { packed_haystack },
        VISIBLE,
        emit_visible_typedef_name_for_index(
            op_id,
            phase_program::NODES,
            phase_program::HAYSTACK,
            None,
            &haystack_len,
            phase_row(),
            VISIBLE,
            "phase_visible_typedef",
            packed_haystack,
        ),
    )
}

pub(in crate::parsing::c::parse::vast) fn c11_typedef_visible_name_for_row() -> Program {
    visible_name_phase_program(VISIBLE_NAME_FOR_ROW_OP_ID, false)
}

pub(in crate::parsing::c::parse::vast) fn c11_typedef_visible_name_for_row_packed_haystack(
) -> Program {
    visible_name_phase_program(VISIBLE_NAME_FOR_ROW_PACKED_OP_ID, true)
}
