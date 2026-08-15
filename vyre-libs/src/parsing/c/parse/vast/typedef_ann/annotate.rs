use super::*;

pub fn c11_annotate_typedef_names(
    vast_nodes: &str,
    haystack: &str,
    haystack_len: Expr,
    num_nodes: Expr,
    out_annotated_vast_nodes: &str,
) -> Program {
    c11_annotate_typedef_names_impl(
        vast_nodes,
        haystack,
        haystack_len,
        num_nodes,
        out_annotated_vast_nodes,
        false,
        false,
        None,
        None,
    )
}

pub fn c11_annotate_typedef_names_packed_haystack(
    vast_nodes: &str,
    haystack: &str,
    haystack_len: Expr,
    num_nodes: Expr,
    out_annotated_vast_nodes: &str,
) -> Program {
    c11_annotate_typedef_names_impl(
        vast_nodes,
        haystack,
        haystack_len,
        num_nodes,
        out_annotated_vast_nodes,
        true,
        false,
        None,
        None,
    )
}

pub fn c11_annotate_typedef_names_precomputed_scope(
    vast_nodes: &str,
    haystack: &str,
    haystack_len: Expr,
    num_nodes: Expr,
    out_annotated_vast_nodes: &str,
) -> Program {
    c11_annotate_typedef_names_impl(
        vast_nodes,
        haystack,
        haystack_len,
        num_nodes,
        out_annotated_vast_nodes,
        false,
        true,
        None,
        None,
    )
}

pub fn c11_annotate_typedef_names_precomputed_scope_packed_haystack(
    vast_nodes: &str,
    haystack: &str,
    haystack_len: Expr,
    num_nodes: Expr,
    out_annotated_vast_nodes: &str,
) -> Program {
    c11_annotate_typedef_names_impl(
        vast_nodes,
        haystack,
        haystack_len,
        num_nodes,
        out_annotated_vast_nodes,
        true,
        true,
        None,
        None,
    )
}

pub fn c11_annotate_typedef_names_precomputed_context(
    vast_nodes: &str,
    haystack: &str,
    decl_contexts: &str,
    visible_type: &str,
    haystack_len: Expr,
    num_nodes: Expr,
    out_annotated_vast_nodes: &str,
) -> Program {
    c11_annotate_typedef_names_impl(
        vast_nodes,
        haystack,
        haystack_len,
        num_nodes,
        out_annotated_vast_nodes,
        false,
        true,
        Some(decl_contexts),
        Some(visible_type),
    )
}

pub fn c11_annotate_typedef_names_precomputed_context_packed_haystack(
    vast_nodes: &str,
    haystack: &str,
    decl_contexts: &str,
    visible_type: &str,
    haystack_len: Expr,
    num_nodes: Expr,
    out_annotated_vast_nodes: &str,
) -> Program {
    c11_annotate_typedef_names_impl(
        vast_nodes,
        haystack,
        haystack_len,
        num_nodes,
        out_annotated_vast_nodes,
        true,
        true,
        Some(decl_contexts),
        Some(visible_type),
    )
}

/// Annotate typedef names over a VAST, emitting the visibility and declaration-kind passes.
///
/// # Panics
/// Panics when precomputed-context annotation is requested without the visible-type
/// side table. The two arrive together from the caller, so a missing table means the
/// wiring is wrong rather than the input.
#[allow(clippy::too_many_arguments)]
pub(super) fn c11_annotate_typedef_names_impl(
    vast_nodes: &str,
    haystack: &str,
    haystack_len: Expr,
    num_nodes: Expr,
    out_annotated_vast_nodes: &str,
    packed_haystack: bool,
    precomputed_scope: bool,
    decl_contexts: Option<&str>,
    visible_type: Option<&str>,
) -> Program {
    let t = Expr::InvocationId { axis: 0 };
    let base = Expr::mul(t.clone(), Expr::u32(VAST_NODE_STRIDE_U32));

    let row = IdentifierRowHash {
        vast_nodes,
        haystack,
        haystack_len: &haystack_len,
        row_base: base.clone(),
        packed_haystack,
        names: IdentifierRowHashNames {
            start: "tok_start",
            len: "tok_len",
            hash: "name_hash",
            cursor: "hash_i",
            byte: "hash_byte",
        },
    };

    let mut loop_body = vec![Node::let_bind(
        "raw_kind",
        Expr::load(vast_nodes, base.clone()),
    )];
    loop_body.extend(row.bindings());
    loop_body.push(row.update(Expr::and(
        Expr::eq(Expr::var("raw_kind"), Expr::u32(TOK_IDENTIFIER)),
        row.hash_is_unset(),
    )));
    loop_body.push(Node::let_bind(
        "scope_open",
        if precomputed_scope {
            Expr::load(
                vast_nodes,
                Expr::add(base.clone(), Expr::u32(VAST_TYPEDEF_SCOPE_FIELD)),
            )
        } else {
            Expr::u32(SENTINEL)
        },
    ));
    loop_body.extend([
        Node::let_bind("scope_depth", Expr::u32(0)),
        Node::let_bind("last_decl_kind", Expr::u32(0)),
        Node::let_bind("typedef_flags", Expr::u32(0)),
        Node::let_bind("annot_num_nodes", num_nodes.clone()),
    ]);

    // The scope walker must run for EVERY row, not just for IDENTIFIER rows.
    // CPU oracle (`reference_c11_annotate_typedef_names_from_words`) writes
    // `scope_open_before(node_idx)` to the SCOPE field unconditionally, so the
    // GPU annotation must populate the scope_open carrier on every invocation
    // before the unconditional store-back loop reads `scope_open` at the end.
    // Gating it inside the `raw_kind == TOK_IDENTIFIER` branch (where it used
    // to live) leaves scope_open at its initial SENTINEL on every non-identifier
    // row, diverging from the CPU oracle on every brace, paren, semicolon, etc.
    if !precomputed_scope {
        loop_body.push(Node::assign(
            "scope_open",
            Expr::call(
                row_phases::SCOPE_OPEN_FOR_ROW_OP_ID,
                vec![Expr::buffer_ref(vast_nodes), t.clone()],
            ),
        ));
    }
    let mut identifier_annotation: Vec<Node> = Vec::new();
    identifier_annotation.extend([
        Node::let_bind(
            "prev_idx",
            Expr::select(
                Expr::gt(t.clone(), Expr::u32(0)),
                Expr::sub(t.clone(), Expr::u32(1)),
                Expr::u32(0),
            ),
        ),
        Node::let_bind(
            "prev_kind_loaded",
            Expr::load(
                vast_nodes,
                Expr::mul(Expr::var("prev_idx"), Expr::u32(VAST_NODE_STRIDE_U32)),
            ),
        ),
        Node::let_bind(
            "prev_kind",
            Expr::select(
                Expr::gt(t.clone(), Expr::u32(0)),
                Expr::var("prev_kind_loaded"),
                Expr::u32(SENTINEL),
            ),
        ),
        Node::let_bind(
            "next_idx",
            Expr::select(
                Expr::lt(Expr::add(t.clone(), Expr::u32(1)), num_nodes.clone()),
                Expr::add(t.clone(), Expr::u32(1)),
                t.clone(),
            ),
        ),
        Node::let_bind(
            "next_kind_loaded",
            Expr::load(
                vast_nodes,
                Expr::mul(Expr::var("next_idx"), Expr::u32(VAST_NODE_STRIDE_U32)),
            ),
        ),
        Node::let_bind(
            "next_kind",
            Expr::select(
                Expr::lt(Expr::add(t.clone(), Expr::u32(1)), num_nodes.clone()),
                Expr::var("next_kind_loaded"),
                Expr::u32(SENTINEL),
            ),
        ),
    ]);
    // The CPU oracle (`reference_c11_annotate_typedef_names_from_words`)
    // resolves typedef visibility for every IDENTIFIER row that is not itself
    // a declarator, regardless of preceding tokens. The previous
    // `needs_typedef_visibility` gate excluded identifiers preceded by
    // STRUCT/UNION/ENUM/DOT/ARROW/GOTO or followed by COLON, which made
    // GPU output 0 in the TYPEDEF_FLAGS field for visible typedef names
    // appearing as struct/union/enum tags (e.g. row 5 of the tags fixture
    // where `typedef int S;` is later reused as `struct S { ... }`). The
    // scan produces a per-row result via the carrier; gating it here
    // diverged from the CPU contract on every tag spot.
    if let Some(decl_contexts) = decl_contexts {
        let visible_type = visible_type
            .expect("precomputed-context annotation requires the visible-type side table");
        identifier_annotation.extend(emit_typedef_visibility_scan_precomputed_context(
            vast_nodes,
            decl_contexts,
            visible_type,
            t.clone(),
        ));
    } else {
        identifier_annotation.push(Node::let_bind(
            "current_visible_typedef_name",
            Expr::call(
                if packed_haystack {
                    row_phases::VISIBLE_NAME_FOR_ROW_PACKED_OP_ID
                } else {
                    row_phases::VISIBLE_NAME_FOR_ROW_OP_ID
                },
                vec![
                    Expr::buffer_ref(vast_nodes),
                    Expr::buffer_ref(haystack),
                    t.clone(),
                    haystack_len.clone(),
                    num_nodes.clone(),
                ],
            ),
        ));
        identifier_annotation.push(Node::assign(
            "last_decl_kind",
            Expr::select(
                Expr::eq(Expr::var("current_visible_typedef_name"), Expr::u32(1)),
                Expr::u32(1),
                Expr::u32(0),
            ),
        ));
    }
    identifier_annotation.extend([
        Node::let_bind(
            "possible_declarator",
            is_typedef_symbol_link_follower_token(Expr::var("next_kind")),
        ),
        Node::let_bind("current_decl_flags", Expr::u32(0)),
        Node::let_bind(
            "declaration_candidate",
            Expr::and(
                Expr::var("possible_declarator"),
                Expr::and(
                    Expr::not(is_declaration_previous_disqualifier_token(Expr::var(
                        "prev_kind",
                    ))),
                    Expr::ne(Expr::var("next_kind"), Expr::u32(TOK_COLON)),
                ),
            ),
        ),
    ]);
    let mut declaration_annotation = if let Some(decl_contexts) = decl_contexts {
        // The precomputed-context path is only correct when it is paired with the
        // per-node visible-type table (`c11_precompute_vast_visible_type`); the two
        // public wrappers require both, so this unwrap is an internal invariant.
        let visible_type = visible_type
            .expect("precomputed-context annotation requires the visible-type side table");
        let mut nodes = emit_precomputed_declaration_kind_for_index(
            vast_nodes,
            decl_contexts,
            visible_type,
            t.clone(),
            "current_decl_result_kind",
            "current_decl_precomputed",
        );
        nodes.push(Node::assign(
            "current_decl_flags",
            Expr::select(
                Expr::eq(Expr::var("current_decl_result_kind"), Expr::u32(1)),
                Expr::u32(C_TYPEDEF_FLAG_TYPEDEF_DECLARATOR),
                Expr::select(
                    Expr::eq(Expr::var("current_decl_result_kind"), Expr::u32(2)),
                    Expr::u32(C_TYPEDEF_FLAG_ORDINARY_DECLARATOR),
                    Expr::u32(0),
                ),
            ),
        ));
        nodes
    } else {
        vec![
            Node::let_bind(
                "current_decl_result_kind",
                Expr::call(
                    if packed_haystack {
                        row_phases::DECL_KIND_FOR_ROW_PACKED_OP_ID
                    } else {
                        row_phases::DECL_KIND_FOR_ROW_OP_ID
                    },
                    vec![
                        Expr::buffer_ref(vast_nodes),
                        Expr::buffer_ref(haystack),
                        t.clone(),
                        haystack_len.clone(),
                        num_nodes.clone(),
                    ],
                ),
            ),
            Node::assign(
                "current_decl_flags",
                Expr::select(
                    Expr::eq(Expr::var("current_decl_result_kind"), Expr::u32(1)),
                    Expr::u32(C_TYPEDEF_FLAG_TYPEDEF_DECLARATOR),
                    Expr::select(
                        Expr::eq(Expr::var("current_decl_result_kind"), Expr::u32(2)),
                        Expr::u32(C_TYPEDEF_FLAG_ORDINARY_DECLARATOR),
                        Expr::u32(0),
                    ),
                ),
            ),
        ]
    };

    declaration_annotation.extend([
        Node::if_then(
            Expr::and(
                Expr::eq(Expr::var("raw_kind"), Expr::u32(TOK_IDENTIFIER)),
                Expr::and(
                    Expr::eq(Expr::var("last_decl_kind"), Expr::u32(1)),
                    Expr::eq(Expr::var("current_decl_result_kind"), Expr::u32(0)),
                ),
            ),
            vec![Node::assign(
                "typedef_flags",
                Expr::bitor(
                    Expr::var("typedef_flags"),
                    Expr::u32(C_TYPEDEF_FLAG_VISIBLE_TYPEDEF_NAME),
                ),
            )],
        ),
        Node::if_then(
            is_typedef_declarator_annotation(Expr::var("current_decl_flags")),
            vec![Node::assign(
                "typedef_flags",
                Expr::bitor(
                    Expr::var("typedef_flags"),
                    Expr::u32(C_TYPEDEF_FLAG_TYPEDEF_DECLARATOR),
                ),
            )],
        ),
        Node::if_then(
            is_ordinary_declarator_annotation(Expr::var("current_decl_flags")),
            vec![Node::assign(
                "typedef_flags",
                Expr::bitor(
                    Expr::var("typedef_flags"),
                    Expr::u32(C_TYPEDEF_FLAG_ORDINARY_DECLARATOR),
                ),
            )],
        ),
    ]);
    identifier_annotation.push(Node::if_then(
        Expr::var("declaration_candidate"),
        declaration_annotation,
    ));
    identifier_annotation.push(Node::if_then(
        Expr::and(
            Expr::not(Expr::var("declaration_candidate")),
            Expr::eq(Expr::var("last_decl_kind"), Expr::u32(1)),
        ),
        vec![Node::assign(
            "typedef_flags",
            Expr::bitor(
                Expr::var("typedef_flags"),
                Expr::u32(C_TYPEDEF_FLAG_VISIBLE_TYPEDEF_NAME),
            ),
        )],
    ));
    loop_body.push(Node::if_then(
        Expr::eq(Expr::var("raw_kind"), Expr::u32(TOK_IDENTIFIER)),
        identifier_annotation,
    ));

    for field in 0..VAST_NODE_STRIDE_U32 {
        let value = match field {
            VAST_TYPEDEF_FLAGS_FIELD => Expr::var("typedef_flags"),
            VAST_TYPEDEF_SCOPE_FIELD => Expr::var("scope_open"),
            VAST_TYPEDEF_SYMBOL_FIELD => Expr::var("name_hash"),
            _ => Expr::load(vast_nodes, Expr::add(base.clone(), Expr::u32(field))),
        };
        loop_body.push(Node::store(
            out_annotated_vast_nodes,
            Expr::add(base.clone(), Expr::u32(field)),
            value,
        ));
    }

    let n = node_count(&num_nodes).max(1);
    let mut buffers = vec![
        BufferDecl::storage(vast_nodes, 0, BufferAccess::ReadOnly, DataType::U32)
            .with_count(n.saturating_mul(VAST_NODE_STRIDE_U32)),
        BufferDecl::storage(haystack, 1, BufferAccess::ReadOnly, DataType::U32)
            .with_count(haystack_word_count(&haystack_len, packed_haystack)),
    ];
    let out_binding = if let Some(decl_contexts) = decl_contexts {
        let visible_type = visible_type
            .expect("precomputed-context annotation requires the visible-type side table");
        buffers.push(
            BufferDecl::storage(decl_contexts, 2, BufferAccess::ReadOnly, DataType::U32)
                .with_count(n.saturating_mul(VAST_DECL_CONTEXT_STRIDE_U32)),
        );
        buffers.push(
            BufferDecl::storage(visible_type, 3, BufferAccess::ReadOnly, DataType::U32)
                .with_count(n),
        );
        4
    } else {
        2
    };
    buffers.push(
        BufferDecl::output(out_annotated_vast_nodes, out_binding, DataType::U32)
            .with_count(n.saturating_mul(VAST_NODE_STRIDE_U32)),
    );
    Program::wrapped(
        buffers,
        [256, 1, 1],
        vec![wrap_anonymous(
            ANNOTATE_TYPEDEF_OP_ID,
            vec![Node::if_then(Expr::lt(t, num_nodes), loop_body)],
        )],
    )
    .with_entry_op_id(ANNOTATE_TYPEDEF_OP_ID)
    .with_non_composable_with_self(true)
}
