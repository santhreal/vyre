use super::*;

/// Bind `out_flag` to 1 when the row's symbol hash occurs in the global
/// typedef-name hash table.
///
/// The fast pass asks this question twice per row, once for the row itself and
/// once for every identifier in its declaration prefix, so the guarded table
/// scan is emitted from one place.
fn emit_global_typedef_hash_flag(
    global_typedef_hashes: &str,
    num_global_typedefs: &Expr,
    scan_var: &str,
    kind: Expr,
    hash: Expr,
    out_flag: &str,
) -> Vec<Node> {
    vec![
        Node::let_bind(out_flag, Expr::u32(0)),
        Node::if_then(
            Expr::and(
                Expr::eq(kind, Expr::u32(TOK_IDENTIFIER)),
                Expr::ne(hash.clone(), Expr::u32(0)),
            ),
            vec![Node::loop_for(
                scan_var,
                Expr::u32(0),
                num_global_typedefs.clone(),
                vec![Node::if_then(
                    Expr::eq(Expr::load(global_typedef_hashes, Expr::var(scan_var)), hash),
                    vec![Node::assign(out_flag, Expr::u32(1))],
                )],
            )],
        ),
    ]
}

/// Emit a backward scan over the declaration prefix that ends at `end`
/// (exclusive), skipping balanced `( ... )` and `{ ... }` groups and stopping
/// at the first declaration-reset token.
///
/// For every visited row the scan binds `{prefix}_scan_idx`,
/// `{prefix}_scan_base` and `{prefix}_scan_kind`, runs `per_row`, then runs
/// `unskipped` only for rows that lie outside every skipped group. The scan
/// owns its stop flag `{prefix}_done`, which `unskipped` sets to end it.
///
/// Group skipping is what makes the prefix of `struct S { int a; } x;` reach
/// back past the record body to `struct`, so this is not merely a reverse walk
/// to the nearest `;`.
fn emit_decl_prefix_back_scan(
    vast_nodes: &str,
    prefix: &str,
    end: Expr,
    per_row: Vec<Node>,
    unskipped: Vec<Node>,
) -> Vec<Node> {
    let done = format!("{prefix}_done");
    let paren_depth = format!("{prefix}_skipped_paren_depth");
    let brace_depth = format!("{prefix}_skipped_brace_depth");
    let scan_idx = format!("{prefix}_scan_idx");
    let scan_base = format!("{prefix}_scan_base");
    let scan_kind = format!("{prefix}_scan_kind");
    let in_paren = format!("{prefix}_in_skipped_paren");
    let in_brace = format!("{prefix}_in_skipped_brace");

    let mut row_body = vec![
        Node::let_bind(
            &scan_idx,
            Expr::sub(
                Expr::sub(end.clone(), Expr::u32(1)),
                Expr::var(format!("{prefix}_back_scan")),
            ),
        ),
        Node::let_bind(&scan_base, vast_row_base_expr(Expr::var(&scan_idx))),
        Node::let_bind(
            &scan_kind,
            vast_row_kind_from_base_expr(vast_nodes, Expr::var(&scan_base)),
        ),
    ];
    row_body.extend(per_row);
    row_body.extend([
        Node::let_bind(
            &in_paren,
            Expr::or(
                Expr::gt(Expr::var(&paren_depth), Expr::u32(0)),
                Expr::eq(Expr::var(&scan_kind), Expr::u32(TOK_RPAREN)),
            ),
        ),
        Node::let_bind(
            &in_brace,
            Expr::or(
                Expr::gt(Expr::var(&brace_depth), Expr::u32(0)),
                Expr::eq(Expr::var(&scan_kind), Expr::u32(TOK_RBRACE)),
            ),
        ),
        Node::if_then(
            Expr::eq(Expr::var(&scan_kind), Expr::u32(TOK_RBRACE)),
            vec![Node::assign(
                &brace_depth,
                Expr::add(Expr::var(&brace_depth), Expr::u32(1)),
            )],
        ),
        Node::if_then(
            Expr::and(
                Expr::gt(Expr::var(&brace_depth), Expr::u32(0)),
                Expr::eq(Expr::var(&scan_kind), Expr::u32(TOK_LBRACE)),
            ),
            vec![Node::assign(
                &brace_depth,
                Expr::sub(Expr::var(&brace_depth), Expr::u32(1)),
            )],
        ),
        Node::if_then(
            Expr::eq(Expr::var(&scan_kind), Expr::u32(TOK_RPAREN)),
            vec![Node::assign(
                &paren_depth,
                Expr::add(Expr::var(&paren_depth), Expr::u32(1)),
            )],
        ),
        Node::if_then(
            Expr::and(
                Expr::gt(Expr::var(&paren_depth), Expr::u32(0)),
                Expr::eq(Expr::var(&scan_kind), Expr::u32(TOK_LPAREN)),
            ),
            vec![Node::assign(
                &paren_depth,
                Expr::sub(Expr::var(&paren_depth), Expr::u32(1)),
            )],
        ),
        Node::if_then(
            Expr::not(Expr::or(Expr::var(&in_brace), Expr::var(&in_paren))),
            unskipped,
        ),
    ]);

    vec![
        Node::let_bind(&done, Expr::u32(0)),
        Node::let_bind(&paren_depth, Expr::u32(0)),
        Node::let_bind(&brace_depth, Expr::u32(0)),
        Node::loop_for(
            format!("{prefix}_back_scan"),
            Expr::u32(0),
            end,
            vec![Node::if_then(
                Expr::eq(Expr::var(&done), Expr::u32(0)),
                row_body,
            )],
        ),
    ]
}

#[must_use]
pub fn c11_annotate_global_typedef_names_fast(
    vast_nodes: &str,
    global_typedef_hashes: &str,
    num_nodes: Expr,
    num_global_typedefs: Expr,
    out_annotated_vast_nodes: &str,
) -> Program {
    let t = Expr::InvocationId { axis: 0 };
    let base = vast_row_base_expr(t.clone());
    let mut loop_body = vec![
        Node::let_bind(
            "raw_kind",
            vast_row_kind_from_base_expr(vast_nodes, base.clone()),
        ),
        Node::let_bind(
            "name_hash",
            Expr::load(
                vast_nodes,
                Expr::add(base.clone(), Expr::u32(VAST_TYPEDEF_SYMBOL_FIELD)),
            ),
        ),
    ];
    loop_body.extend(emit_global_typedef_hash_flag(
        global_typedef_hashes,
        &num_global_typedefs,
        "global_typedef_hash_scan",
        Expr::var("raw_kind"),
        Expr::var("name_hash"),
        "is_global_typedef_hash",
    ));
    loop_body.extend([
        Node::let_bind(
            "prev_kind",
            vast_prior_row_kind_expr(vast_nodes, t.clone(), 1),
        ),
        Node::let_bind(
            "prev_prev_kind",
            vast_prior_row_kind_expr(vast_nodes, t.clone(), 2),
        ),
        Node::let_bind(
            "next_kind",
            vast_next_row_kind_expr(vast_nodes, t.clone(), &num_nodes, Expr::u32(SENTINEL)),
        ),
        Node::let_bind("has_decl_prefix", Expr::u32(0)),
        Node::let_bind("has_typedef_prefix", Expr::u32(0)),
    ]);

    let mut decl_prefix_per_row = vec![Node::let_bind(
        "decl_prefix_scan_name_hash",
        Expr::load(
            vast_nodes,
            Expr::add(
                Expr::var("decl_prefix_scan_base"),
                Expr::u32(VAST_TYPEDEF_SYMBOL_FIELD),
            ),
        ),
    )];
    decl_prefix_per_row.extend(emit_global_typedef_hash_flag(
        global_typedef_hashes,
        &num_global_typedefs,
        "decl_prefix_global_typedef_hash_scan",
        Expr::var("decl_prefix_scan_kind"),
        Expr::var("decl_prefix_scan_name_hash"),
        "decl_prefix_scan_is_typedef_name",
    ));
    loop_body.extend(emit_decl_prefix_back_scan(
        vast_nodes,
        "decl_prefix",
        t.clone(),
        decl_prefix_per_row,
        vec![
            Node::if_then(
                Expr::eq(Expr::var("decl_prefix_scan_kind"), Expr::u32(TOK_TYPEDEF)),
                vec![Node::assign("has_typedef_prefix", Expr::u32(1))],
            ),
            Node::if_then(
                Expr::or(
                    Expr::eq(Expr::var("decl_prefix_scan_is_typedef_name"), Expr::u32(1)),
                    // Deliberately not `is_decl_prefix_token`: that set omits
                    // `auto` and `register`, which this pass must treat as
                    // declaration prefixes, and the two sets have never been
                    // reconciled.
                    any_token_eq(
                        Expr::var("decl_prefix_scan_kind"),
                        &[
                            TOK_TYPEDEF,
                            TOK_INT,
                            TOK_CHAR_KW,
                            TOK_VOID,
                            TOK_DOUBLE,
                            TOK_FLOAT_KW,
                            TOK_LONG,
                            TOK_SHORT,
                            TOK_SIGNED,
                            TOK_UNSIGNED,
                            TOK_BOOL,
                            TOK_STRUCT,
                            TOK_UNION,
                            TOK_ENUM,
                            TOK_AUTO,
                            TOK_CONST,
                            TOK_VOLATILE,
                            TOK_STATIC,
                            TOK_EXTERN,
                            TOK_REGISTER,
                            TOK_RESTRICT,
                            TOK_INLINE,
                            TOK_ALIGNAS,
                            TOK_ATOMIC,
                            TOK_GNU_AUTO_TYPE,
                            TOK_GNU_TYPEOF,
                            TOK_GNU_TYPEOF_UNQUAL,
                            TOK_GNU_INT128,
                            TOK_GNU_BUILTIN_VA_LIST,
                            TOK_FLOAT16_KW,
                            TOK_FLOAT32_KW,
                            TOK_FLOAT64_KW,
                            TOK_FLOAT128_KW,
                            TOK_GNU_FLOAT128_KW,
                            TOK_GNU_BF16_KW,
                            TOK_GNU_FP16_KW,
                        ],
                    ),
                ),
                vec![Node::assign("has_decl_prefix", Expr::u32(1))],
            ),
            Node::if_then(
                is_decl_prefix_reset_token(Expr::var("decl_prefix_scan_kind")),
                vec![Node::assign("decl_prefix_done", Expr::u32(1))],
            ),
        ],
    ));
    loop_body.extend([
        Node::let_bind(
            "scope_open",
            Expr::load(
                vast_nodes,
                Expr::add(base.clone(), Expr::u32(VAST_TYPEDEF_SCOPE_FIELD)),
            ),
        ),
        Node::let_bind("in_aggregate_body", Expr::bool(false)),
        Node::let_bind("aggregate_scan_done", Expr::bool(false)),
        // Walks back from the enclosing `{` rather than testing only the two
        // rows in front of it, so an attributed aggregate
        // (`struct S __attribute__((packed)) { ... }`) is still recognised as a
        // record body. `is_aggregate_specifier_body_open` answers the same
        // question in constant time but misses that spelling.
        Node::loop_for(
            "aggregate_scope_back_scan",
            Expr::u32(0),
            Expr::select(
                Expr::ne(Expr::var("scope_open"), Expr::u32(SENTINEL)),
                Expr::var("scope_open"),
                Expr::u32(0),
            ),
            vec![Node::if_then(
                Expr::not(Expr::var("aggregate_scan_done")),
                vec![
                    Node::let_bind(
                        "aggregate_scan_idx",
                        Expr::sub(
                            Expr::sub(Expr::var("scope_open"), Expr::u32(1)),
                            Expr::var("aggregate_scope_back_scan"),
                        ),
                    ),
                    Node::let_bind(
                        "aggregate_scan_kind",
                        vast_row_kind_expr(vast_nodes, Expr::var("aggregate_scan_idx")),
                    ),
                    Node::if_then(
                        any_token_eq(
                            Expr::var("aggregate_scan_kind"),
                            &[TOK_STRUCT, TOK_UNION, TOK_ENUM],
                        ),
                        vec![
                            Node::assign("in_aggregate_body", Expr::bool(true)),
                            Node::assign("aggregate_scan_done", Expr::bool(true)),
                        ],
                    ),
                    Node::if_then(
                        is_decl_prefix_reset_token(Expr::var("aggregate_scan_kind")),
                        vec![Node::assign("aggregate_scan_done", Expr::bool(true))],
                    ),
                ],
            )],
        ),
        Node::let_bind(
            "possible_declarator",
            is_typedef_symbol_link_follower_token(Expr::var("next_kind")),
        ),
        Node::let_bind(
            "declaration_candidate",
            Expr::and(
                Expr::var("possible_declarator"),
                Expr::and(
                    Expr::not(is_declaration_previous_disqualifier_token(Expr::var(
                        "prev_kind",
                    ))),
                    Expr::and(
                        Expr::ne(Expr::var("next_kind"), Expr::u32(TOK_COLON)),
                        Expr::and(
                            Expr::not(Expr::and(
                                Expr::eq(Expr::var("prev_kind"), Expr::u32(TOK_STAR)),
                                Expr::eq(Expr::var("prev_prev_kind"), Expr::u32(TOK_RPAREN)),
                            )),
                            Expr::and(
                                Expr::eq(Expr::var("has_decl_prefix"), Expr::u32(1)),
                                Expr::not(Expr::var("in_aggregate_body")),
                            ),
                        ),
                    ),
                ),
            ),
        ),
        Node::let_bind(
            "typedef_name_context",
            Expr::and(
                Expr::not(is_declaration_previous_disqualifier_token(Expr::var(
                    "prev_kind",
                ))),
                Expr::ne(Expr::var("next_kind"), Expr::u32(TOK_COLON)),
            ),
        ),
        Node::let_bind("has_prior_same_hash", Expr::u32(0)),
        Node::let_bind("prior_same_hash_done", Expr::u32(0)),
    ]);

    let mut prior_same_hash_body = vec![
        Node::let_bind("prior_same_hash_has_typedef", Expr::u32(0)),
        Node::assign("prior_same_hash_done", Expr::u32(1)),
    ];
    prior_same_hash_body.extend(emit_decl_prefix_back_scan(
        vast_nodes,
        "prior_same_hash_prefix",
        Expr::var("prior_typedef_hash_idx"),
        Vec::new(),
        vec![
            Node::if_then(
                Expr::eq(
                    Expr::var("prior_same_hash_prefix_scan_kind"),
                    Expr::u32(TOK_TYPEDEF),
                ),
                vec![
                    Node::assign("prior_same_hash_has_typedef", Expr::u32(1)),
                    Node::assign("prior_same_hash_prefix_done", Expr::u32(1)),
                ],
            ),
            Node::if_then(
                is_decl_prefix_reset_token(Expr::var("prior_same_hash_prefix_scan_kind")),
                vec![Node::assign("prior_same_hash_prefix_done", Expr::u32(1))],
            ),
        ],
    ));
    prior_same_hash_body.push(Node::assign(
        "has_prior_same_hash",
        Expr::var("prior_same_hash_has_typedef"),
    ));

    loop_body.push(Node::loop_for(
        "prior_typedef_hash_scan",
        Expr::u32(0),
        t.clone(),
        vec![Node::if_then(
            Expr::eq(Expr::var("prior_same_hash_done"), Expr::u32(0)),
            vec![
                Node::let_bind(
                    "prior_typedef_hash_idx",
                    Expr::sub(
                        Expr::sub(t.clone(), Expr::u32(1)),
                        Expr::var("prior_typedef_hash_scan"),
                    ),
                ),
                Node::let_bind(
                    "prior_typedef_hash_base",
                    vast_row_base_expr(Expr::var("prior_typedef_hash_idx")),
                ),
                Node::let_bind(
                    "prior_typedef_hash_kind",
                    vast_row_kind_from_base_expr(vast_nodes, Expr::var("prior_typedef_hash_base")),
                ),
                Node::if_then(
                    Expr::and(
                        Expr::eq(
                            Expr::var("prior_typedef_hash_kind"),
                            Expr::u32(TOK_IDENTIFIER),
                        ),
                        Expr::eq(
                            Expr::load(
                                vast_nodes,
                                Expr::add(
                                    Expr::var("prior_typedef_hash_base"),
                                    Expr::u32(VAST_TYPEDEF_SYMBOL_FIELD),
                                ),
                            ),
                            Expr::var("name_hash"),
                        ),
                    ),
                    prior_same_hash_body,
                ),
            ],
        )],
    ));

    loop_body.extend([
        Node::let_bind("typedef_flags", Expr::u32(0)),
        Node::if_then(
            Expr::and(
                Expr::eq(Expr::var("raw_kind"), Expr::u32(TOK_IDENTIFIER)),
                Expr::and(
                    Expr::eq(Expr::var("is_global_typedef_hash"), Expr::u32(1)),
                    Expr::eq(Expr::var("has_prior_same_hash"), Expr::u32(1)),
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
            Expr::and(
                Expr::eq(Expr::var("raw_kind"), Expr::u32(TOK_IDENTIFIER)),
                Expr::var("declaration_candidate"),
            ),
            vec![Node::assign(
                "typedef_flags",
                Expr::select(
                    Expr::eq(Expr::var("has_typedef_prefix"), Expr::u32(1)),
                    Expr::u32(C_TYPEDEF_FLAG_TYPEDEF_DECLARATOR),
                    Expr::u32(C_TYPEDEF_FLAG_ORDINARY_DECLARATOR),
                ),
            )],
        ),
    ]);
    loop_body.extend(store_row_with_overrides(
        out_annotated_vast_nodes,
        vast_nodes,
        &base,
        &[(VAST_TYPEDEF_FLAGS_FIELD, "typedef_flags")],
    ));

    let rows = declared_rows(&num_nodes);
    Program::wrapped(
        vec![
            vast_nodes_input(vast_nodes, 0, rows),
            BufferDecl::storage(
                global_typedef_hashes,
                1,
                BufferAccess::ReadOnly,
                DataType::U32,
            )
            .with_count(declared_rows(&num_global_typedefs)),
            vast_nodes_scratch(out_annotated_vast_nodes, 2, rows),
        ],
        [256, 1, 1],
        vec![wrap_anonymous_region(
            ANNOTATE_TYPEDEF_OP_ID,
            vec![Node::if_then(Expr::lt(t, num_nodes), loop_body)],
        )],
    )
    .with_entry_op_id(ANNOTATE_TYPEDEF_OP_ID)
}
