use super::*;

/// Precompute, per VAST node, whether that node is an identifier that acts as a
/// TYPE in a declaration prefix, i.e. a visible typedef-name or a GNU `typeof`
/// keyword-hash identifier. Writes a `0/1` word per node into `out_visible_type`.
///
/// This is the separate pass that unblocks the precomputed-CONTEXT typedef
/// annotation variant. The base annotator (`c11_annotate_typedef_names`) resolves
/// typedef-name-as-type INLINE for every prefix identifier via
/// `emit_visible_typedef_name_for_index`; the precomputed-context declaration-kind
/// path is haystack-free and can only match builtin type KEYWORDS, so it dropped
/// the ordinary-declarator flag for `T x;` where `T` is a typedef-name (the
/// divergence recorded as WIRING-vyrelibs-precomputed-context-divergence). Rather
/// than re-run the O(chain) resolver inside every annotate invocation, we resolve
/// it ONCE per node here, reading the already-materialized `decl_contexts` table
///: and the annotate path just reads this bit. Correct AND faster than the inline
/// re-resolution (Law 7): the resolver runs N times total, not N-per-declaration.
///
/// Ordering: this pass reads the COMPLETED `decl_contexts` table (produced by
/// `c11_precompute_vast_decl_contexts`) and `vast_nodes`/`haystack`, and writes its
/// own output buffer. Every invocation touches only its own output word, so it is
/// race-free by construction and safe to dispatch in parallel after the
/// decl-context pass has fully settled.
pub fn c11_precompute_vast_visible_type(
    vast_nodes: &str,
    haystack: &str,
    decl_contexts: &str,
    haystack_len: Expr,
    num_nodes: Expr,
    out_visible_type: &str,
) -> Program {
    c11_precompute_vast_visible_type_impl(
        vast_nodes,
        haystack,
        decl_contexts,
        haystack_len,
        num_nodes,
        out_visible_type,
        false,
    )
}

/// Packed-haystack variant of [`c11_precompute_vast_visible_type`]: the source
/// bytes are packed 4-per-word rather than one byte per word.
pub fn c11_precompute_vast_visible_type_packed_haystack(
    vast_nodes: &str,
    haystack: &str,
    decl_contexts: &str,
    haystack_len: Expr,
    num_nodes: Expr,
    out_visible_type: &str,
) -> Program {
    c11_precompute_vast_visible_type_impl(
        vast_nodes,
        haystack,
        decl_contexts,
        haystack_len,
        num_nodes,
        out_visible_type,
        true,
    )
}

fn c11_precompute_vast_visible_type_impl(
    vast_nodes: &str,
    haystack: &str,
    decl_contexts: &str,
    haystack_len: Expr,
    num_nodes: Expr,
    out_visible_type: &str,
    packed_haystack: bool,
) -> Program {
    let t = Expr::InvocationId { axis: 0 };
    let base = Expr::mul(t.clone(), Expr::u32(VAST_NODE_STRIDE_U32));

    let mut row_body = vec![
        // `emit_visible_typedef_name_for_index` reads `annot_num_nodes` for its
        // forward-neighbour bound; bind it the same way the annotate loop does.
        Node::let_bind("annot_num_nodes", num_nodes.clone()),
        Node::let_bind("vt_kind", Expr::load(vast_nodes, base)),
        Node::let_bind("vt_result", Expr::u32(0)),
    ];

    // Only identifiers can be a type-name; keywords/punctuation never are. This
    // mirrors the `prefix_kind == TOK_IDENTIFIER` gate the base prefix scan applies
    // before it runs the GNU-typeof / visible-typedef lookups.
    let mut ident_body = emit_identifier_source_hash_for_index(
        vast_nodes,
        haystack,
        &haystack_len,
        t.clone(),
        "vt_hash",
        "vt_hash",
        packed_haystack,
    );
    ident_body.push(Node::if_then(
        is_gnu_typeof_symbol_hash(Expr::var("vt_hash")),
        vec![Node::assign("vt_result", Expr::u32(1))],
    ));
    ident_body.extend(emit_visible_typedef_name_for_index(
        vast_nodes,
        haystack,
        Some(decl_contexts),
        &haystack_len,
        t.clone(),
        "vt_visible",
        "vt_typedef",
        packed_haystack,
    ));
    ident_body.push(Node::if_then(
        Expr::eq(Expr::var("vt_visible"), Expr::u32(1)),
        vec![Node::assign("vt_result", Expr::u32(1))],
    ));
    row_body.push(Node::if_then(
        Expr::eq(Expr::var("vt_kind"), Expr::u32(TOK_IDENTIFIER)),
        ident_body,
    ));
    row_body.push(Node::store(
        out_visible_type,
        t.clone(),
        Expr::var("vt_result"),
    ));

    let n = node_count(&num_nodes).max(1);
    Program::wrapped(
        vec![
            BufferDecl::storage(vast_nodes, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(n.saturating_mul(VAST_NODE_STRIDE_U32)),
            BufferDecl::storage(haystack, 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(haystack_word_count(&haystack_len, packed_haystack)),
            BufferDecl::storage(decl_contexts, 2, BufferAccess::ReadOnly, DataType::U32)
                .with_count(n.saturating_mul(VAST_DECL_CONTEXT_STRIDE_U32)),
            BufferDecl::output(out_visible_type, 3, DataType::U32).with_count(n),
        ],
        [256, 1, 1],
        vec![wrap_anonymous(
            PRECOMPUTE_VAST_VISIBLE_TYPE_OP_ID,
            vec![Node::if_then(Expr::lt(t, num_nodes), row_body)],
        )],
    )
    .with_entry_op_id(PRECOMPUTE_VAST_VISIBLE_TYPE_OP_ID)
    .with_non_composable_with_self(true)
}
