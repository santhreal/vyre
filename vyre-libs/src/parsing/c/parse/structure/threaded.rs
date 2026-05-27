use super::*;

pub(super) const STRUCTURE_WORKGROUP_SIZE: u32 = 256;

pub(super) fn literal_u32_or(expr: &Expr, fallback: u32) -> u32 {
    match expr {
        Expr::LitU32(value) => *value,
        _ => fallback,
    }
}

#[derive(Clone, Copy, Default)]
pub(super) struct TokenContextOptions {
    pub(super) prev_prev_type: bool,
    pub(super) next2_type_and_rparen: bool,
    pub(super) before_wrapper_type: bool,
    pub(super) parenthesized_wrapper_rparen: bool,
    pub(super) after_wrapper_type_and_rparen: bool,
}

pub(super) fn emit_token_context(
    tok_types: &str,
    paren_pairs: &str,
    num_tokens: &Expr,
    t: &Expr,
    options: TokenContextOptions,
) -> Vec<Node> {
    let mut nodes = vec![
        Node::let_bind("tok_type", Expr::load(tok_types, t.clone())),
        Node::let_bind("prev_type", Expr::u32(0)),
        Node::if_then(
            Expr::gt(t.clone(), Expr::u32(0)),
            vec![Node::assign(
                "prev_type",
                Expr::load(tok_types, Expr::sub(t.clone(), Expr::u32(1))),
            )],
        ),
        Node::let_bind(
            "next_type",
            Expr::load(tok_types, Expr::add(t.clone(), Expr::u32(1))),
        ),
        Node::let_bind(
            "matching_rparen",
            Expr::load(paren_pairs, Expr::add(t.clone(), Expr::u32(1))),
        ),
    ];

    if options.prev_prev_type {
        nodes.extend([
            Node::let_bind("prev_prev_type", Expr::u32(0)),
            Node::if_then(
                Expr::gt(t.clone(), Expr::u32(1)),
                vec![Node::assign(
                    "prev_prev_type",
                    Expr::load(tok_types, Expr::sub(t.clone(), Expr::u32(2))),
                )],
            ),
        ]);
    }

    if options.next2_type_and_rparen {
        nodes.extend([
            Node::let_bind("next2_type", Expr::u32(0)),
            Node::let_bind("numeric_suffix_rparen", Expr::u32(u32::MAX)),
            Node::if_then(
                Expr::lt(Expr::add(t.clone(), Expr::u32(2)), num_tokens.clone()),
                vec![
                    Node::assign(
                        "next2_type",
                        Expr::load(tok_types, Expr::add(t.clone(), Expr::u32(2))),
                    ),
                    Node::assign(
                        "numeric_suffix_rparen",
                        Expr::load(paren_pairs, Expr::add(t.clone(), Expr::u32(2))),
                    ),
                ],
            ),
        ]);
    }

    if options.before_wrapper_type {
        nodes.extend([
            Node::let_bind("before_wrapper_type", Expr::u32(TOK_EOF)),
            Node::if_then(
                Expr::gt(t.clone(), Expr::u32(1)),
                vec![Node::assign(
                    "before_wrapper_type",
                    Expr::load(tok_types, Expr::sub(t.clone(), Expr::u32(2))),
                )],
            ),
        ]);
    }

    if options.parenthesized_wrapper_rparen {
        nodes.extend([
            Node::let_bind("parenthesized_wrapper_rparen", Expr::u32(u32::MAX)),
            Node::if_then(
                Expr::gt(t.clone(), Expr::u32(0)),
                vec![Node::assign(
                    "parenthesized_wrapper_rparen",
                    Expr::load(paren_pairs, Expr::sub(t.clone(), Expr::u32(1))),
                )],
            ),
        ]);
    }

    if options.after_wrapper_type_and_rparen {
        nodes.extend([
            Node::let_bind("after_wrapper_type", Expr::u32(TOK_EOF)),
            Node::let_bind("after_wrapper_rparen", Expr::u32(u32::MAX)),
            Node::if_then(
                Expr::lt(Expr::add(t.clone(), Expr::u32(2)), num_tokens.clone()),
                vec![
                    Node::assign(
                        "after_wrapper_type",
                        Expr::load(tok_types, Expr::add(t.clone(), Expr::u32(2))),
                    ),
                    Node::assign(
                        "after_wrapper_rparen",
                        Expr::load(paren_pairs, Expr::add(t.clone(), Expr::u32(2))),
                    ),
                ],
            ),
        ]);
    }

    nodes
}

pub(super) fn token_pair_input_buffers(
    tok_types: &str,
    paren_pairs: &str,
    tok_count: u32,
) -> Vec<BufferDecl> {
    vec![
        BufferDecl::storage(tok_types, 0, BufferAccess::ReadOnly, DataType::U32)
            .with_count(tok_count),
        BufferDecl::storage(paren_pairs, 1, BufferAccess::ReadOnly, DataType::U32)
            .with_count(tok_count),
    ]
}

pub(super) fn append_record_output_buffers(
    buffers: &mut Vec<BufferDecl>,
    records: &str,
    records_binding: u32,
    record_words: u32,
    record_count: u32,
    counts: &str,
    counts_binding: u32,
    counts_live_out: bool,
) {
    buffers.push(
        BufferDecl::output(records, records_binding, DataType::U32)
            .with_count(record_words.saturating_mul(record_count).max(record_words)),
    );
    let mut counts_decl = BufferDecl::storage(
        counts,
        counts_binding,
        BufferAccess::ReadWrite,
        DataType::U32,
    )
    .with_count(1);
    if counts_live_out {
        counts_decl = counts_decl.with_pipeline_live_out(true);
    }
    buffers.push(counts_decl);
}

pub(super) fn emit_atomic_record_append(
    out_records: &str,
    out_counts: &str,
    record_idx_var: &'static str,
    fields: Vec<Expr>,
) -> Vec<Node> {
    let record_words = fields.len() as u32;
    let mut nodes = vec![Node::let_bind(
        record_idx_var,
        Expr::atomic_add(out_counts, Expr::u32(0), Expr::u32(record_words)),
    )];

    for (field_idx, field) in fields.into_iter().enumerate() {
        let out_index = if field_idx == 0 {
            Expr::var(record_idx_var)
        } else {
            Expr::add(Expr::var(record_idx_var), Expr::u32(field_idx as u32))
        };
        nodes.push(Node::store(out_records, out_index, field));
    }

    nodes
}

pub(super) fn emit_sparse_record_zero(
    out_records: &str,
    t: Expr,
    num_records: Expr,
    record_words: u32,
) -> Vec<Node> {
    let base = "sparse_record_zero_base";
    let mut stores = Vec::new();
    for field_idx in 0..record_words {
        let out_index = if field_idx == 0 {
            Expr::var(base)
        } else {
            Expr::add(Expr::var(base), Expr::u32(field_idx))
        };
        stores.push(Node::store(out_records, out_index, Expr::u32(0)));
    }

    vec![
        Node::let_bind(base, Expr::mul(t.clone(), Expr::u32(record_words))),
        Node::if_then(Expr::lt(t, num_records), stores),
    ]
}

pub(super) fn threaded_structure_program(
    entry_op_id: &'static str,
    buffers: Vec<BufferDecl>,
    pre_loop_nodes: Vec<Node>,
    loop_guard: Expr,
    loop_body: Vec<Node>,
) -> Program {
    let t = Expr::var("t");
    let mut body = vec![
        Node::let_bind("lane", Expr::LocalId { axis: 0 }),
        Node::let_bind("block", Expr::WorkgroupId { axis: 0 }),
        Node::let_bind(
            "t",
            Expr::add(
                Expr::mul(Expr::var("block"), Expr::u32(STRUCTURE_WORKGROUP_SIZE)),
                Expr::var("lane"),
            ),
        ),
    ];
    body.extend(pre_loop_nodes);
    body.push(Node::if_then(loop_guard, loop_body));

    Program::wrapped(
        buffers,
        [STRUCTURE_WORKGROUP_SIZE, 1, 1],
        vec![wrap_anonymous(entry_op_id, body)],
    )
    .with_entry_op_id(entry_op_id)
    .with_non_composable_with_self(true)
}
