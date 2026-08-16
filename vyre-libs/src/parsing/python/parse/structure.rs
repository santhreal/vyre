use super::walk::{pack_sparse_tokens, DottedName, TokenPass};
use super::{
    find_matching_delimiter, find_matching_delimiter_into, load_u32, search_next_token,
    search_next_token_into, search_prev_token, store_words, write_words,
};
use crate::parsing::python::{
    DEF_RECORD_WORDS, IMPORT_RECORD_WORDS, INVALID_POS, WITH_RECORD_WORDS,
};
use vyre_foundation::ir::{Expr, Node, Program};
use vyre_spec::python_token::{
    TOK_ASYNC, TOK_CLASS, TOK_COLON, TOK_COMMA, TOK_DEF, TOK_FROM, TOK_IDENTIFIER, TOK_IMPORT,
    TOK_LBRACKET, TOK_LPAREN, TOK_RBRACKET, TOK_RPAREN, TOK_WITH,
};

const STRUCTURE_OP_ID: &str = "vyre-libs::parsing::python312_extract_structure";
const IMPORTS_OP_ID: &str = "vyre-libs::parsing::python312_extract_imports";
const WITH_BLOCKS_OP_ID: &str = "vyre-libs::parsing::python312_extract_with_blocks";

fn line_index_pass<'a>(
    op_id: &'a str,
    tok_types: &'a str,
    tok_starts: &'a str,
    tok_lens: &'a str,
    haystack_len: u32,
) -> TokenPass<'a> {
    TokenPass {
        op_id,
        child_op_id: crate::text::LINE_INDEX_OP_ID,
        tok_types,
        tok_starts,
        tok_lens,
        haystack_len,
    }
}

/// Extract `def`, `async def`, and `class` declarations.
#[must_use]
pub fn python312_extract_structure(
    tok_types: &str,
    tok_starts: &str,
    tok_lens: &str,
    out_records: &str,
    out_counts: &str,
    haystack_len: u32,
) -> Program {
    let t = Expr::InvocationId { axis: 0 };
    let mut body = vec![
        Node::let_bind("tok", load_u32(tok_types, t.clone())),
        Node::let_bind("emit_kind", Expr::u32(0)),
        Node::let_bind("keyword_pos", Expr::u32(INVALID_POS)),
        Node::if_then(
            Expr::eq(Expr::var("tok"), Expr::u32(TOK_DEF)),
            vec![
                Node::assign("emit_kind", Expr::u32(1)),
                Node::assign("keyword_pos", t.clone()),
            ],
        ),
        Node::if_then(
            Expr::eq(Expr::var("tok"), Expr::u32(TOK_CLASS)),
            vec![
                Node::assign("emit_kind", Expr::u32(3)),
                Node::assign("keyword_pos", t.clone()),
            ],
        ),
    ];
    body.extend(search_next_token(
        "async_next",
        Expr::add(t.clone(), Expr::u32(1)),
        tok_types,
        haystack_len,
    ));
    body.push(Node::if_then(
        Expr::and(
            Expr::eq(Expr::var("tok"), Expr::u32(TOK_ASYNC)),
            Expr::eq(
                load_u32(tok_types, Expr::var("async_next")),
                Expr::u32(TOK_DEF),
            ),
        ),
        vec![
            Node::assign("emit_kind", Expr::u32(2)),
            Node::assign("keyword_pos", Expr::var("async_next")),
        ],
    ));
    body.extend(search_next_token(
        "name_pos",
        Expr::add(Expr::var("keyword_pos"), Expr::u32(1)),
        tok_types,
        haystack_len,
    ));
    body.extend(search_next_token(
        "post_name",
        Expr::add(Expr::var("name_pos"), Expr::u32(1)),
        tok_types,
        haystack_len,
    ));
    body.extend(find_matching_delimiter(
        "type_params_end",
        Expr::var("post_name"),
        tok_types,
        haystack_len,
        TOK_LBRACKET,
        TOK_RBRACKET,
    ));
    body.push(Node::if_then(
        Expr::and(
            Expr::ne(Expr::var("emit_kind"), Expr::u32(0)),
            Expr::eq(
                load_u32(tok_types, Expr::var("name_pos")),
                Expr::u32(TOK_IDENTIFIER),
            ),
        ),
        vec![
            Node::let_bind("params_start", Expr::u32(INVALID_POS)),
            Node::let_bind("params_end", Expr::u32(INVALID_POS)),
            Node::let_bind("colon_pos", Expr::u32(INVALID_POS)),
            // Hoist `after_type_params` and `after_params` to the
            // outer scope so the if-block bodies (which assign them)
            // and the later if-blocks (which read them) share one
            // binding. Pre-T-V2 the per-branch `Node::let_bind` lived
            // inside each block, the validator scoped the binding to
            // the block, and the read sites failed with "reference to
            // undeclared variable `after_type_params`" / `after_params`.
            Node::let_bind("after_type_params", Expr::u32(INVALID_POS)),
            Node::let_bind("after_params", Expr::u32(INVALID_POS)),
            Node::if_then_else(
                Expr::eq(
                    load_u32(tok_types, Expr::var("post_name")),
                    Expr::u32(TOK_LBRACKET),
                ),
                search_next_token_into(
                    "after_type_params",
                    Expr::add(Expr::var("type_params_end"), Expr::u32(1)),
                    tok_types,
                    haystack_len,
                ),
                vec![Node::assign("after_type_params", Expr::var("post_name"))],
            ),
            Node::if_then(
                Expr::eq(
                    load_u32(tok_types, Expr::var("after_type_params")),
                    Expr::u32(TOK_LPAREN),
                ),
                vec![
                    Node::assign("params_start", Expr::var("after_type_params")),
                    Node::assign("params_end", Expr::u32(INVALID_POS)),
                ]
                .into_iter()
                .chain(find_matching_delimiter_into(
                    "params_end",
                    Expr::var("after_type_params"),
                    tok_types,
                    haystack_len,
                    TOK_LPAREN,
                    TOK_RPAREN,
                ))
                .collect(),
            ),
            Node::if_then_else(
                Expr::ne(Expr::var("params_end"), Expr::u32(INVALID_POS)),
                search_next_token_into(
                    "after_params",
                    Expr::add(Expr::var("params_end"), Expr::u32(1)),
                    tok_types,
                    haystack_len,
                ),
                vec![Node::assign("after_params", Expr::var("after_type_params"))],
            ),
            Node::if_then(
                Expr::eq(
                    load_u32(tok_types, Expr::var("after_params")),
                    Expr::u32(TOK_COLON),
                ),
                vec![Node::assign("colon_pos", Expr::var("after_params"))],
            ),
            Node::let_bind(
                "slot",
                Expr::atomic_add(out_counts, Expr::u32(0), Expr::u32(DEF_RECORD_WORDS)),
            ),
        ]
        .into_iter()
        .chain(store_words(
            out_records,
            "slot",
            &[
                Expr::var("emit_kind"),
                load_u32(tok_starts, Expr::var("name_pos")),
                load_u32(tok_lens, Expr::var("name_pos")),
                Expr::var("params_start"),
                Expr::var("params_end"),
                Expr::var("colon_pos"),
            ],
        ))
        .collect(),
    ));

    let pass = line_index_pass(
        STRUCTURE_OP_ID,
        tok_types,
        tok_starts,
        tok_lens,
        haystack_len,
    );
    let mut buffers = pass.token_buffers();
    buffers.extend(pass.record_buffers(out_records, out_counts, 3, DEF_RECORD_WORDS));
    pass.program(buffers, body)
}

/// Extract `import` and `from ... import ...` statements.
#[must_use]
pub fn python312_extract_imports(
    tok_types: &str,
    tok_starts: &str,
    tok_lens: &str,
    out_records: &str,
    out_counts: &str,
    haystack_len: u32,
) -> Program {
    let t = Expr::InvocationId { axis: 0 };
    let name = DottedName {
        tok_types,
        haystack_len,
        head: t.clone(),
        accumulator: "name_end",
    };
    let mut body = vec![
        Node::let_bind("tok", load_u32(tok_types, t.clone())),
        Node::let_bind("record_kind", Expr::u32(0)),
    ];
    body.extend(search_prev_token("prev_tok", t.clone(), tok_types));
    body.extend(search_next_token(
        "next_tok",
        Expr::add(t.clone(), Expr::u32(1)),
        tok_types,
        haystack_len,
    ));
    body.push(Node::if_then(
        Expr::and(
            Expr::eq(Expr::var("tok"), Expr::u32(TOK_IDENTIFIER)),
            Expr::or(
                Expr::eq(
                    load_u32(tok_types, Expr::var("prev_tok")),
                    Expr::u32(TOK_IMPORT),
                ),
                Expr::eq(
                    load_u32(tok_types, Expr::var("prev_tok")),
                    Expr::u32(TOK_FROM),
                ),
            ),
        ),
        vec![Node::assign(
            "record_kind",
            Expr::select(
                Expr::eq(
                    load_u32(tok_types, Expr::var("prev_tok")),
                    Expr::u32(TOK_IMPORT),
                ),
                Expr::u32(1),
                Expr::u32(2),
            ),
        )],
    ));
    body.push(Node::if_then(
        Expr::and(
            Expr::eq(Expr::var("tok"), Expr::u32(TOK_IDENTIFIER)),
            Expr::eq(
                load_u32(tok_types, Expr::var("prev_tok")),
                Expr::u32(TOK_COMMA),
            ),
        ),
        vec![Node::assign("record_kind", Expr::u32(1))],
    ));
    let span = name.span(tok_starts, tok_lens);
    body.push(Node::if_then(
        Expr::ne(Expr::var("record_kind"), Expr::u32(0)),
        name.carriers()
            .into_iter()
            .chain([
                name.walk(),
                Node::let_bind(
                    "slot",
                    Expr::atomic_add(out_counts, Expr::u32(0), Expr::u32(IMPORT_RECORD_WORDS)),
                ),
            ])
            .chain(store_words(
                out_records,
                "slot",
                &[
                    Expr::var("record_kind"),
                    span[0].clone(),
                    span[1].clone(),
                    Expr::var("prev_tok"),
                    Expr::var("name_end"),
                    Expr::var("next_tok"),
                ],
            ))
            .collect(),
    ));

    let pass = line_index_pass(IMPORTS_OP_ID, tok_types, tok_starts, tok_lens, haystack_len);
    let mut buffers = pass.token_buffers();
    buffers.extend(pass.record_buffers(out_records, out_counts, 3, IMPORT_RECORD_WORDS));
    pass.program(buffers, body)
}

/// Extract `with` / `async with` headers.
#[must_use]
pub fn python312_extract_with_blocks(
    tok_types: &str,
    tok_starts: &str,
    tok_lens: &str,
    out_records: &str,
    out_counts: &str,
    haystack_len: u32,
) -> Program {
    let t = Expr::InvocationId { axis: 0 };
    let name = DottedName {
        tok_types,
        haystack_len,
        head: Expr::var("manager_pos"),
        accumulator: "manager_end",
    };
    let mut body = vec![
        Node::let_bind("tok", load_u32(tok_types, t.clone())),
        Node::let_bind("with_pos", Expr::u32(INVALID_POS)),
        Node::let_bind("flags", Expr::u32(0)),
    ];
    body.extend(search_prev_token("prev_tok", t.clone(), tok_types));
    body.push(Node::if_then(
        Expr::and(
            Expr::eq(Expr::var("tok"), Expr::u32(TOK_WITH)),
            Expr::ne(
                load_u32(tok_types, Expr::var("prev_tok")),
                Expr::u32(TOK_ASYNC),
            ),
        ),
        vec![Node::assign("with_pos", t.clone())],
    ));
    body.extend(search_next_token(
        "async_next",
        Expr::add(t.clone(), Expr::u32(1)),
        tok_types,
        haystack_len,
    ));
    body.push(Node::if_then(
        Expr::and(
            Expr::eq(Expr::var("tok"), Expr::u32(TOK_ASYNC)),
            Expr::eq(
                load_u32(tok_types, Expr::var("async_next")),
                Expr::u32(TOK_WITH),
            ),
        ),
        vec![
            Node::assign("with_pos", Expr::var("async_next")),
            Node::assign("flags", Expr::u32(1)),
        ],
    ));
    body.extend(search_next_token(
        "manager_pos",
        Expr::add(Expr::var("with_pos"), Expr::u32(1)),
        tok_types,
        haystack_len,
    ));
    body.extend(search_next_token(
        "after_manager",
        Expr::add(Expr::var("manager_pos"), Expr::u32(1)),
        tok_types,
        haystack_len,
    ));
    let span = name.span(tok_starts, tok_lens);
    body.push(Node::if_then(
        Expr::and(
            Expr::ne(Expr::var("with_pos"), Expr::u32(INVALID_POS)),
            Expr::eq(
                load_u32(tok_types, Expr::var("manager_pos")),
                Expr::u32(TOK_IDENTIFIER),
            ),
        ),
        name.carriers()
            .into_iter()
            .chain([
                name.walk(),
                Node::let_bind("colon_pos", Expr::u32(INVALID_POS)),
                Node::loop_for(
                    "scan",
                    Expr::add(Expr::var("manager_end"), Expr::u32(1)),
                    Expr::u32(haystack_len),
                    vec![Node::if_then(
                        Expr::and(
                            Expr::eq(Expr::var("colon_pos"), Expr::u32(INVALID_POS)),
                            Expr::eq(load_u32(tok_types, Expr::var("scan")), Expr::u32(TOK_COLON)),
                        ),
                        vec![Node::assign("colon_pos", Expr::var("scan"))],
                    )],
                ),
                Node::let_bind(
                    "slot",
                    Expr::atomic_add(out_counts, Expr::u32(0), Expr::u32(WITH_RECORD_WORDS)),
                ),
            ])
            .chain(store_words(
                out_records,
                "slot",
                &[
                    span[0].clone(),
                    span[1].clone(),
                    Expr::var("with_pos"),
                    Expr::var("colon_pos"),
                    Expr::var("flags"),
                    Expr::u32(0),
                ],
            ))
            .collect(),
    ));

    let pass = line_index_pass(
        WITH_BLOCKS_OP_ID,
        tok_types,
        tok_starts,
        tok_lens,
        haystack_len,
    );
    let mut buffers = pass.token_buffers();
    buffers.extend(pass.record_buffers(out_records, out_counts, 3, WITH_RECORD_WORDS));
    pass.program(buffers, body)
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        STRUCTURE_OP_ID,
        || python312_extract_structure("tok_types", "tok_starts", "tok_lens", "out_records", "out_counts", 16),
        Some(structure_fixture_inputs),
        Some(structure_fixture_expected),
    )
    .with_category("parsing")
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        IMPORTS_OP_ID,
        || python312_extract_imports("tok_types", "tok_starts", "tok_lens", "out_records", "out_counts", 16),
        Some(import_fixture_inputs),
        Some(import_fixture_expected),
    )
    .with_category("parsing")
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        WITH_BLOCKS_OP_ID,
        || python312_extract_with_blocks("tok_types", "tok_starts", "tok_lens", "out_records", "out_counts", 16),
        Some(with_fixture_inputs),
        Some(with_fixture_expected),
    )
    .with_category("parsing")
}

fn structure_fixture_inputs() -> Vec<Vec<Vec<u8>>> {
    let (tok_types, tok_starts, tok_lens) = pack_sparse_tokens(
        &[
            (0, TOK_DEF, 3),
            (4, TOK_IDENTIFIER, 1),
            (5, TOK_LPAREN, 1),
            (6, TOK_RPAREN, 1),
            (7, TOK_COLON, 1),
        ],
        16,
    );
    vec![vec![
        tok_types,
        tok_starts,
        tok_lens,
        vec![0u8; 16 * DEF_RECORD_WORDS as usize * 4],
        vec![0u8; 4],
    ]]
}

fn structure_fixture_expected() -> Vec<Vec<Vec<u8>>> {
    let mut records = vec![0u8; 16 * DEF_RECORD_WORDS as usize * 4];
    write_words(&mut records, &[1, 4, 1, 5, 6, 7]);
    vec![vec![records, DEF_RECORD_WORDS.to_le_bytes().to_vec()]]
}

fn import_fixture_inputs() -> Vec<Vec<Vec<u8>>> {
    let (tok_types, tok_starts, tok_lens) =
        pack_sparse_tokens(&[(0, TOK_IMPORT, 6), (7, TOK_IDENTIFIER, 2)], 16);
    vec![vec![
        tok_types,
        tok_starts,
        tok_lens,
        vec![0u8; 16 * IMPORT_RECORD_WORDS as usize * 4],
        vec![0u8; 4],
    ]]
}

fn import_fixture_expected() -> Vec<Vec<Vec<u8>>> {
    let mut records = vec![0u8; 16 * IMPORT_RECORD_WORDS as usize * 4];
    write_words(
        &mut records,
        &[1, 7, 2, 0, 7, crate::parsing::python::INVALID_POS],
    );
    vec![vec![records, IMPORT_RECORD_WORDS.to_le_bytes().to_vec()]]
}

fn with_fixture_inputs() -> Vec<Vec<Vec<u8>>> {
    let (tok_types, tok_starts, tok_lens) = pack_sparse_tokens(
        &[
            (0, TOK_ASYNC, 5),
            (6, TOK_WITH, 4),
            (11, TOK_IDENTIFIER, 3),
            (14, TOK_COLON, 1),
        ],
        16,
    );
    vec![vec![
        tok_types,
        tok_starts,
        tok_lens,
        vec![0u8; 16 * WITH_RECORD_WORDS as usize * 4],
        vec![0u8; 4],
    ]]
}

fn with_fixture_expected() -> Vec<Vec<Vec<u8>>> {
    let mut records = vec![0u8; 16 * WITH_RECORD_WORDS as usize * 4];
    write_words(&mut records, &[11, 3, 6, 14, 1, 0]);
    vec![vec![records, WITH_RECORD_WORDS.to_le_bytes().to_vec()]]
}
