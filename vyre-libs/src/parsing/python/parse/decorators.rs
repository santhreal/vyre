use super::walk::{pack_sparse_tokens, DottedName, TokenPass};
use super::{
    find_matching_delimiter, load_u32, search_next_token, search_next_token_into, store_words,
    write_words,
};
use crate::parsing::python::lex::{
    TOK_ASYNC, TOK_AT, TOK_CLASS, TOK_DEF, TOK_IDENTIFIER, TOK_LPAREN, TOK_RPAREN,
};
use crate::parsing::python::{DECORATOR_RECORD_WORDS, INVALID_POS};
use vyre_foundation::ir::{Expr, Node, Program};

const OP_ID: &str = "vyre-libs::parsing::python312_extract_decorators";

/// Extract decorator occurrences and their immediate target.
#[must_use]
pub fn python312_extract_decorators(
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
        head: Expr::var("decorator_name"),
        accumulator: "decorator_end",
    };
    let mut body = Vec::new();
    body.extend(search_next_token(
        "decorator_name",
        Expr::add(t.clone(), Expr::u32(1)),
        tok_types,
        haystack_len,
    ));
    body.push(Node::let_bind("tok", load_u32(tok_types, t.clone())));
    // Hoist every cross-block name to the outer body so it survives
    // the if_then / loop_for scopes that assign it. Each helper we
    // call inside an if_then uses the `_into` (assign-only) variant
    // so the outer let_bind isn't redeclared (V008/V032 noise).
    body.extend(name.carriers());
    body.push(Node::let_bind("after_decorator", Expr::u32(INVALID_POS)));
    body.push(Node::let_bind("target_tok", Expr::u32(INVALID_POS)));
    body.push(Node::let_bind("target_name", Expr::u32(INVALID_POS)));
    body.push(Node::let_bind("target_kind", Expr::u32(0)));
    body.push(Node::let_bind("async_def", Expr::u32(INVALID_POS)));
    body.extend(find_matching_delimiter(
        "decorator_rparen",
        Expr::var("decorator_name"),
        tok_types,
        haystack_len,
        TOK_LPAREN,
        TOK_RPAREN,
    ));
    let span = name.span(tok_starts, tok_lens);
    body.push(Node::if_then(
        Expr::and(
            Expr::eq(Expr::var("tok"), Expr::u32(TOK_AT)),
            Expr::eq(
                load_u32(tok_types, Expr::var("decorator_name")),
                Expr::u32(TOK_IDENTIFIER),
            ),
        ),
        vec![name.walk()]
            .into_iter()
            .chain(search_next_token_into(
                "after_decorator",
                Expr::add(Expr::var("decorator_end"), Expr::u32(1)),
                tok_types,
                haystack_len,
            ))
            .chain(vec![Node::if_then_else(
                Expr::eq(
                    load_u32(tok_types, Expr::var("after_decorator")),
                    Expr::u32(TOK_LPAREN),
                ),
                search_next_token_into(
                    "target_tok",
                    Expr::add(Expr::var("decorator_rparen"), Expr::u32(1)),
                    tok_types,
                    haystack_len,
                ),
                search_next_token_into(
                    "target_tok",
                    Expr::add(Expr::var("decorator_end"), Expr::u32(1)),
                    tok_types,
                    haystack_len,
                ),
            )])
            .chain(vec![
                Node::if_then(
                    Expr::eq(
                        load_u32(tok_types, Expr::var("target_tok")),
                        Expr::u32(TOK_DEF),
                    ),
                    vec![
                        Node::assign("target_kind", Expr::u32(1)),
                        Node::assign("target_name", Expr::u32(INVALID_POS)),
                    ]
                    .into_iter()
                    .chain(search_next_token_into(
                        "target_name",
                        Expr::add(Expr::var("target_tok"), Expr::u32(1)),
                        tok_types,
                        haystack_len,
                    ))
                    .collect(),
                ),
                Node::if_then(
                    Expr::eq(
                        load_u32(tok_types, Expr::var("target_tok")),
                        Expr::u32(TOK_CLASS),
                    ),
                    vec![
                        Node::assign("target_kind", Expr::u32(3)),
                        Node::assign("target_name", Expr::u32(INVALID_POS)),
                    ]
                    .into_iter()
                    .chain(search_next_token_into(
                        "target_name",
                        Expr::add(Expr::var("target_tok"), Expr::u32(1)),
                        tok_types,
                        haystack_len,
                    ))
                    .collect(),
                ),
                Node::if_then(
                    Expr::eq(
                        load_u32(tok_types, Expr::var("target_tok")),
                        Expr::u32(TOK_ASYNC),
                    ),
                    vec![
                        Node::assign("target_kind", Expr::u32(2)),
                        Node::assign("target_name", Expr::u32(INVALID_POS)),
                    ]
                    .into_iter()
                    .chain(search_next_token_into(
                        "async_def",
                        Expr::add(Expr::var("target_tok"), Expr::u32(1)),
                        tok_types,
                        haystack_len,
                    ))
                    .chain(search_next_token_into(
                        "target_name",
                        Expr::add(Expr::var("async_def"), Expr::u32(1)),
                        tok_types,
                        haystack_len,
                    ))
                    .collect(),
                ),
                Node::let_bind(
                    "slot",
                    Expr::atomic_add(out_counts, Expr::u32(0), Expr::u32(DECORATOR_RECORD_WORDS)),
                ),
            ])
            .chain(store_words(
                out_records,
                "slot",
                &[
                    span[0].clone(),
                    span[1].clone(),
                    Expr::var("target_kind"),
                    load_u32(tok_starts, Expr::var("target_name")),
                    load_u32(tok_lens, Expr::var("target_name")),
                    Expr::var("target_tok"),
                ],
            ))
            .collect(),
    ));

    let pass = TokenPass {
        op_id: OP_ID,
        child_op_id: crate::parsing::core_delimiter_match::OP_ID,
        tok_types,
        tok_starts,
        tok_lens,
        haystack_len,
    };
    let mut buffers = pass.token_buffers();
    buffers.extend(pass.record_buffers(out_records, out_counts, 3, DECORATOR_RECORD_WORDS));
    pass.program(buffers, body)
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || python312_extract_decorators("tok_types", "tok_starts", "tok_lens", "out_records", "out_counts", 16),
        Some(decorator_fixture_inputs),
        Some(decorator_fixture_expected),
    )
    .with_category("parsing")
}

fn decorator_fixture_inputs() -> Vec<Vec<Vec<u8>>> {
    let (tok_types, tok_starts, tok_lens) = pack_sparse_tokens(
        &[
            (0, TOK_AT, 1),
            (1, TOK_IDENTIFIER, 1),
            (3, TOK_ASYNC, 5),
            (9, TOK_DEF, 3),
            (13, TOK_IDENTIFIER, 1),
        ],
        16,
    );

    vec![vec![
        tok_types,
        tok_starts,
        tok_lens,
        vec![0u8; 16 * DECORATOR_RECORD_WORDS as usize * 4],
        vec![0u8; 4],
    ]]
}

fn decorator_fixture_expected() -> Vec<Vec<Vec<u8>>> {
    let mut records = vec![0u8; 16 * DECORATOR_RECORD_WORDS as usize * 4];
    write_words(&mut records, &[1, 1, 2, 13, 1, 3]);

    vec![vec![records, DECORATOR_RECORD_WORDS.to_le_bytes().to_vec()]]
}
