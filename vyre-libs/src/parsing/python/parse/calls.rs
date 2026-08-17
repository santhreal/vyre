use super::walk::{pack_sparse_tokens, DottedName, TokenPass};
use super::{find_matching_delimiter, load_u32, search_next_token, search_prev_token, store_words};
use crate::parsing::python::{CALL_RECORD_WORDS, INVALID_POS, KWARG_RECORD_WORDS};
use vyre_foundation::ir::{Expr, Node, Program};
use vyre_spec::python_token::{
    TOK_AWAIT, TOK_DOT, TOK_EQ, TOK_IDENTIFIER, TOK_LBRACKET, TOK_LPAREN, TOK_NUMBER, TOK_RBRACKET,
    TOK_RPAREN,
};

const OP_ID: &str = "vyre-libs::parsing::python312_extract_calls";

/// Extract Python call sites plus top-level keyword arguments.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn python312_extract_calls(
    tok_types: &str,
    tok_starts: &str,
    tok_lens: &str,
    out_calls: &str,
    out_call_counts: &str,
    out_kwargs: &str,
    out_kw_counts: &str,
    haystack_len: u32,
) -> Program {
    let t = Expr::InvocationId { axis: 0 };
    let name = DottedName {
        tok_types,
        haystack_len,
        head: t.clone(),
        accumulator: "name_end",
    };
    // The dotted-name carriers are hoisted to the outer body so they outlive
    // the if_then block that assigns them and remain in scope for the
    // post-if_then `search_next_token` call that reads `name_end`.
    let mut body = vec![
        Node::let_bind("tok", load_u32(tok_types, t.clone())),
        Node::let_bind("emit", Expr::u32(0)),
        Node::let_bind("is_call_head", Expr::u32(0)),
    ];
    body.extend(name.carriers());
    body.extend(search_prev_token("prev_tok", t.clone(), tok_types));
    body.push(Node::if_then(
        Expr::and(
            Expr::eq(Expr::var("tok"), Expr::u32(TOK_IDENTIFIER)),
            Expr::ne(
                load_u32(tok_types, Expr::var("prev_tok")),
                Expr::u32(TOK_DOT),
            ),
        ),
        vec![Node::assign("is_call_head", Expr::u32(1)), name.walk()],
    ));
    body.extend(search_next_token(
        "after_name",
        Expr::add(Expr::var("name_end"), Expr::u32(1)),
        tok_types,
        haystack_len,
    ));
    body.extend(find_matching_delimiter(
        "rparen",
        Expr::var("after_name"),
        tok_types,
        haystack_len,
        TOK_LPAREN,
        TOK_RPAREN,
    ));
    body.push(Node::if_then(
        Expr::and(
            Expr::eq(Expr::var("is_call_head"), Expr::u32(1)),
            Expr::and(
                Expr::eq(
                    load_u32(tok_types, Expr::var("after_name")),
                    Expr::u32(TOK_LPAREN),
                ),
                Expr::ne(Expr::var("rparen"), Expr::u32(INVALID_POS)),
            ),
        ),
        vec![Node::assign("emit", Expr::u32(1))],
    ));
    let span = name.span(tok_starts, tok_lens);
    body.push(Node::if_then(
        Expr::eq(Expr::var("emit"), Expr::u32(1)),
        vec![
            Node::let_bind("kw_base", Expr::load(out_kw_counts, Expr::u32(0))),
            Node::let_bind("kw_count", Expr::u32(0)),
            Node::let_bind("paren_depth", Expr::u32(0)),
            Node::let_bind("bracket_depth", Expr::u32(0)),
            Node::loop_for(
                "scan",
                Expr::add(Expr::var("after_name"), Expr::u32(1)),
                Expr::var("rparen"),
                vec![
                    Node::let_bind("scan_tok", load_u32(tok_types, Expr::var("scan"))),
                    Node::if_then(
                        Expr::eq(Expr::var("scan_tok"), Expr::u32(TOK_LPAREN)),
                        vec![Node::assign(
                            "paren_depth",
                            Expr::add(Expr::var("paren_depth"), Expr::u32(1)),
                        )],
                    ),
                    Node::if_then(
                        Expr::eq(Expr::var("scan_tok"), Expr::u32(TOK_RPAREN)),
                        vec![Node::if_then(
                            Expr::gt(Expr::var("paren_depth"), Expr::u32(0)),
                            vec![Node::assign(
                                "paren_depth",
                                Expr::sub(Expr::var("paren_depth"), Expr::u32(1)),
                            )],
                        )],
                    ),
                    Node::if_then(
                        Expr::eq(Expr::var("scan_tok"), Expr::u32(TOK_LBRACKET)),
                        vec![Node::assign(
                            "bracket_depth",
                            Expr::add(Expr::var("bracket_depth"), Expr::u32(1)),
                        )],
                    ),
                    Node::if_then(
                        Expr::eq(Expr::var("scan_tok"), Expr::u32(TOK_RBRACKET)),
                        vec![Node::if_then(
                            Expr::gt(Expr::var("bracket_depth"), Expr::u32(0)),
                            vec![Node::assign(
                                "bracket_depth",
                                Expr::sub(Expr::var("bracket_depth"), Expr::u32(1)),
                            )],
                        )],
                    ),
                    Node::if_then(
                        Expr::and(
                            Expr::and(
                                Expr::eq(Expr::var("scan_tok"), Expr::u32(TOK_IDENTIFIER)),
                                Expr::eq(Expr::var("paren_depth"), Expr::u32(0)),
                            ),
                            Expr::eq(Expr::var("bracket_depth"), Expr::u32(0)),
                        ),
                        // Drop the explicit `Node::let_bind` siblings  -
                        // `search_next_token` / `search_prev_token` each
                        // emit their own outer let_bind, so the manual
                        // ones here were duplicate-sibling V032 errors.
                        search_next_token(
                            "kw_eq_pos",
                            Expr::add(Expr::var("scan"), Expr::u32(1)),
                            tok_types,
                            haystack_len,
                        )
                        .into_iter()
                        .chain(search_prev_token("kw_prev", Expr::var("scan"), tok_types))
                        .chain(vec![Node::if_then(
                            Expr::and(
                                Expr::eq(
                                    load_u32(tok_types, Expr::var("kw_eq_pos")),
                                    Expr::u32(TOK_EQ),
                                ),
                                Expr::ne(
                                    load_u32(tok_types, Expr::var("kw_prev")),
                                    Expr::u32(TOK_DOT),
                                ),
                            ),
                            vec![
                                Node::let_bind(
                                    "kw_slot",
                                    Expr::atomic_add(
                                        out_kw_counts,
                                        Expr::u32(0),
                                        Expr::u32(KWARG_RECORD_WORDS),
                                    ),
                                ),
                                Node::store(
                                    out_kwargs,
                                    Expr::var("kw_slot"),
                                    load_u32(tok_starts, Expr::var("scan")),
                                ),
                                Node::store(
                                    out_kwargs,
                                    Expr::add(Expr::var("kw_slot"), Expr::u32(1)),
                                    load_u32(tok_lens, Expr::var("scan")),
                                ),
                                Node::assign(
                                    "kw_count",
                                    Expr::add(Expr::var("kw_count"), Expr::u32(1)),
                                ),
                            ],
                        )])
                        .collect(),
                    ),
                ],
            ),
            Node::let_bind(
                "call_slot",
                Expr::atomic_add(out_call_counts, Expr::u32(0), Expr::u32(CALL_RECORD_WORDS)),
            ),
        ]
        .into_iter()
        .chain(store_words(
            out_calls,
            "call_slot",
            &[
                span[0].clone(),
                span[1].clone(),
                Expr::var("after_name"),
                Expr::var("rparen"),
                Expr::var("kw_base"),
                Expr::var("kw_count"),
                Expr::select(
                    Expr::eq(
                        load_u32(tok_types, Expr::var("prev_tok")),
                        Expr::u32(TOK_AWAIT),
                    ),
                    Expr::u32(1),
                    Expr::u32(0),
                ),
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
    buffers.extend(pass.record_buffers(out_calls, out_call_counts, 3, CALL_RECORD_WORDS));
    buffers.extend(pass.record_buffers(out_kwargs, out_kw_counts, 5, KWARG_RECORD_WORDS));
    pass.program(buffers, body)
}

const EXPECTED_CALLS_RECORDS_BYTES: [u8; 448] = [
    6, 0, 0, 0, 3, 0, 0, 0, 9, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0,
];
const EXPECTED_CALL_COUNTS_BYTES: [u8; 4] = [7, 0, 0, 0];
const EXPECTED_KWARGS_RECORDS_BYTES: [u8; 128] = [
    10, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0,
];
const EXPECTED_KW_COUNTS_BYTES: [u8; 4] = [2, 0, 0, 0];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || python312_extract_calls(
            "tok_types", "tok_starts", "tok_lens", "out_calls", "out_call_counts", "out_kwargs", "out_kw_counts", 16
        ),
        Some(call_fixture_inputs),
        Some(|| vec![vec![
            EXPECTED_CALLS_RECORDS_BYTES.to_vec(),
            EXPECTED_CALL_COUNTS_BYTES.to_vec(),
            EXPECTED_KWARGS_RECORDS_BYTES.to_vec(),
            EXPECTED_KW_COUNTS_BYTES.to_vec(),
        ]]),
    )
    .with_category("parsing")
}

fn call_fixture_inputs() -> Vec<Vec<Vec<u8>>> {
    let (tok_types, tok_starts, tok_lens) = pack_sparse_tokens(
        &[
            (0, TOK_AWAIT, 5),
            (6, TOK_IDENTIFIER, 3),
            (9, TOK_LPAREN, 1),
            (10, TOK_IDENTIFIER, 1),
            (11, TOK_EQ, 1),
            (12, TOK_NUMBER, 1),
            (13, TOK_RPAREN, 1),
        ],
        16,
    );

    vec![vec![
        tok_types,
        tok_starts,
        tok_lens,
        vec![0u8; 16 * CALL_RECORD_WORDS as usize * 4],
        vec![0u8; 4],
        vec![0u8; 16 * KWARG_RECORD_WORDS as usize * 4],
        vec![0u8; 4],
    ]]
}
