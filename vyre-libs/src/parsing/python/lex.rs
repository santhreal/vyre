use crate::parsing::composition::child_phase;
use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

// `vyre_spec::python_token` owns the numbering of these ids. They are the wire
// contract between the GPU lexer program below and every host matcher that
// reads its sparse token rows, so a caller that reads a token kind names that
// module rather than this one.
use vyre_spec::python_token::*;

fn load_byte(buffer: &str, index: Expr) -> Expr {
    crate::builder::state_machine::TableStateMachineComposer::masked_byte_load(buffer, index)
}

fn ascii(ch: u8) -> Expr {
    Expr::u32(ch as u32)
}

fn is_between(value: Expr, start: u8, end: u8) -> Expr {
    Expr::and(
        Expr::ge(value.clone(), ascii(start)),
        Expr::le(value, ascii(end)),
    )
}

fn is_alpha(value: Expr) -> Expr {
    Expr::or(
        is_between(value.clone(), b'a', b'z'),
        is_between(value, b'A', b'Z'),
    )
}

fn is_ident_continue(value: Expr) -> Expr {
    Expr::or(
        Expr::or(
            is_alpha(value.clone()),
            is_between(value.clone(), b'0', b'9'),
        ),
        Expr::eq(value, ascii(b'_')),
    )
}

fn is_ident_start(value: Expr) -> Expr {
    Expr::or(is_alpha(value.clone()), Expr::eq(value, ascii(b'_')))
}

fn keyword_match(haystack: &str, base: Expr, len_var: &str, word: &[u8]) -> Expr {
    let mut expr = Expr::eq(Expr::var(len_var), Expr::u32(word.len() as u32));
    for (offset, byte) in word.iter().enumerate() {
        expr = Expr::and(
            expr,
            Expr::eq(
                load_byte(haystack, Expr::add(base.clone(), Expr::u32(offset as u32))),
                ascii(*byte),
            ),
        );
    }
    expr
}

fn classify_keyword(haystack: &str, base: Expr) -> Vec<Node> {
    vec![
        Node::if_then(
            keyword_match(haystack, base.clone(), "token_len", b"def"),
            vec![Node::assign("token_type", Expr::u32(TOK_DEF))],
        ),
        Node::if_then(
            keyword_match(haystack, base.clone(), "token_len", b"async"),
            vec![Node::assign("token_type", Expr::u32(TOK_ASYNC))],
        ),
        Node::if_then(
            keyword_match(haystack, base.clone(), "token_len", b"class"),
            vec![Node::assign("token_type", Expr::u32(TOK_CLASS))],
        ),
        Node::if_then(
            keyword_match(haystack, base.clone(), "token_len", b"import"),
            vec![Node::assign("token_type", Expr::u32(TOK_IMPORT))],
        ),
        Node::if_then(
            keyword_match(haystack, base.clone(), "token_len", b"from"),
            vec![Node::assign("token_type", Expr::u32(TOK_FROM))],
        ),
        Node::if_then(
            keyword_match(haystack, base.clone(), "token_len", b"as"),
            vec![Node::assign("token_type", Expr::u32(TOK_AS))],
        ),
        Node::if_then(
            keyword_match(haystack, base.clone(), "token_len", b"with"),
            vec![Node::assign("token_type", Expr::u32(TOK_WITH))],
        ),
        Node::if_then(
            keyword_match(haystack, base.clone(), "token_len", b"await"),
            vec![Node::assign("token_type", Expr::u32(TOK_AWAIT))],
        ),
        Node::if_then(
            keyword_match(haystack, base.clone(), "token_len", b"match"),
            vec![Node::assign("token_type", Expr::u32(TOK_MATCH))],
        ),
        Node::if_then(
            keyword_match(haystack, base.clone(), "token_len", b"case"),
            vec![Node::assign("token_type", Expr::u32(TOK_CASE))],
        ),
        Node::if_then(
            keyword_match(haystack, base, "token_len", b"except"),
            vec![Node::assign("token_type", Expr::u32(TOK_EXCEPT))],
        ),
    ]
}

/// GPU Python 3.12 sparse lexer.
///
/// Each invocation owns one byte offset. Token starts write their
/// classification to the same index; all other offsets stay zero.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn python312_lexer(
    haystack: &str,
    out_tok_types: &str,
    out_tok_starts: &str,
    out_tok_lens: &str,
    out_counts: &str,
    haystack_len: u32,
) -> Program {
    let t = Expr::InvocationId { axis: 0 };
    let body = vec![
        Node::let_bind("ch", load_byte(haystack, t.clone())),
        Node::let_bind(
            "prev",
            Expr::select(
                Expr::gt(t.clone(), Expr::u32(0)),
                load_byte(haystack, Expr::sub(t.clone(), Expr::u32(1))),
                Expr::u32(0),
            ),
        ),
        Node::let_bind("comment_scan_active", Expr::u32(1)),
        Node::let_bind("in_comment_tail", Expr::u32(0)),
        Node::loop_for(
            "comment_rev",
            Expr::u32(0),
            t.clone(),
            vec![Node::if_then(
                Expr::eq(Expr::var("comment_scan_active"), Expr::u32(1)),
                vec![
                    Node::let_bind(
                        "comment_pos",
                        Expr::sub(Expr::sub(t.clone(), Expr::u32(1)), Expr::var("comment_rev")),
                    ),
                    Node::let_bind("comment_ch", load_byte(haystack, Expr::var("comment_pos"))),
                    Node::if_then(
                        Expr::eq(Expr::var("comment_ch"), ascii(b'\n')),
                        vec![Node::assign("comment_scan_active", Expr::u32(0))],
                    ),
                    Node::if_then(
                        Expr::eq(Expr::var("comment_ch"), ascii(b'#')),
                        vec![
                            Node::assign("in_comment_tail", Expr::u32(1)),
                            Node::assign("comment_scan_active", Expr::u32(0)),
                        ],
                    ),
                ],
            )],
        ),
        Node::let_bind("emit", Expr::u32(0)),
        Node::let_bind("token_type", Expr::u32(TOK_NONE)),
        Node::let_bind("token_len", Expr::u32(0)),
        // Store as u32(0|1) so later sites can `Expr::eq(_, Expr::u32(0))`
        // without the validator rejecting bool/u32 mismatches. The bool-
        // valued helpers `is_ident_start` / `is_ident_continue` return
        // genuine boolean exprs; coercing through `select` here keeps the
        // downstream call sites uniform with the surrounding u32 vars.
        Node::let_bind(
            "is_ident_start",
            Expr::select(is_ident_start(Expr::var("ch")), Expr::u32(1), Expr::u32(0)),
        ),
        Node::let_bind(
            "prev_identish",
            Expr::select(
                is_ident_continue(Expr::var("prev")),
                Expr::u32(1),
                Expr::u32(0),
            ),
        ),
        Node::if_then(
            Expr::eq(Expr::var("ch"), ascii(b'\n')),
            vec![
                Node::assign("emit", Expr::u32(1)),
                Node::assign("token_type", Expr::u32(TOK_NEWLINE)),
                Node::assign("token_len", Expr::u32(1)),
            ],
        ),
        Node::if_then(
            Expr::and(
                Expr::eq(Expr::var("emit"), Expr::u32(0)),
                Expr::eq(Expr::var("ch"), ascii(b'#')),
            ),
            vec![
                Node::let_bind("active", Expr::u32(1)),
                Node::let_bind("scan_len", Expr::u32(1)),
                Node::loop_for(
                    "j",
                    Expr::add(t.clone(), Expr::u32(1)),
                    Expr::u32(haystack_len),
                    vec![Node::if_then(
                        Expr::eq(Expr::var("active"), Expr::u32(1)),
                        vec![
                            Node::let_bind("cur", load_byte(haystack, Expr::var("j"))),
                            Node::if_then_else(
                                Expr::eq(Expr::var("cur"), ascii(b'\n')),
                                vec![Node::assign("active", Expr::u32(0))],
                                vec![Node::assign(
                                    "scan_len",
                                    Expr::add(Expr::var("scan_len"), Expr::u32(1)),
                                )],
                            ),
                        ],
                    )],
                ),
                Node::assign("emit", Expr::u32(1)),
                Node::assign("token_type", Expr::u32(TOK_COMMENT)),
                Node::assign("token_len", Expr::var("scan_len")),
            ],
        ),
        Node::if_then(
            Expr::and(
                Expr::eq(Expr::var("emit"), Expr::u32(0)),
                Expr::or(
                    Expr::eq(Expr::var("ch"), ascii(b'\'')),
                    Expr::eq(Expr::var("ch"), ascii(b'"')),
                ),
            ),
            vec![
                Node::let_bind("quote", Expr::var("ch")),
                Node::let_bind("active", Expr::u32(1)),
                Node::let_bind("escaped", Expr::u32(0)),
                Node::let_bind("scan_len", Expr::u32(1)),
                Node::loop_for(
                    "j",
                    Expr::add(t.clone(), Expr::u32(1)),
                    Expr::u32(haystack_len),
                    vec![Node::if_then(
                        Expr::eq(Expr::var("active"), Expr::u32(1)),
                        vec![
                            Node::let_bind("cur", load_byte(haystack, Expr::var("j"))),
                            Node::assign(
                                "scan_len",
                                Expr::add(Expr::var("scan_len"), Expr::u32(1)),
                            ),
                            Node::if_then_else(
                                Expr::eq(Expr::var("escaped"), Expr::u32(1)),
                                vec![Node::assign("escaped", Expr::u32(0))],
                                vec![
                                    Node::if_then(
                                        Expr::eq(Expr::var("cur"), ascii(b'\\')),
                                        vec![Node::assign("escaped", Expr::u32(1))],
                                    ),
                                    Node::if_then(
                                        Expr::eq(Expr::var("cur"), Expr::var("quote")),
                                        vec![Node::assign("active", Expr::u32(0))],
                                    ),
                                ],
                            ),
                        ],
                    )],
                ),
                Node::assign("emit", Expr::u32(1)),
                Node::assign("token_type", Expr::u32(TOK_STRING)),
                Node::assign("token_len", Expr::var("scan_len")),
            ],
        ),
        Node::if_then(
            Expr::and(
                Expr::eq(Expr::var("emit"), Expr::u32(0)),
                Expr::and(
                    Expr::eq(Expr::var("is_ident_start"), Expr::u32(1)),
                    Expr::eq(Expr::var("prev_identish"), Expr::u32(0)),
                ),
            ),
            vec![
                Node::let_bind("active", Expr::u32(1)),
                Node::let_bind("scan_len", Expr::u32(0)),
                Node::loop_for(
                    "j",
                    t.clone(),
                    Expr::u32(haystack_len),
                    vec![Node::if_then(
                        Expr::eq(Expr::var("active"), Expr::u32(1)),
                        vec![
                            Node::let_bind("cur", load_byte(haystack, Expr::var("j"))),
                            Node::if_then_else(
                                is_ident_continue(Expr::var("cur")),
                                vec![Node::assign(
                                    "scan_len",
                                    Expr::add(Expr::var("scan_len"), Expr::u32(1)),
                                )],
                                vec![Node::assign("active", Expr::u32(0))],
                            ),
                        ],
                    )],
                ),
                Node::assign("emit", Expr::u32(1)),
                Node::assign("token_type", Expr::u32(TOK_IDENTIFIER)),
                Node::assign("token_len", Expr::var("scan_len")),
            ]
            .into_iter()
            .chain(classify_keyword(haystack, t.clone()))
            .collect(),
        ),
        Node::if_then(
            Expr::and(
                Expr::eq(Expr::var("emit"), Expr::u32(0)),
                Expr::and(
                    is_between(Expr::var("ch"), b'0', b'9'),
                    Expr::eq(Expr::var("prev_identish"), Expr::u32(0)),
                ),
            ),
            vec![
                Node::let_bind("active", Expr::u32(1)),
                Node::let_bind("scan_len", Expr::u32(0)),
                Node::loop_for(
                    "j",
                    t.clone(),
                    Expr::u32(haystack_len),
                    vec![Node::if_then(
                        Expr::eq(Expr::var("active"), Expr::u32(1)),
                        vec![
                            Node::let_bind("cur", load_byte(haystack, Expr::var("j"))),
                            Node::if_then_else(
                                Expr::or(
                                    Expr::or(
                                        is_between(Expr::var("cur"), b'0', b'9'),
                                        Expr::eq(Expr::var("cur"), ascii(b'_')),
                                    ),
                                    Expr::eq(Expr::var("cur"), ascii(b'.')),
                                ),
                                vec![Node::assign(
                                    "scan_len",
                                    Expr::add(Expr::var("scan_len"), Expr::u32(1)),
                                )],
                                vec![Node::assign("active", Expr::u32(0))],
                            ),
                        ],
                    )],
                ),
                Node::assign("emit", Expr::u32(1)),
                Node::assign("token_type", Expr::u32(TOK_NUMBER)),
                Node::assign("token_len", Expr::var("scan_len")),
            ],
        ),
        Node::if_then(
            Expr::and(
                Expr::eq(Expr::var("emit"), Expr::u32(0)),
                Expr::eq(Expr::var("ch"), ascii(b'(')),
            ),
            vec![
                Node::assign("emit", Expr::u32(1)),
                Node::assign("token_type", Expr::u32(TOK_LPAREN)),
                Node::assign("token_len", Expr::u32(1)),
            ],
        ),
        Node::if_then(
            Expr::and(
                Expr::eq(Expr::var("emit"), Expr::u32(0)),
                Expr::eq(Expr::var("ch"), ascii(b')')),
            ),
            vec![
                Node::assign("emit", Expr::u32(1)),
                Node::assign("token_type", Expr::u32(TOK_RPAREN)),
                Node::assign("token_len", Expr::u32(1)),
            ],
        ),
        Node::if_then(
            Expr::and(
                Expr::eq(Expr::var("emit"), Expr::u32(0)),
                Expr::eq(Expr::var("ch"), ascii(b'[')),
            ),
            vec![
                Node::assign("emit", Expr::u32(1)),
                Node::assign("token_type", Expr::u32(TOK_LBRACKET)),
                Node::assign("token_len", Expr::u32(1)),
            ],
        ),
        Node::if_then(
            Expr::and(
                Expr::eq(Expr::var("emit"), Expr::u32(0)),
                Expr::eq(Expr::var("ch"), ascii(b']')),
            ),
            vec![
                Node::assign("emit", Expr::u32(1)),
                Node::assign("token_type", Expr::u32(TOK_RBRACKET)),
                Node::assign("token_len", Expr::u32(1)),
            ],
        ),
        Node::if_then(
            Expr::and(
                Expr::eq(Expr::var("emit"), Expr::u32(0)),
                Expr::eq(Expr::var("ch"), ascii(b'{')),
            ),
            vec![
                Node::assign("emit", Expr::u32(1)),
                Node::assign("token_type", Expr::u32(TOK_LBRACE)),
                Node::assign("token_len", Expr::u32(1)),
            ],
        ),
        Node::if_then(
            Expr::and(
                Expr::eq(Expr::var("emit"), Expr::u32(0)),
                Expr::eq(Expr::var("ch"), ascii(b'}')),
            ),
            vec![
                Node::assign("emit", Expr::u32(1)),
                Node::assign("token_type", Expr::u32(TOK_RBRACE)),
                Node::assign("token_len", Expr::u32(1)),
            ],
        ),
        Node::if_then(
            Expr::and(
                Expr::eq(Expr::var("emit"), Expr::u32(0)),
                Expr::eq(Expr::var("ch"), ascii(b':')),
            ),
            vec![
                Node::assign("emit", Expr::u32(1)),
                Node::assign("token_type", Expr::u32(TOK_COLON)),
                Node::assign("token_len", Expr::u32(1)),
            ],
        ),
        Node::if_then(
            Expr::and(
                Expr::eq(Expr::var("emit"), Expr::u32(0)),
                Expr::eq(Expr::var("ch"), ascii(b',')),
            ),
            vec![
                Node::assign("emit", Expr::u32(1)),
                Node::assign("token_type", Expr::u32(TOK_COMMA)),
                Node::assign("token_len", Expr::u32(1)),
            ],
        ),
        Node::if_then(
            Expr::and(
                Expr::eq(Expr::var("emit"), Expr::u32(0)),
                Expr::eq(Expr::var("ch"), ascii(b'.')),
            ),
            vec![
                Node::assign("emit", Expr::u32(1)),
                Node::assign("token_type", Expr::u32(TOK_DOT)),
                Node::assign("token_len", Expr::u32(1)),
            ],
        ),
        Node::if_then(
            Expr::and(
                Expr::eq(Expr::var("emit"), Expr::u32(0)),
                Expr::eq(Expr::var("ch"), ascii(b'=')),
            ),
            vec![
                Node::assign("emit", Expr::u32(1)),
                Node::assign("token_type", Expr::u32(TOK_EQ)),
                Node::assign("token_len", Expr::u32(1)),
            ],
        ),
        Node::if_then(
            Expr::and(
                Expr::eq(Expr::var("emit"), Expr::u32(0)),
                Expr::eq(Expr::var("ch"), ascii(b'@')),
            ),
            vec![
                Node::assign("emit", Expr::u32(1)),
                Node::assign("token_type", Expr::u32(TOK_AT)),
                Node::assign("token_len", Expr::u32(1)),
            ],
        ),
        Node::if_then(
            Expr::and(
                Expr::eq(Expr::var("emit"), Expr::u32(0)),
                Expr::eq(Expr::var("ch"), ascii(b'*')),
            ),
            vec![
                Node::assign("emit", Expr::u32(1)),
                Node::assign("token_type", Expr::u32(TOK_STAR)),
                Node::assign("token_len", Expr::u32(1)),
            ],
        ),
        Node::if_then(
            Expr::and(
                Expr::eq(Expr::var("in_comment_tail"), Expr::u32(1)),
                Expr::ne(Expr::var("ch"), ascii(b'\n')),
            ),
            vec![Node::assign("emit", Expr::u32(0))],
        ),
        Node::if_then(
            Expr::eq(Expr::var("emit"), Expr::u32(1)),
            vec![
                Node::store(out_tok_types, t.clone(), Expr::var("token_type")),
                Node::store(out_tok_starts, t.clone(), t.clone()),
                Node::store(out_tok_lens, t.clone(), Expr::var("token_len")),
                Node::let_bind(
                    "token_slot",
                    Expr::atomic_add(out_counts, Expr::u32(0), Expr::u32(1)),
                ),
                Node::assign("token_slot", Expr::var("token_slot")),
            ],
        ),
    ];

    Program::wrapped(
        vec![
            BufferDecl::storage(haystack, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(haystack_len),
            BufferDecl::storage(out_tok_types, 1, BufferAccess::ReadWrite, DataType::U32)
                .with_count(haystack_len),
            BufferDecl::storage(out_tok_starts, 2, BufferAccess::ReadWrite, DataType::U32)
                .with_count(haystack_len),
            BufferDecl::storage(out_tok_lens, 3, BufferAccess::ReadWrite, DataType::U32)
                .with_count(haystack_len),
            BufferDecl::storage(out_counts, 4, BufferAccess::ReadWrite, DataType::U32)
                .with_count(1),
        ],
        [256, 1, 1],
        vec![wrap_anonymous_region(
            "vyre-libs::parsing::python312_lexer",
            vec![child_phase(
                "vyre-libs::parsing::python312_lexer",
                crate::text::LINE_INDEX_OP_ID,
                vec![Node::if_then(
                    Expr::lt(t.clone(), Expr::u32(haystack_len)),
                    body,
                )],
            )],
        )],
    )
    .with_entry_op_id("vyre-libs::parsing::python312_lexer")
    .with_non_composable_with_self(true)
}

const EXPECTED_LEXER_TOK_TYPES_BYTES: [u8; 64] = [
    100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 10, 0, 0, 0, 1, 0, 0, 0, 11, 0,
    0, 0, 16, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0,
];
const EXPECTED_LEXER_TOK_STARTS_BYTES: [u8; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 6, 0, 0, 0, 7, 0, 0, 0,
    8, 0, 0, 0, 9, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0,
];
const EXPECTED_LEXER_TOK_LENS_BYTES: [u8; 64] = [
    3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
    1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];
const EXPECTED_LEXER_COUNTS_BYTES: [u8; 4] = [9, 0, 0, 0];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        "vyre-libs::parsing::python312_lexer",
        || python312_lexer("haystack", "tok_types", "tok_starts", "tok_lens", "counts", 16),
        Some(lexer_fixture_inputs),
        Some(|| vec![vec![
            EXPECTED_LEXER_TOK_TYPES_BYTES.to_vec(),
            EXPECTED_LEXER_TOK_STARTS_BYTES.to_vec(),
            EXPECTED_LEXER_TOK_LENS_BYTES.to_vec(),
            EXPECTED_LEXER_COUNTS_BYTES.to_vec(),
        ]]),
    )
    .with_category("parsing")
}

fn lexer_fixture_inputs() -> Vec<Vec<Vec<u8>>> {
    let source = b"def f(x):\n#z\n";
    let mut haystack = vec![0u8; 16 * 4];
    for (idx, byte) in source.iter().enumerate() {
        haystack[idx * 4..idx * 4 + 4].copy_from_slice(&u32::from(*byte).to_le_bytes());
    }
    vec![vec![
        haystack,
        vec![0u8; 16 * 4],
        vec![0u8; 16 * 4],
        vec![0u8; 16 * 4],
        vec![0u8; 4],
    ]]
}
