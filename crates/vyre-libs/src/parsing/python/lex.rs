//! The Python lexer program.
//!
//! Token numbering is owned by `vyre_spec::python_token`, which is the wire
//! contract between this program and every host matcher that reads its rows.

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

/// A source byte at `index`, or zero past the end of the source.
///
/// The keyword matcher and the lookbehind read positions a comparison in the
/// same expression later discards, and both `Expr::and` and `Expr::select`
/// evaluate every operand. The index is folded into range before the load, so a
/// read that will be discarded is still inside the buffer.
const T: Expr = Expr::LogicalIndex { axis: 0 };

const KEYWORDS: &[(&[u8], u32)] = &[
    (b"def", TOK_DEF),
    (b"async", TOK_ASYNC),
    (b"class", TOK_CLASS),
    (b"import", TOK_IMPORT),
    (b"from", TOK_FROM),
    (b"as", TOK_AS),
    (b"with", TOK_WITH),
    (b"await", TOK_AWAIT),
    (b"match", TOK_MATCH),
    (b"case", TOK_CASE),
    (b"except", TOK_EXCEPT),
];

const PUNCTUATION: &[(u8, u32)] = &[
    (b'(', TOK_LPAREN),
    (b')', TOK_RPAREN),
    (b'[', TOK_LBRACKET),
    (b']', TOK_RBRACKET),
    (b'{', TOK_LBRACE),
    (b'}', TOK_RBRACE),
    (b':', TOK_COLON),
    (b',', TOK_COMMA),
    (b'.', TOK_DOT),
    (b'=', TOK_EQ),
    (b'@', TOK_AT),
    (b'*', TOK_STAR),
];

fn load_byte_bounded(buffer: &str, make_index: impl Fn() -> Expr, haystack_len: u32) -> Expr {
    let in_range = || Expr::lt(make_index(), Expr::u32(haystack_len));
    Expr::select(
        in_range(),
        load_byte(buffer, Expr::select(in_range(), make_index(), Expr::u32(0))),
        Expr::u32(0),
    )
}

fn ascii(ch: u8) -> Expr {
    Expr::u32(ch as u32)
}

fn is_between(var: &str, start: u8, end: u8) -> Expr {
    Expr::and(
        Expr::ge(Expr::var(var), ascii(start)),
        Expr::le(Expr::var(var), ascii(end)),
    )
}

fn is_alpha(var: &str) -> Expr {
    Expr::or(is_between(var, b'a', b'z'), is_between(var, b'A', b'Z'))
}

fn is_ident_continue(var: &str) -> Expr {
    Expr::or(
        Expr::or(is_alpha(var), is_between(var, b'0', b'9')),
        Expr::eq(Expr::var(var), ascii(b'_')),
    )
}

fn is_ident_start(var: &str) -> Expr {
    Expr::or(is_alpha(var), Expr::eq(Expr::var(var), ascii(b'_')))
}

fn keyword_match(haystack: &str, len_var: &str, word: &[u8], haystack_len: u32) -> Expr {
    let mut expr = Expr::eq(Expr::var(len_var), Expr::u32(word.len() as u32));
    for (offset, byte) in word.iter().enumerate() {
        let off = offset as u32;
        expr = Expr::and(
            expr,
            Expr::eq(
                load_byte_bounded(haystack, || Expr::add(T, Expr::u32(off)), haystack_len),
                ascii(*byte),
            ),
        );
    }
    expr
}

fn classify_keyword(haystack: &str, haystack_len: u32) -> impl Iterator<Item = Node> + '_ {
    KEYWORDS.iter().map(move |(word, tok)| {
        Node::if_then(
            keyword_match(haystack, "token_len", word, haystack_len),
            vec![Node::assign("token_type", Expr::u32(*tok))],
        )
    })
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
    let mut body = vec![
        Node::let_bind("ch", load_byte(haystack, T)),
        Node::let_bind(
            "prev",
            // `t - 1` wraps to `u32::MAX` at the first byte, and the select
            // evaluates its taken and untaken arm alike, so the index is folded
            // into range before the load.
            Expr::select(
                Expr::gt(T, Expr::u32(0)),
                load_byte_bounded(
                    haystack,
                    || {
                        Expr::select(
                            Expr::gt(T, Expr::u32(0)),
                            Expr::sub(T, Expr::u32(1)),
                            Expr::u32(0),
                        )
                    },
                    haystack_len,
                ),
                Expr::u32(0),
            ),
        ),
        Node::let_bind("comment_scan_active", Expr::u32(1)),
        Node::let_bind("in_comment_tail", Expr::u32(0)),
        Node::loop_for(
            "comment_rev",
            Expr::u32(0),
            T,
            vec![Node::if_then(
                Expr::eq(Expr::var("comment_scan_active"), Expr::u32(1)),
                vec![
                    Node::let_bind(
                        "comment_pos",
                        Expr::sub(Expr::sub(T, Expr::u32(1)), Expr::var("comment_rev")),
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
            Expr::select(is_ident_start("ch"), Expr::u32(1), Expr::u32(0)),
        ),
        Node::let_bind(
            "prev_identish",
            Expr::select(is_ident_continue("prev"), Expr::u32(1), Expr::u32(0)),
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
                    Expr::add(T, Expr::u32(1)),
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
                    Expr::add(T, Expr::u32(1)),
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
                    T,
                    Expr::u32(haystack_len),
                    vec![Node::if_then(
                        Expr::eq(Expr::var("active"), Expr::u32(1)),
                        vec![
                            Node::let_bind("cur", load_byte(haystack, Expr::var("j"))),
                            Node::if_then_else(
                                is_ident_continue("cur"),
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
            .chain(classify_keyword(haystack, haystack_len))
            .collect(),
        ),
        Node::if_then(
            Expr::and(
                Expr::eq(Expr::var("emit"), Expr::u32(0)),
                Expr::and(
                    is_between("ch", b'0', b'9'),
                    Expr::eq(Expr::var("prev_identish"), Expr::u32(0)),
                ),
            ),
            vec![
                Node::let_bind("active", Expr::u32(1)),
                Node::let_bind("scan_len", Expr::u32(0)),
                Node::loop_for(
                    "j",
                    T,
                    Expr::u32(haystack_len),
                    vec![Node::if_then(
                        Expr::eq(Expr::var("active"), Expr::u32(1)),
                        vec![
                            Node::let_bind("cur", load_byte(haystack, Expr::var("j"))),
                            Node::if_then_else(
                                Expr::or(
                                    Expr::or(
                                        is_between("cur", b'0', b'9'),
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
    ];
    for &(ch, tok) in PUNCTUATION {
        body.push(Node::if_then(
            Expr::and(
                Expr::eq(Expr::var("emit"), Expr::u32(0)),
                Expr::eq(Expr::var("ch"), ascii(ch)),
            ),
            vec![
                Node::assign("emit", Expr::u32(1)),
                Node::assign("token_type", Expr::u32(tok)),
                Node::assign("token_len", Expr::u32(1)),
            ],
        ));
    }
    body.extend([
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
                Node::store(out_tok_types, T, Expr::var("token_type")),
                Node::store(out_tok_starts, T, T),
                Node::store(out_tok_lens, T, Expr::var("token_len")),
                Node::let_bind(
                    "token_slot",
                    Expr::atomic_add(out_counts, Expr::u32(0), Expr::u32(1)),
                ),
                Node::assign("token_slot", Expr::var("token_slot")),
            ],
        ),
    ]);

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
                vec![Node::if_then(Expr::lt(T, Expr::u32(haystack_len)), body)],
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
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
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
    .with_uncharacterized()
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
