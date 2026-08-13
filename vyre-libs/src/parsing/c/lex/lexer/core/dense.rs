use super::helpers::{
    block_comment_scan, block_comment_start, char_start, classify_prologue, float_start,
    identifier_scan, identifier_start, integer_start, line_comment_scan, line_comment_start,
    number_scan, preproc_start, string_start, ClassifyCtx, ScanNames, SerialLexer,
};
use super::*;
use crate::parsing::c::lex::lexer::sections;

/// Full C11 serial lexer. This is the contiguous-haystack composition of the
/// shared classification walk: it selects every stage in `helpers`, adds the
/// three classifiers that only the full grammar has (directive-line splicing,
/// encoding-prefixed literals, and escape-validating literal bodies), and runs
/// them under the shared serial shell.
pub fn c11_lexer(
    haystack: &str,
    out_tok_types: &str,
    out_tok_starts: &str,
    out_tok_lens: &str,
    out_counts: &str,
    haystack_len: u32,
) -> Program {
    let ctx = ClassifyCtx::contiguous(haystack, haystack_len);
    let mut classify_at_pos = classify_prologue(&ctx, &Expr::var("pos"), false);

    classify_at_pos.push(preproc_start("line_allows_directive"));
    classify_at_pos.push(preproc_row_scan(&ctx));

    classify_at_pos.push(line_comment_start());
    classify_at_pos.push(line_comment_scan(
        &ctx,
        &ScanNames {
            done: "comment_done",
            scan: "scan_comment",
        },
    ));

    classify_at_pos.push(block_comment_start());
    classify_at_pos.push(block_comment_scan(
        &ctx,
        &ScanNames {
            done: "block_done",
            scan: "scan_block_comment",
        },
    ));

    classify_at_pos.push(prefixed_literal_start(b'"', TOK_STRING));
    classify_at_pos.push(prefixed_literal_start(b'\'', TOK_CHAR));

    classify_at_pos.push(identifier_start());
    classify_at_pos.push(identifier_scan(
        &ctx,
        &ScanNames {
            done: "ident_done",
            scan: "scan_ident",
        },
        false,
    ));

    classify_at_pos.push(integer_start());
    classify_at_pos.push(float_start());
    classify_at_pos.push(number_scan(
        &ctx,
        &ScanNames {
            done: "number_done",
            scan: "scan_number",
        },
        "number_is_float",
    ));

    classify_at_pos.push(string_start());
    classify_at_pos.push(char_start(false));
    classify_at_pos.push(escaped_literal_scan(&ctx, haystack_len));

    classify_at_pos.extend(sections::operator_punct_pushes());
    classify_at_pos.extend(sections::store_token_and_advance_pushes(
        haystack,
        haystack_len,
        out_tok_types,
        out_tok_starts,
        out_tok_lens,
    ));

    SerialLexer {
        op_id: "vyre-libs::parsing::c_lexer",
        haystack,
        out_tok_types,
        out_tok_starts,
        out_tok_lens,
        out_counts,
        haystack_len,
    }
    .build(classify_at_pos)
}

/// Extend a directive row to its unspliced line end. Unlike the sparse preproc
/// scan this honours backslash-newline splices, including the `\r\n` pair, so a
/// multi-line directive stays one token.
fn preproc_row_scan(ctx: &ClassifyCtx<'_>) -> Node {
    let start = Expr::add(Expr::var("pos"), Expr::u32(1));
    Node::if_then(
        Expr::eq(Expr::var("tok_type"), Expr::u32(TOK_PREPROC)),
        vec![
            Node::let_bind("preproc_done", Expr::u32(0)),
            Node::let_bind("preproc_spliced_cr", Expr::u32(0)),
            Node::loop_for(
                "scan_preproc",
                start.clone(),
                ctx.scan_bound(start, MAX_PREPROC_SCAN),
                vec![Node::if_then(
                    Expr::eq(Expr::var("preproc_done"), Expr::u32(0)),
                    vec![
                        Node::let_bind("scan_byte", ctx.byte_at(Expr::var("scan_preproc"))),
                        Node::let_bind(
                            "scan_prev",
                            Expr::select(
                                Expr::gt(Expr::var("scan_preproc"), Expr::var("pos")),
                                ctx.byte_at(Expr::saturating_sub(
                                    Expr::var("scan_preproc"),
                                    Expr::u32(1),
                                )),
                                Expr::u32(0),
                            ),
                        ),
                        Node::if_then_else(
                            Expr::or(
                                byte_eq(Expr::var("scan_byte"), b'\n'),
                                byte_eq(Expr::var("scan_byte"), b'\r'),
                            ),
                            vec![Node::if_then_else(
                                Expr::or(
                                    byte_eq(Expr::var("scan_prev"), b'\\'),
                                    Expr::and(
                                        byte_eq(Expr::var("scan_byte"), b'\n'),
                                        Expr::eq(Expr::var("preproc_spliced_cr"), Expr::u32(1)),
                                    ),
                                ),
                                vec![
                                    Node::assign(
                                        "tok_len",
                                        Expr::add(Expr::var("tok_len"), Expr::u32(1)),
                                    ),
                                    Node::assign(
                                        "preproc_spliced_cr",
                                        Expr::select(
                                            byte_eq(Expr::var("scan_byte"), b'\r'),
                                            Expr::u32(1),
                                            Expr::u32(0),
                                        ),
                                    ),
                                ],
                                vec![Node::assign("preproc_done", Expr::u32(1))],
                            )],
                            vec![Node::assign(
                                "tok_len",
                                Expr::add(Expr::var("tok_len"), Expr::u32(1)),
                            )],
                        ),
                    ],
                )],
            ),
        ],
    )
}

/// Open an encoding-prefixed literal: `L`, `u`, or `U` before `quote`, or the
/// three-byte `u8` prefix.
fn prefixed_literal_start(quote: u8, token: u32) -> Node {
    set_token(
        Expr::or(
            Expr::and(
                Expr::or(
                    byte_eq(Expr::var("byte"), b'L'),
                    Expr::or(
                        byte_eq(Expr::var("byte"), b'u'),
                        byte_eq(Expr::var("byte"), b'U'),
                    ),
                ),
                byte_eq(Expr::var("next_byte"), quote),
            ),
            Expr::and(
                Expr::and(
                    byte_eq(Expr::var("byte"), b'u'),
                    byte_eq(Expr::var("next_byte"), b'8'),
                ),
                byte_eq(Expr::var("next2_byte"), quote),
            ),
        ),
        token,
        Expr::select(
            Expr::and(
                byte_eq(Expr::var("byte"), b'u'),
                byte_eq(Expr::var("next_byte"), b'8'),
            ),
            Expr::u32(3),
            Expr::u32(2),
        ),
    )
}

/// Extend a string or character literal past its encoding prefix to the
/// matching quote, validating every escape sequence and reporting an
/// unterminated or invalid-escape literal as a diagnostic token.
fn escaped_literal_scan(ctx: &ClassifyCtx<'_>, haystack_len: u32) -> Node {
    let literal_start = Expr::add(
        Expr::add(Expr::var("pos"), Expr::var("literal_quote_offset")),
        Expr::u32(1),
    );
    Node::if_then(
        Expr::or(
            Expr::eq(Expr::var("tok_type"), Expr::u32(TOK_STRING)),
            Expr::eq(Expr::var("tok_type"), Expr::u32(TOK_CHAR)),
        ),
        vec![
            Node::let_bind(
                "literal_quote_offset",
                Expr::select(
                    Expr::or(
                        byte_eq(Expr::var("byte"), b'"'),
                        byte_eq(Expr::var("byte"), b'\''),
                    ),
                    Expr::u32(0),
                    Expr::select(
                        Expr::and(
                            byte_eq(Expr::var("byte"), b'u'),
                            byte_eq(Expr::var("next_byte"), b'8'),
                        ),
                        Expr::u32(2),
                        Expr::u32(1),
                    ),
                ),
            ),
            Node::let_bind(
                "quote",
                ctx.byte_at(Expr::add(
                    Expr::var("pos"),
                    Expr::var("literal_quote_offset"),
                )),
            ),
            Node::let_bind("literal_done", Expr::u32(0)),
            Node::let_bind("escaped", Expr::u32(0)),
            Node::let_bind("literal_unterminated", Expr::u32(0)),
            Node::let_bind("invalid_escape", Expr::u32(0)),
            Node::loop_for(
                "scan_literal",
                literal_start.clone(),
                ctx.scan_bound(literal_start, MAX_LITERAL_SCAN),
                vec![Node::if_then(
                    Expr::eq(Expr::var("literal_done"), Expr::u32(0)),
                    vec![
                        Node::assign("tok_len", Expr::add(Expr::var("tok_len"), Expr::u32(1))),
                        Node::let_bind("scan_byte", ctx.byte_at(Expr::var("scan_literal"))),
                        Node::if_then_else(
                            Expr::eq(Expr::var("escaped"), Expr::u32(1)),
                            vec![
                                Node::if_then(
                                    Expr::not(is_valid_escape_byte(
                                        ctx.haystack(),
                                        Expr::var("scan_literal"),
                                        Expr::var("scan_byte"),
                                        haystack_len,
                                    )),
                                    vec![Node::assign("invalid_escape", Expr::u32(1))],
                                ),
                                Node::assign("escaped", Expr::u32(0)),
                            ],
                            vec![Node::if_then_else(
                                byte_eq(Expr::var("scan_byte"), b'\\'),
                                vec![Node::assign("escaped", Expr::u32(1))],
                                vec![Node::if_then_else(
                                    Expr::eq(Expr::var("scan_byte"), Expr::var("quote")),
                                    vec![Node::assign("literal_done", Expr::u32(1))],
                                    vec![Node::if_then(
                                        Expr::or(
                                            byte_eq(Expr::var("scan_byte"), b'\n'),
                                            byte_eq(Expr::var("scan_byte"), b'\r'),
                                        ),
                                        vec![
                                            Node::assign("literal_unterminated", Expr::u32(1)),
                                            Node::assign("literal_done", Expr::u32(1)),
                                        ],
                                    )],
                                )],
                            )],
                        ),
                    ],
                )],
            ),
            Node::if_then(
                Expr::eq(Expr::var("literal_done"), Expr::u32(0)),
                vec![Node::assign("literal_unterminated", Expr::u32(1))],
            ),
            Node::if_then(
                Expr::eq(Expr::var("literal_unterminated"), Expr::u32(1)),
                vec![Node::assign(
                    "tok_type",
                    Expr::select(
                        Expr::eq(Expr::var("quote"), ascii(b'"')),
                        Expr::u32(TOK_ERR_UNTERMINATED_STRING),
                        Expr::u32(TOK_ERR_UNTERMINATED_CHAR),
                    ),
                )],
            ),
            Node::if_then(
                Expr::and(
                    Expr::eq(Expr::var("literal_unterminated"), Expr::u32(0)),
                    Expr::eq(Expr::var("invalid_escape"), Expr::u32(1)),
                ),
                vec![Node::assign("tok_type", Expr::u32(TOK_ERR_INVALID_ESCAPE))],
            ),
        ],
    )
}
