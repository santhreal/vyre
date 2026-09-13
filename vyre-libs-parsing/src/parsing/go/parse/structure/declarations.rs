//! Function, method, and interface declaration extraction.

use crate::parsing::go::parse::token_predicates::{token_is_ident, token_is_keyword, token_type_eq};
use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_libs_builder::builder::trip_count::clamped_by_extents;
use vyre_spec::go_token::{TOK_LPAREN, TOK_RPAREN};

use super::decl_span::GoDeclSpan;
use super::{GO_DECL_FUNC, GO_DECL_INTERFACE, GO_DECL_METHOD};

/// Extract function, method, and interface declarations.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn go_extract_declarations(
    tok_types: &str,
    tok_starts: &str,
    tok_lens: &str,
    haystack: &str,
    num_tokens: Expr,
    out_decls: &str,
    out_decl_counts: &str,
) -> Program {
    let num_tokens = clamped_by_extents(num_tokens, tok_types, [tok_starts, tok_lens]);
    let t = Expr::logical_index(0);
    let decl_span = GoDeclSpan {
        tok_types,
        tok_starts,
        tok_lens,
        out_decls,
        out_decl_counts,
        num_tokens: &num_tokens,
    };
    let body = vec![
        Node::if_then(
            // `t + 1` is the token that carries the name, or opens a method
            // receiver list. A `func` that is the last token has neither, so the
            // branch is admitted only when that token exists; without the bound
            // the probe below reads one past the token arrays. Both operands are
            // safe to evaluate for every lane, which the non-short-circuiting
            // `and` requires.
            Expr::and(
                Expr::lt(Expr::add(t.clone(), Expr::u32(1)), num_tokens.clone()),
                token_is_keyword(
                    haystack,
                    tok_types,
                    tok_starts,
                    tok_lens,
                    t.clone(),
                    b"func",
                ),
            ),
            vec![
                Node::let_bind("decl_kind", Expr::u32(GO_DECL_FUNC)),
                Node::let_bind("name_tok", Expr::add(t.clone(), Expr::u32(1))),
                Node::if_then(
                    token_type_eq(tok_types, Expr::var("name_tok"), TOK_LPAREN),
                    vec![
                        Node::assign("decl_kind", Expr::u32(GO_DECL_METHOD)),
                        Node::let_bind("recv_depth", Expr::u32(0)),
                        Node::let_bind("recv_done", Expr::u32(0)),
                        Node::let_bind("recv_end", Expr::var("name_tok")),
                        Node::loop_for(
                            "scan",
                            Expr::var("name_tok"),
                            num_tokens.clone(),
                            vec![Node::if_then(
                                Expr::eq(Expr::var("recv_done"), Expr::u32(0)),
                                vec![
                                    Node::if_then(
                                        token_type_eq(tok_types, Expr::var("scan"), TOK_LPAREN),
                                        vec![Node::assign(
                                            "recv_depth",
                                            Expr::add(Expr::var("recv_depth"), Expr::u32(1)),
                                        )],
                                    ),
                                    Node::if_then(
                                        token_type_eq(tok_types, Expr::var("scan"), TOK_RPAREN),
                                        vec![
                                            Node::assign(
                                                "recv_depth",
                                                Expr::sub(Expr::var("recv_depth"), Expr::u32(1)),
                                            ),
                                            Node::if_then(
                                                Expr::eq(Expr::var("recv_depth"), Expr::u32(0)),
                                                vec![
                                                    Node::assign("recv_done", Expr::u32(1)),
                                                    Node::assign("recv_end", Expr::var("scan")),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            )],
                        ),
                        Node::assign("name_tok", Expr::add(Expr::var("recv_end"), Expr::u32(1))),
                    ],
                ),
                // The receiver scan above can leave `name_tok` one past the last
                // token, when a method's receiver list closes on it. The bound is
                // a separate `if_then` rather than a conjunct because the
                // identifier test loads the token kind at that index.
                Node::if_then(
                    Expr::lt(Expr::var("name_tok"), num_tokens.clone()),
                    vec![Node::if_then(
                        token_is_ident(tok_types, Expr::var("name_tok")),
                        decl_span.nodes(
                            Expr::add(Expr::var("name_tok"), Expr::u32(1)),
                            Expr::var("decl_kind"),
                            Expr::var("name_tok"),
                        ),
                    )],
                ),
            ],
        ),
        Node::if_then(
            Expr::lt(Expr::add(t.clone(), Expr::u32(2)), num_tokens.clone()),
            vec![Node::if_then(
                Expr::and(
                    token_is_keyword(
                        haystack,
                        tok_types,
                        tok_starts,
                        tok_lens,
                        t.clone(),
                        b"type",
                    ),
                    Expr::and(
                        token_is_ident(tok_types, Expr::add(t.clone(), Expr::u32(1))),
                        token_is_keyword(
                            haystack,
                            tok_types,
                            tok_starts,
                            tok_lens,
                            Expr::add(t.clone(), Expr::u32(2)),
                            b"interface",
                        ),
                    ),
                ),
                decl_span.nodes(
                    Expr::add(t.clone(), Expr::u32(3)),
                    Expr::u32(GO_DECL_INTERFACE),
                    Expr::add(t.clone(), Expr::u32(1)),
                ),
            )],
        ),
    ];

    let mut buffers = super::super::token_stream_decls(tok_types, tok_starts, tok_lens, haystack);
    buffers.extend([
        BufferDecl::storage(out_decls, 4, BufferAccess::ReadWrite, DataType::U32),
        BufferDecl::storage(out_decl_counts, 5, BufferAccess::ReadWrite, DataType::U32).with_count(1),
    ]);

    Program::wrapped(
        buffers,
        [256, 1, 1],
        vec![wrap_anonymous_region(
            "vyre-libs::parsing::go_extract_declarations",
            vec![Node::if_then(Expr::lt(t, num_tokens), body)],
        )],
    )
    .with_entry_op_id("vyre-libs::parsing::go_extract_declarations")
    .with_non_composable_with_self(true)
}
