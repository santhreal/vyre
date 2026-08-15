use crate::parsing::go::lex::{TOK_ARROW, TOK_IDENTIFIER, TOK_LPAREN};
use crate::parsing::go::parse::structure::GO_SPAN_RECORD_WORDS;
use crate::parsing::go::parse::token_predicates::{
    token_is_chan_keyword, token_is_keyword, token_is_receive_leading_keyword, token_len,
    token_start, token_type_eq,
};
use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

macro_rules! define_go_keyword_call_extractor {
    ($name:ident, $keyword:literal, $op_id:literal, $doc:literal) => {
        #[doc = $doc]
        #[must_use]
        #[allow(clippy::too_many_arguments)]
        pub fn $name(
            tok_types: &str,
            tok_starts: &str,
            tok_lens: &str,
            haystack: &str,
            num_tokens: Expr,
            out_calls: &str,
            out_counts: &str,
        ) -> Program {
            go_extract_keyword_calls(
                tok_types, tok_starts, tok_lens, haystack, num_tokens, out_calls, out_counts,
                $keyword, $op_id,
            )
        }
    };
}

define_go_keyword_call_extractor!(
    go_extract_goroutine_calls,
    b"go",
    "vyre-libs::parsing::go_extract_goroutine_calls",
    "Extract goroutine launches (`go f(...)`) as callee spans."
);

define_go_keyword_call_extractor!(
    go_extract_defer_calls,
    b"defer",
    "vyre-libs::parsing::go_extract_defer_calls",
    "Extract Go `defer f(...)` calls as callee spans."
);

#[allow(clippy::too_many_arguments)]
fn go_extract_keyword_calls(
    tok_types: &str,
    tok_starts: &str,
    tok_lens: &str,
    haystack: &str,
    num_tokens: Expr,
    out_calls: &str,
    out_counts: &str,
    keyword: &[u8],
    op_id: &str,
) -> Program {
    let t = Expr::gid_x();
    let body = vec![Node::if_then(
        Expr::lt(Expr::add(t.clone(), Expr::u32(1)), num_tokens.clone()),
        vec![Node::if_then(
            token_is_keyword(
                haystack,
                tok_types,
                tok_starts,
                tok_lens,
                t.clone(),
                keyword,
            ),
            vec![Node::if_then(
                token_type_eq(
                    tok_types,
                    Expr::add(t.clone(), Expr::u32(1)),
                    TOK_IDENTIFIER,
                ),
                vec![
                    Node::let_bind(
                        "call_idx",
                        Expr::atomic_add(out_counts, Expr::u32(0), Expr::u32(GO_SPAN_RECORD_WORDS)),
                    ),
                    Node::store(
                        out_calls,
                        Expr::var("call_idx"),
                        token_start(tok_starts, Expr::add(t.clone(), Expr::u32(1))),
                    ),
                    Node::store(
                        out_calls,
                        Expr::add(Expr::var("call_idx"), Expr::u32(1)),
                        token_len(tok_lens, Expr::add(t.clone(), Expr::u32(1))),
                    ),
                ],
            )],
        )],
    )];

    Program::wrapped(
        vec![
            BufferDecl::storage(tok_types, 0, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage(tok_starts, 1, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage(tok_lens, 2, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage(haystack, 3, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage(out_calls, 4, BufferAccess::ReadWrite, DataType::U32),
            BufferDecl::storage(out_counts, 5, BufferAccess::ReadWrite, DataType::U32)
                .with_count(1),
        ],
        [256, 1, 1],
        vec![wrap_anonymous_region(
            op_id,
            vec![Node::if_then(Expr::lt(t, num_tokens), body)],
        )],
    )
    .with_entry_op_id(op_id)
    .with_non_composable_with_self(true)
}

/// Extract channel sends (`ch <- value`) as channel operand spans.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn go_extract_channel_sends(
    tok_types: &str,
    tok_starts: &str,
    tok_lens: &str,
    haystack: &str,
    num_tokens: Expr,
    out_ops: &str,
    out_counts: &str,
) -> Program {
    let t = Expr::gid_x();
    let next = Expr::add(t.clone(), Expr::u32(1));
    let after = Expr::add(t.clone(), Expr::u32(2));
    let body = vec![Node::if_then(
        Expr::lt(Expr::add(t.clone(), Expr::u32(1)), num_tokens.clone()),
        vec![Node::if_then(
            Expr::and(
                Expr::and(
                    token_type_eq(tok_types, t.clone(), TOK_IDENTIFIER),
                    token_type_eq(tok_types, next.clone(), TOK_ARROW),
                ),
                Expr::and(
                    // `chan<- T` is a send-only channel TYPE, not a send.
                    Expr::not(token_is_chan_keyword(
                        haystack,
                        tok_types,
                        tok_starts,
                        tok_lens,
                        t.clone(),
                    )),
                    Expr::and(
                        // `in <-chan T` is a receive-only channel TYPE in a
                        // parameter list. The arrow binds to `chan`, not to
                        // `in`, so the identifier before it is not a channel
                        // being sent to.
                        Expr::not(token_is_chan_keyword(
                            haystack,
                            tok_types,
                            tok_starts,
                            tok_lens,
                            after.clone(),
                        )),
                        // `return <-ch` and friends are receives. Without this
                        // the keyword in front of the arrow reads as the
                        // channel operand and the receive is miscounted as a
                        // send.
                        Expr::not(token_is_receive_leading_keyword(
                            haystack,
                            tok_types,
                            tok_starts,
                            tok_lens,
                            t.clone(),
                        )),
                    ),
                ),
            ),
            vec![
                Node::let_bind(
                    "send_idx",
                    Expr::atomic_add(out_counts, Expr::u32(0), Expr::u32(GO_SPAN_RECORD_WORDS)),
                ),
                Node::store(
                    out_ops,
                    Expr::var("send_idx"),
                    token_start(tok_starts, t.clone()),
                ),
                Node::store(
                    out_ops,
                    Expr::add(Expr::var("send_idx"), Expr::u32(1)),
                    token_len(tok_lens, t.clone()),
                ),
            ],
        )],
    )];

    Program::wrapped(
        vec![
            BufferDecl::storage(tok_types, 0, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage(tok_starts, 1, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage(tok_lens, 2, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage(haystack, 3, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage(out_ops, 4, BufferAccess::ReadWrite, DataType::U32),
            BufferDecl::storage(out_counts, 5, BufferAccess::ReadWrite, DataType::U32)
                .with_count(1),
        ],
        [256, 1, 1],
        vec![wrap_anonymous_region(
            "vyre-libs::parsing::go_extract_channel_sends",
            vec![Node::if_then(Expr::lt(t, num_tokens), body)],
        )],
    )
    .with_entry_op_id("vyre-libs::parsing::go_extract_channel_sends")
    .with_non_composable_with_self(true)
}

/// Extract channel receives (`<-ch`) as channel operand spans.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn go_extract_channel_receives(
    tok_types: &str,
    tok_starts: &str,
    tok_lens: &str,
    haystack: &str,
    num_tokens: Expr,
    out_ops: &str,
    out_counts: &str,
) -> Program {
    let t = Expr::gid_x();
    let next = Expr::add(t.clone(), Expr::u32(1));
    // Saturating: at t == 0 this stays 0 and the guard below is written so the
    // token at 0 can never be its own predecessor.
    let previous = Expr::select(
        Expr::gt(t.clone(), Expr::u32(0)),
        Expr::sub(t.clone(), Expr::u32(1)),
        Expr::u32(0),
    );
    // A receive's operand may not be preceded by a plain identifier: that
    // shape is `ch <- v`, a SEND. It MAY be preceded by a keyword such as
    // `return`, because `return <-ch` is a receive in expression position.
    let preceded_by_operand = Expr::and(
        Expr::gt(t.clone(), Expr::u32(0)),
        Expr::and(
            token_type_eq(tok_types, previous.clone(), TOK_IDENTIFIER),
            Expr::not(token_is_receive_leading_keyword(
                haystack, tok_types, tok_starts, tok_lens, previous,
            )),
        ),
    );
    let body = vec![Node::if_then(
        Expr::lt(Expr::add(t.clone(), Expr::u32(1)), num_tokens.clone()),
        vec![Node::if_then(
            Expr::and(
                Expr::and(
                    token_type_eq(tok_types, t.clone(), TOK_ARROW),
                    token_type_eq(tok_types, next.clone(), TOK_IDENTIFIER),
                ),
                Expr::and(
                    // `<-chan T` is a receive-only channel TYPE. The token
                    // after the arrow is the `chan` keyword, not a channel
                    // being received from.
                    Expr::not(token_is_chan_keyword(
                        haystack,
                        tok_types,
                        tok_starts,
                        tok_lens,
                        next.clone(),
                    )),
                    Expr::not(preceded_by_operand),
                ),
            ),
            vec![
                Node::let_bind(
                    "recv_idx",
                    Expr::atomic_add(out_counts, Expr::u32(0), Expr::u32(GO_SPAN_RECORD_WORDS)),
                ),
                Node::store(
                    out_ops,
                    Expr::var("recv_idx"),
                    token_start(tok_starts, Expr::add(t.clone(), Expr::u32(1))),
                ),
                Node::store(
                    out_ops,
                    Expr::add(Expr::var("recv_idx"), Expr::u32(1)),
                    token_len(tok_lens, Expr::add(t.clone(), Expr::u32(1))),
                ),
            ],
        )],
    )];

    Program::wrapped(
        vec![
            BufferDecl::storage(tok_types, 0, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage(tok_starts, 1, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage(tok_lens, 2, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage(haystack, 3, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage(out_ops, 4, BufferAccess::ReadWrite, DataType::U32),
            BufferDecl::storage(out_counts, 5, BufferAccess::ReadWrite, DataType::U32)
                .with_count(1),
        ],
        [256, 1, 1],
        vec![wrap_anonymous_region(
            "vyre-libs::parsing::go_extract_channel_receives",
            vec![Node::if_then(Expr::lt(t, num_tokens), body)],
        )],
    )
    .with_entry_op_id("vyre-libs::parsing::go_extract_channel_receives")
    .with_non_composable_with_self(true)
}

/// Extract `make(chan T, ...)` constructions as the `make` call span.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn go_extract_channel_creations(
    tok_types: &str,
    tok_starts: &str,
    tok_lens: &str,
    haystack: &str,
    num_tokens: Expr,
    out_ops: &str,
    out_counts: &str,
) -> Program {
    let t = Expr::gid_x();
    let body = vec![Node::if_then(
        Expr::lt(Expr::add(t.clone(), Expr::u32(2)), num_tokens.clone()),
        vec![Node::if_then(
            Expr::and(
                token_is_keyword(
                    haystack,
                    tok_types,
                    tok_starts,
                    tok_lens,
                    t.clone(),
                    b"make",
                ),
                Expr::and(
                    token_type_eq(tok_types, Expr::add(t.clone(), Expr::u32(1)), TOK_LPAREN),
                    token_is_keyword(
                        haystack,
                        tok_types,
                        tok_starts,
                        tok_lens,
                        Expr::add(t.clone(), Expr::u32(2)),
                        b"chan",
                    ),
                ),
            ),
            vec![
                Node::let_bind(
                    "create_idx",
                    Expr::atomic_add(out_counts, Expr::u32(0), Expr::u32(GO_SPAN_RECORD_WORDS)),
                ),
                Node::store(
                    out_ops,
                    Expr::var("create_idx"),
                    token_start(tok_starts, t.clone()),
                ),
                Node::store(
                    out_ops,
                    Expr::add(Expr::var("create_idx"), Expr::u32(1)),
                    token_len(tok_lens, t.clone()),
                ),
            ],
        )],
    )];

    Program::wrapped(
        vec![
            BufferDecl::storage(tok_types, 0, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage(tok_starts, 1, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage(tok_lens, 2, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage(haystack, 3, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage(out_ops, 4, BufferAccess::ReadWrite, DataType::U32),
            BufferDecl::storage(out_counts, 5, BufferAccess::ReadWrite, DataType::U32)
                .with_count(1),
        ],
        [256, 1, 1],
        vec![wrap_anonymous_region(
            "vyre-libs::parsing::go_extract_channel_creations",
            vec![Node::if_then(Expr::lt(t, num_tokens), body)],
        )],
    )
    .with_entry_op_id("vyre-libs::parsing::go_extract_channel_creations")
    .with_non_composable_with_self(true)
}
