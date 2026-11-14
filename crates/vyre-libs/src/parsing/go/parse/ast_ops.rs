//! Go keyword-call extractors, generated from one macro.
//!
//! Every extractor differs only in the keyword it anchors on, so the span
//! recording is written once.

use crate::parsing::go::parse::token_predicates::{
    clamped_token_index, emit_keyword_span_record_nodes, emit_span_record_nodes,
    token_is_chan_keyword, token_is_keyword, token_is_receive_leading_keyword, token_type_eq,
};
use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_spec::go_token::{TOK_ARROW, TOK_IDENTIFIER, TOK_LPAREN};

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

const T: Expr = Expr::LogicalIndex { axis: 0 };

#[allow(clippy::too_many_arguments)]
fn go_extract_span_program(
    tok_types: &str,
    tok_starts: &str,
    tok_lens: &str,
    haystack: &str,
    num_tokens: Expr,
    out_records: &str,
    out_counts: &str,
    op_id: &str,
    body: Vec<Node>,
) -> Program {
    let mut buffers = super::token_stream_decls(tok_types, tok_starts, tok_lens, haystack);
    buffers.extend([
        BufferDecl::storage(out_records, 4, BufferAccess::ReadWrite, DataType::U32),
        BufferDecl::storage(out_counts, 5, BufferAccess::ReadWrite, DataType::U32).with_count(1),
    ]);
    Program::wrapped(
        buffers,
        [256, 1, 1],
        vec![wrap_anonymous_region(
            op_id,
            vec![Node::if_then(Expr::lt(T, num_tokens), body)],
        )],
    )
    .with_entry_op_id(op_id)
    .with_non_composable_with_self(true)
}

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
    let body = vec![emit_keyword_span_record_nodes(
        haystack,
        tok_types,
        tok_starts,
        tok_lens,
        T,
        num_tokens.clone(),
        keyword,
        token_type_eq(tok_types, Expr::add(T, Expr::u32(1)), TOK_IDENTIFIER),
        out_calls,
        out_counts,
        "call_idx",
    )];
    go_extract_span_program(
        tok_types, tok_starts, tok_lens, haystack, num_tokens, out_calls, out_counts, op_id, body,
    )
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
    let next = || Expr::add(T, Expr::u32(1));
    let after = || Expr::add(T, Expr::u32(2));
    let body = vec![Node::if_then(
        Expr::lt(next(), num_tokens.clone()),
        vec![Node::if_then(
            Expr::and(
                Expr::and(
                    token_type_eq(tok_types, T, TOK_IDENTIFIER),
                    token_type_eq(tok_types, next(), TOK_ARROW),
                ),
                Expr::and(
                    // `chan<- T` is a send-only channel TYPE, not a send.
                    Expr::not(token_is_chan_keyword(
                        haystack, tok_types, tok_starts, tok_lens, T,
                    )),
                    Expr::and(
                        // `in <-chan T` is a receive-only channel TYPE in a
                        // parameter list. The arrow binds to `chan`, not to
                        // `in`, so the identifier before it is not a channel
                        // being sent to.
                        //
                        // This probe reaches `t + 2`, one further than the
                        // guard above admits, so it carries its own bound. The
                        // bound is a conjunct and the index is clamped: a
                        // conjunct alone leaves the loads beside it running.
                        Expr::not(Expr::and(
                            Expr::lt(after(), num_tokens.clone()),
                            token_is_chan_keyword(
                                haystack,
                                tok_types,
                                tok_starts,
                                tok_lens,
                                clamped_token_index(after(), num_tokens.clone()),
                            ),
                        )),
                        // `return <-ch` and friends are receives. Without this
                        // the keyword in front of the arrow reads as the
                        // channel operand and the receive is miscounted as a
                        // send.
                        Expr::not(token_is_receive_leading_keyword(
                            haystack, tok_types, tok_starts, tok_lens, T,
                        )),
                    ),
                ),
            ),
            emit_span_record_nodes(tok_starts, tok_lens, out_ops, out_counts, "send_idx", T),
        )],
    )];

    go_extract_span_program(
        tok_types,
        tok_starts,
        tok_lens,
        haystack,
        num_tokens,
        out_ops,
        out_counts,
        "vyre-libs::parsing::go_extract_channel_sends",
        body,
    )
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
    let next = || Expr::add(T, Expr::u32(1));
    // Saturating: at t == 0 this stays 0 and the guard below is written so the
    // token at 0 can never be its own predecessor.
    let previous = || {
        Expr::select(
            Expr::gt(T, Expr::u32(0)),
            Expr::sub(T, Expr::u32(1)),
            Expr::u32(0),
        )
    };
    // A receive's operand may not be preceded by a plain identifier: that
    // shape is `ch <- v`, a SEND. It MAY be preceded by a keyword such as
    // `return`, because `return <-ch` is a receive in expression position.
    let preceded_by_operand = Expr::and(
        Expr::gt(T, Expr::u32(0)),
        Expr::and(
            token_type_eq(tok_types, previous(), TOK_IDENTIFIER),
            Expr::not(token_is_receive_leading_keyword(
                haystack,
                tok_types,
                tok_starts,
                tok_lens,
                previous(),
            )),
        ),
    );
    let body = vec![Node::if_then(
        Expr::lt(next(), num_tokens.clone()),
        vec![Node::if_then(
            Expr::and(
                Expr::and(
                    token_type_eq(tok_types, T, TOK_ARROW),
                    token_type_eq(tok_types, next(), TOK_IDENTIFIER),
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
                        next(),
                    )),
                    Expr::not(preceded_by_operand),
                ),
            ),
            emit_span_record_nodes(
                tok_starts,
                tok_lens,
                out_ops,
                out_counts,
                "recv_idx",
                next(),
            ),
        )],
    )];

    go_extract_span_program(
        tok_types,
        tok_starts,
        tok_lens,
        haystack,
        num_tokens,
        out_ops,
        out_counts,
        "vyre-libs::parsing::go_extract_channel_receives",
        body,
    )
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
    let body = vec![Node::if_then(
        Expr::lt(Expr::add(T, Expr::u32(2)), num_tokens.clone()),
        vec![Node::if_then(
            Expr::and(
                token_is_keyword(haystack, tok_types, tok_starts, tok_lens, T, b"make"),
                Expr::and(
                    token_type_eq(tok_types, Expr::add(T, Expr::u32(1)), TOK_LPAREN),
                    token_is_keyword(
                        haystack,
                        tok_types,
                        tok_starts,
                        tok_lens,
                        Expr::add(T, Expr::u32(2)),
                        b"chan",
                    ),
                ),
            ),
            emit_span_record_nodes(tok_starts, tok_lens, out_ops, out_counts, "create_idx", T),
        )],
    )];
    go_extract_span_program(
        tok_types,
        tok_starts,
        tok_lens,
        haystack,
        num_tokens,
        out_ops,
        out_counts,
        "vyre-libs::parsing::go_extract_channel_creations",
        body,
    )
}
