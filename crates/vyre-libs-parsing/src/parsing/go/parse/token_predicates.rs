//! Shared token predicates for Go structural extraction kernels.

use vyre_foundation::ir::{Expr, Node};
use vyre_spec::go_token::TOK_IDENTIFIER;

/// Test a token kind at `idx`.
pub(super) fn token_type_eq(tok_types: &str, idx: Expr, token: u32) -> Expr {
    Expr::eq(Expr::load(tok_types, idx), Expr::u32(token))
}

/// Load a token start offset.
pub(super) fn token_start(tok_starts: &str, idx: Expr) -> Expr {
    Expr::load(tok_starts, idx)
}

/// Load a token byte length.
pub(super) fn token_len(tok_lens: &str, idx: Expr) -> Expr {
    Expr::load(tok_lens, idx)
}

/// Compare a token's source bytes against a static byte string.
///
/// The length test and the per-byte tests are one `Expr`, and `Expr::and`
/// evaluates both operands, so the byte loads run even for a token whose
/// length already ruled the needle out. A token near the end of the source is
/// exactly that case: `token_start + needle.len()` reaches past the haystack
/// while the length conjunct is already false. The loads therefore go through
/// the canonical clamp, which keeps the access in bounds on every backend; the
/// clamped byte only ever reaches a conjunction the length test has already
/// falsified, so the result is unchanged.
pub(super) fn token_bytes_eq(
    haystack: &str,
    tok_starts: &str,
    tok_lens: &str,
    idx: Expr,
    needle: &[u8],
) -> Expr {
    let mut expr = Expr::eq(
        token_len(tok_lens, idx.clone()),
        Expr::u32(needle.len() as u32),
    );
    for (offset, byte) in needle.iter().enumerate() {
        expr = Expr::and(
            expr,
            Expr::eq(
                Expr::bitand(
                    vyre_primitives::ir_safe::clamped_load(
                        haystack,
                        Expr::add(
                            token_start(tok_starts, idx.clone()),
                            Expr::u32(offset as u32),
                        ),
                    ),
                    Expr::u32(0xFF),
                ),
                Expr::u32(u32::from(*byte)),
            ),
        );
    }
    expr
}

/// Test whether a token is an identifier.
pub(super) fn token_is_ident(tok_types: &str, idx: Expr) -> Expr {
    token_type_eq(tok_types, idx, TOK_IDENTIFIER)
}

/// Clamp a token-array index to the token count.
///
/// A predicate over a token at `t + k` is an `Expr`, and `Expr::and` evaluates
/// both operands, so writing the bound as a conjunct beside the predicate does
/// not stop the predicate's loads. The clamp keeps those loads inside the token
/// arrays; the caller supplies the value an out-of-range index must yield by
/// conjoining the same `idx < num_tokens` test, whose result the clamped
/// predicate cannot change.
pub(super) fn clamped_token_index(idx: Expr, num_tokens: Expr) -> Expr {
    Expr::select(
        Expr::lt(idx.clone(), num_tokens.clone()),
        idx,
        Expr::saturating_sub(num_tokens, Expr::u32(1)),
    )
}

/// Go keywords that can immediately precede a receive expression.
///
/// A `<-ch` in expression position follows one of these, an operator, or an
/// opening delimiter. It never follows a plain identifier, because
/// `identifier <- value` is a SEND. Distinguishing the two is what these
/// spellings are for: `return <-ch` must read as a receive, while `out <- 1`
/// must read as a send, and both are `IDENTIFIER ARROW ...` to the lexer,
/// which does not classify keywords.
///
/// One owner for the list so the send matcher (which must skip these) and the
/// receive matcher (which must allow them) cannot disagree about which
/// spellings are keywords.
pub(super) const RECEIVE_LEADING_KEYWORDS: &[&[u8]] =
    &[b"return", b"case", b"go", b"defer", b"range"];

/// Test whether a token is one of [`RECEIVE_LEADING_KEYWORDS`].
pub(super) fn token_is_receive_leading_keyword(
    haystack: &str,
    tok_types: &str,
    tok_starts: &str,
    tok_lens: &str,
    idx: Expr,
) -> Expr {
    RECEIVE_LEADING_KEYWORDS
        .iter()
        .map(|keyword| {
            token_is_keyword(
                haystack,
                tok_types,
                tok_starts,
                tok_lens,
                idx.clone(),
                keyword,
            )
        })
        .fold(Expr::bool(false), Expr::or)
}

/// Test whether a token is the `chan` type keyword.
///
/// `chan` is lexed as an ordinary identifier, so the channel TYPES `<-chan T`
/// and `chan<- T` are indistinguishable from the channel OPERATIONS `<-ch` and
/// `ch <- v` on token kinds alone. Both matchers consult this to tell a type
/// from an operation.
pub(super) fn token_is_chan_keyword(
    haystack: &str,
    tok_types: &str,
    tok_starts: &str,
    tok_lens: &str,
    idx: Expr,
) -> Expr {
    token_is_keyword(haystack, tok_types, tok_starts, tok_lens, idx, b"chan")
}

/// Test whether an identifier token matches a keyword spelling.
pub(super) fn token_is_keyword(
    haystack: &str,
    tok_types: &str,
    tok_starts: &str,
    tok_lens: &str,
    idx: Expr,
    keyword: &[u8],
) -> Expr {
    Expr::and(
        token_is_ident(tok_types, idx.clone()),
        token_bytes_eq(haystack, tok_starts, tok_lens, idx, keyword),
    )
}

/// Emit statements to claim an atomic slot and store a 2-word `(start, len)` record for `tok_idx`.
pub(super) fn emit_span_record_nodes(
    tok_starts: &str,
    tok_lens: &str,
    out_records: &str,
    out_counts: &str,
    slot_var: &str,
    tok_idx: Expr,
) -> Vec<Node> {
    vec![
        Node::let_bind(
            slot_var,
            Expr::atomic_add(
                out_counts,
                Expr::u32(0),
                Expr::u32(super::structure::GO_SPAN_RECORD_WORDS),
            ),
        ),
        Node::store(
            out_records,
            Expr::var(slot_var),
            token_start(tok_starts, tok_idx.clone()),
        ),
        Node::store(
            out_records,
            Expr::add(Expr::var(slot_var), Expr::u32(1)),
            token_len(tok_lens, tok_idx),
        ),
    ]
}

/// Test if token `t` matches `keyword` and token `t+1` satisfies `target_cond`, appending a span record if matched.
pub(super) fn emit_keyword_span_record_nodes(
    haystack: &str,
    tok_types: &str,
    tok_starts: &str,
    tok_lens: &str,
    t: Expr,
    num_tokens: Expr,
    keyword: &[u8],
    target_cond: Expr,
    out_records: &str,
    out_counts: &str,
    slot_var: &str,
) -> Node {
    let next_tok = Expr::add(t.clone(), Expr::u32(1));
    Node::if_then(
        Expr::lt(next_tok.clone(), num_tokens),
        vec![Node::if_then(
            token_is_keyword(haystack, tok_types, tok_starts, tok_lens, t, keyword),
            vec![Node::if_then(
                target_cond,
                emit_span_record_nodes(
                    tok_starts,
                    tok_lens,
                    out_records,
                    out_counts,
                    slot_var,
                    next_tok,
                ),
            )],
        )],
    )
}
