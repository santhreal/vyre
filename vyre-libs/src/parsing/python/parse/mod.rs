//! Python structural extractors.

/// Python call-site extractor.
pub mod calls;
/// Python decorator extractor.
pub mod decorators;
/// Python declaration/span extractor.
pub mod structure;
/// The one Python token-stream AST walk every extractor projects from.
mod walk;

use crate::parsing::python::INVALID_POS;
use vyre_foundation::ir::{Expr, Node};

pub(crate) fn store_words(buffer: &str, base_var: &str, words: &[Expr]) -> Vec<Node> {
    words
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            Node::store(
                buffer,
                Expr::add(Expr::var(base_var), Expr::u32(idx as u32)),
                value.clone(),
            )
        })
        .collect()
}

pub(crate) fn write_words(dst: &mut [u8], words: &[u32]) {
    for (idx, word) in words.iter().enumerate() {
        let base = idx * 4;
        dst[base..base + 4].copy_from_slice(&word.to_le_bytes());
    }
}

pub(crate) fn load_u32(buffer: &str, index: Expr) -> Expr {
    Expr::load(buffer, index)
}

/// The word at `pos` in a per-token buffer, or zero when `pos` names no token.
///
/// A scan reports a miss as `INVALID_POS`, and every caller reads that as "no
/// token": type zero, span zero. Indexing the buffer with the sentinel and
/// letting the interpreter's zero-fill supply the answer works on the reference
/// alone, since a backend that does not bounds-check reads whatever follows the
/// allocation. The index is folded into range before the load, so the access is
/// always inside the buffer, and the out-of-range answer is stated here instead
/// of being borrowed from out-of-bounds semantics.
pub(crate) fn token_word_at(buffer: &str, pos: Expr, token_capacity: u32) -> Expr {
    let in_range = Expr::lt(pos.clone(), Expr::u32(token_capacity));
    Expr::select(
        in_range.clone(),
        load_u32(buffer, Expr::select(in_range, pos, Expr::u32(0))),
        Expr::u32(0),
    )
}

pub(crate) fn search_next_token(
    out_var: &str,
    start_expr: Expr,
    tok_types: &str,
    haystack_len: u32,
) -> Vec<Node> {
    let scan = format!("{out_var}_scan");
    vec![
        Node::let_bind(out_var, Expr::u32(INVALID_POS)),
        Node::loop_for(
            scan.clone(),
            start_expr,
            Expr::u32(haystack_len),
            vec![Node::if_then(
                Expr::and(
                    Expr::eq(Expr::var(out_var), Expr::u32(INVALID_POS)),
                    Expr::ne(load_u32(tok_types, Expr::var(scan.clone())), Expr::u32(0)),
                ),
                vec![Node::assign(out_var, Expr::var(scan))],
            )],
        ),
    ]
}

pub(crate) fn search_prev_token(out_var: &str, start_expr: Expr, tok_types: &str) -> Vec<Node> {
    let rev = format!("{out_var}_rev");
    let cand = format!("{out_var}_cand");
    vec![
        Node::let_bind(out_var, Expr::u32(INVALID_POS)),
        Node::loop_for(
            rev.clone(),
            Expr::u32(0),
            start_expr.clone(),
            vec![
                Node::let_bind(
                    cand.clone(),
                    Expr::sub(Expr::sub(start_expr.clone(), Expr::u32(1)), Expr::var(rev)),
                ),
                Node::if_then(
                    Expr::and(
                        Expr::eq(Expr::var(out_var), Expr::u32(INVALID_POS)),
                        Expr::ne(load_u32(tok_types, Expr::var(cand.clone())), Expr::u32(0)),
                    ),
                    vec![Node::assign(out_var, Expr::var(cand))],
                ),
            ],
        ),
    ]
}

/// Same as [`search_next_token`] but skips the leading `Node::let_bind`
/// for `out_var`. The caller must declare `out_var` (typically with
/// `Expr::u32(INVALID_POS)`) in an enclosing scope so the binding
/// outlives the if/loop block this output is consumed inside.
pub(crate) fn search_next_token_into(
    out_var: &str,
    start_expr: Expr,
    tok_types: &str,
    haystack_len: u32,
) -> Vec<Node> {
    let scan = format!("{out_var}_scan");
    vec![Node::loop_for(
        scan.clone(),
        start_expr,
        Expr::u32(haystack_len),
        vec![Node::if_then(
            Expr::and(
                Expr::eq(Expr::var(out_var), Expr::u32(INVALID_POS)),
                Expr::ne(load_u32(tok_types, Expr::var(scan.clone())), Expr::u32(0)),
            ),
            vec![Node::assign(out_var, Expr::var(scan))],
        )],
    )]
}

pub(crate) fn find_matching_delimiter(
    out_var: &str,
    open_pos: Expr,
    tok_types: &str,
    haystack_len: u32,
    open_tok: u32,
    close_tok: u32,
) -> Vec<Node> {
    find_matching_delimiter_nodes(
        out_var,
        open_pos,
        tok_types,
        haystack_len,
        open_tok,
        close_tok,
        true,
    )
}

/// Same as [`find_matching_delimiter`] but skips the leading
/// `Node::let_bind` for `out_var`; caller pre-declares it in the
/// enclosing scope so the binding outlives the if/loop block this
/// output is consumed inside.
pub(crate) fn find_matching_delimiter_into(
    out_var: &str,
    open_pos: Expr,
    tok_types: &str,
    haystack_len: u32,
    open_tok: u32,
    close_tok: u32,
) -> Vec<Node> {
    find_matching_delimiter_nodes(
        out_var,
        open_pos,
        tok_types,
        haystack_len,
        open_tok,
        close_tok,
        false,
    )
}

fn find_matching_delimiter_nodes(
    out_var: &str,
    open_pos: Expr,
    tok_types: &str,
    haystack_len: u32,
    open_tok: u32,
    close_tok: u32,
    declare_out: bool,
) -> Vec<Node> {
    let depth = format!("{out_var}_depth");
    let scan = format!("{out_var}_scan");
    let tok = format!("{out_var}_tok");
    let mut nodes = Vec::with_capacity(3);
    if declare_out {
        nodes.push(Node::let_bind(out_var, Expr::u32(INVALID_POS)));
    }
    nodes.push(Node::let_bind(depth.clone(), Expr::u32(0)));
    // A missing open position is the sentinel, and `INVALID_POS + 1` wraps to 0,
    // which would scan the whole token stream and report the first unmatched
    // close token as the match. Starting at the end makes the range empty.
    let scan_start = Expr::select(
        Expr::eq(open_pos.clone(), Expr::u32(INVALID_POS)),
        Expr::u32(haystack_len),
        Expr::add(open_pos, Expr::u32(1)),
    );
    nodes.push(Node::loop_for(
        scan.clone(),
        scan_start,
        Expr::u32(haystack_len),
        vec![
            Node::let_bind(tok.clone(), load_u32(tok_types, Expr::var(scan.clone()))),
            Node::if_then(
                Expr::eq(Expr::var(out_var), Expr::u32(INVALID_POS)),
                vec![
                    Node::if_then(
                        Expr::eq(Expr::var(tok.clone()), Expr::u32(open_tok)),
                        vec![Node::assign(
                            depth.clone(),
                            Expr::add(Expr::var(depth.clone()), Expr::u32(1)),
                        )],
                    ),
                    Node::if_then(
                        Expr::eq(Expr::var(tok), Expr::u32(close_tok)),
                        vec![Node::if_then_else(
                            Expr::eq(Expr::var(depth.clone()), Expr::u32(0)),
                            vec![Node::assign(out_var, Expr::var(scan))],
                            vec![Node::assign(
                                depth.clone(),
                                Expr::sub(Expr::var(depth), Expr::u32(1)),
                            )],
                        )],
                    ),
                ],
            ),
        ],
    ));
    nodes
}
