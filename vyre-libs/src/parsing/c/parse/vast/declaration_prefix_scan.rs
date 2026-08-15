//! The declaration prefix of a VAST row, walked once.
//!
//! Every path that asks "is this identifier a declarator" first has to read the
//! declaration prefix in front of it: the run of specifier tokens back to the
//! nearest reset token. The walk has one subtlety that decides correctness. It
//! runs from the row backwards and skips balanced paren and brace groups whole,
//! so the `int` inside a cast `(int) a;` is not read as `a`'s type and the body
//! of `struct S { int a; } b;` is not read as `b`'s.
//!
//! Two builders spelled that walk independently, and only one of them skipped
//! the groups: the precomputed-context declaration classifier walked the prefix
//! forwards with no depth tracking, so it called `a` in `(int) a;` an ordinary
//! declarator where the self-contained classifier did not. The walk lives here
//! now and both call it.

#![allow(missing_docs)]

use crate::parsing::c::lex::tokens::{TOK_LBRACE, TOK_LPAREN, TOK_RBRACE, TOK_RPAREN};
use vyre_foundation::ir::{Expr, Node};

use super::build::{vast_row_base_expr, vast_row_kind_from_base_expr};

/// One reverse walk over a row's declaration prefix.
pub(super) struct DeclarationPrefixScan<'a> {
    /// VAST node buffer.
    pub(super) vast_nodes: &'a str,
    /// Row whose prefix is read. The walk covers `prefix_start .. idx`.
    pub(super) idx: Expr,
    /// First row of the prefix: the reset boundary, precomputed or `0`.
    pub(super) prefix_start: Expr,
    /// Binding-name prefix. The walk owns `{prefix}_prefix_*`.
    pub(super) prefix: &'a str,
}

impl DeclarationPrefixScan<'_> {
    /// Name the walk binds the current prefix row's index under.
    pub(super) fn idx_var(&self) -> String {
        format!("{}_prefix_idx", self.prefix)
    }

    /// Name the walk binds the current prefix row's token kind under.
    pub(super) fn kind_var(&self) -> String {
        format!("{}_prefix_kind", self.prefix)
    }

    /// Name of the flag that stops the walk. The caller sets it on a reset token.
    pub(super) fn done_var(&self) -> String {
        format!("{}_prefix_done", self.prefix)
    }
}

/// Emit the reverse prefix walk.
///
/// `row_body` runs for each prefix row outside a skipped paren or brace group,
/// with [`DeclarationPrefixScan::idx_var`] and [`DeclarationPrefixScan::kind_var`]
/// in scope. Setting [`DeclarationPrefixScan::done_var`] ends the walk, which is
/// how a caller stops at a reset token.
pub(super) fn emit_declaration_prefix_back_scan(
    scan: &DeclarationPrefixScan<'_>,
    row_body: Vec<Node>,
) -> Vec<Node> {
    let prefix = scan.prefix;
    let done = scan.done_var();
    let prefix_idx = scan.idx_var();
    let prefix_kind = scan.kind_var();
    let prefix_base = format!("{prefix}_prefix_base");
    let prefix_scan = format!("{prefix}_prefix_scan");
    let paren_depth = format!("{prefix}_prefix_skipped_paren_depth");
    let brace_depth = format!("{prefix}_prefix_skipped_brace_depth");
    let in_paren = format!("{prefix}_prefix_in_skipped_paren");
    let in_brace = format!("{prefix}_prefix_in_skipped_brace");

    vec![
        Node::let_bind(&done, Expr::u32(0)),
        Node::let_bind(&paren_depth, Expr::u32(0)),
        Node::let_bind(&brace_depth, Expr::u32(0)),
        Node::loop_for(
            &prefix_scan,
            scan.prefix_start.clone(),
            scan.idx.clone(),
            vec![Node::if_then(
                Expr::eq(Expr::var(&done), Expr::u32(0)),
                vec![
                    Node::let_bind(
                        &prefix_idx,
                        Expr::sub(
                            Expr::sub(scan.idx.clone(), Expr::u32(1)),
                            Expr::sub(Expr::var(&prefix_scan), scan.prefix_start.clone()),
                        ),
                    ),
                    Node::let_bind(&prefix_base, vast_row_base_expr(Expr::var(&prefix_idx))),
                    Node::let_bind(
                        &prefix_kind,
                        vast_row_kind_from_base_expr(scan.vast_nodes, Expr::var(&prefix_base)),
                    ),
                    // A closing delimiter is itself inside the group it closes, so
                    // the flags are read before the depths move.
                    Node::let_bind(
                        &in_paren,
                        Expr::or(
                            Expr::gt(Expr::var(&paren_depth), Expr::u32(0)),
                            Expr::eq(Expr::var(&prefix_kind), Expr::u32(TOK_RPAREN)),
                        ),
                    ),
                    Node::let_bind(
                        &in_brace,
                        Expr::or(
                            Expr::gt(Expr::var(&brace_depth), Expr::u32(0)),
                            Expr::eq(Expr::var(&prefix_kind), Expr::u32(TOK_RBRACE)),
                        ),
                    ),
                    Node::if_then(
                        Expr::eq(Expr::var(&prefix_kind), Expr::u32(TOK_RBRACE)),
                        vec![Node::assign(
                            &brace_depth,
                            Expr::add(Expr::var(&brace_depth), Expr::u32(1)),
                        )],
                    ),
                    Node::if_then(
                        Expr::and(
                            Expr::gt(Expr::var(&brace_depth), Expr::u32(0)),
                            Expr::eq(Expr::var(&prefix_kind), Expr::u32(TOK_LBRACE)),
                        ),
                        vec![Node::assign(
                            &brace_depth,
                            Expr::sub(Expr::var(&brace_depth), Expr::u32(1)),
                        )],
                    ),
                    Node::if_then(
                        Expr::eq(Expr::var(&prefix_kind), Expr::u32(TOK_RPAREN)),
                        vec![Node::assign(
                            &paren_depth,
                            Expr::add(Expr::var(&paren_depth), Expr::u32(1)),
                        )],
                    ),
                    Node::if_then(
                        Expr::and(
                            Expr::gt(Expr::var(&paren_depth), Expr::u32(0)),
                            Expr::eq(Expr::var(&prefix_kind), Expr::u32(TOK_LPAREN)),
                        ),
                        vec![Node::assign(
                            &paren_depth,
                            Expr::sub(Expr::var(&paren_depth), Expr::u32(1)),
                        )],
                    ),
                    Node::if_then(
                        Expr::not(Expr::or(Expr::var(&in_brace), Expr::var(&in_paren))),
                        row_body,
                    ),
                ],
            )],
        ),
    ]
}
