//! The brace-balanced declaration span every Go declaration kind shares.

use crate::parsing::go::parse::token_predicates::{
    token_len, token_start, token_type_eq,
};
use vyre_foundation::ir::{Expr, Node};
use vyre_spec::go_token::{TOK_LBRACE, TOK_RBRACE};

use super::GO_DECL_RECORD_WORDS;

/// The one brace-balanced body span plus declaration record Go declaration
/// extraction emits.
///
/// Function, method, and interface declarations differ only in where the body
/// scan starts, which kind word is recorded, and which token carries the name.
pub(super) struct GoDeclSpan<'a> {
    pub(super) tok_types: &'a str,
    pub(super) tok_starts: &'a str,
    pub(super) tok_lens: &'a str,
    pub(super) out_decls: &'a str,
    pub(super) out_decl_counts: &'a str,
    pub(super) num_tokens: &'a Expr,
}

impl GoDeclSpan<'_> {
    /// Scans forward from `scan_from` for the outermost balanced `{`..`}` pair,
    /// then appends the five-word record for `name_tok`.
    pub(super) fn nodes(&self, scan_from: Expr, kind: Expr, name_tok: Expr) -> Vec<Node> {
        vec![
            Node::let_bind("body_start", Expr::u32(0)),
            Node::let_bind("body_end", Expr::u32(0)),
            Node::let_bind("brace_depth", Expr::u32(0)),
            Node::let_bind("brace_done", Expr::u32(0)),
            Node::loop_for(
                "scan",
                scan_from,
                self.num_tokens.clone(),
                vec![Node::if_then(
                    Expr::eq(Expr::var("brace_done"), Expr::u32(0)),
                    vec![
                        Node::if_then(
                            token_type_eq(self.tok_types, Expr::var("scan"), TOK_LBRACE),
                            vec![
                                Node::if_then(
                                    Expr::eq(Expr::var("brace_depth"), Expr::u32(0)),
                                    vec![Node::assign(
                                        "body_start",
                                        token_start(self.tok_starts, Expr::var("scan")),
                                    )],
                                ),
                                Node::assign(
                                    "brace_depth",
                                    Expr::add(Expr::var("brace_depth"), Expr::u32(1)),
                                ),
                            ],
                        ),
                        Node::if_then(
                            token_type_eq(self.tok_types, Expr::var("scan"), TOK_RBRACE),
                            vec![
                                Node::assign(
                                    "brace_depth",
                                    Expr::sub(Expr::var("brace_depth"), Expr::u32(1)),
                                ),
                                Node::if_then(
                                    Expr::eq(Expr::var("brace_depth"), Expr::u32(0)),
                                    vec![
                                        Node::assign(
                                            "body_end",
                                            Expr::add(
                                                token_start(self.tok_starts, Expr::var("scan")),
                                                token_len(self.tok_lens, Expr::var("scan")),
                                            ),
                                        ),
                                        Node::assign("brace_done", Expr::u32(1)),
                                    ],
                                ),
                            ],
                        ),
                    ],
                )],
            ),
            Node::let_bind(
                "decl_idx",
                Expr::atomic_add(
                    self.out_decl_counts,
                    Expr::u32(0),
                    Expr::u32(GO_DECL_RECORD_WORDS),
                ),
            ),
            Node::store(self.out_decls, Expr::var("decl_idx"), kind),
            Node::store(
                self.out_decls,
                Expr::add(Expr::var("decl_idx"), Expr::u32(1)),
                token_start(self.tok_starts, name_tok.clone()),
            ),
            Node::store(
                self.out_decls,
                Expr::add(Expr::var("decl_idx"), Expr::u32(2)),
                token_len(self.tok_lens, name_tok),
            ),
            Node::store(
                self.out_decls,
                Expr::add(Expr::var("decl_idx"), Expr::u32(3)),
                Expr::var("body_start"),
            ),
            Node::store(
                self.out_decls,
                Expr::add(Expr::var("decl_idx"), Expr::u32(4)),
                Expr::var("body_end"),
            ),
        ]
    }
}
