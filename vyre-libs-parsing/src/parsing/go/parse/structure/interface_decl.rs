//! Go interface declaration extraction.

use crate::parsing::go::parse::token_predicates::{token_is_ident, token_is_keyword};
use vyre_foundation::ir::{Expr, Node};

use super::decl_span::GoDeclSpan;
use super::GO_DECL_INTERFACE;

impl GoDeclSpan<'_> {
    /// The `type NAME interface { .. }` declaration, scanned from `t`.
    ///
    /// Three tokens form the header, so the branch is admitted only when
    /// `t + 2` exists; without the bound the predicates read one past the
    /// token arrays. Every operand is safe to evaluate for every lane, which
    /// the non-short-circuiting `and` requires.
    pub(super) fn interface_node(&self, haystack: &str, t: &Expr) -> Node {
        Node::if_then(
            Expr::lt(Expr::add(t.clone(), Expr::u32(2)), self.num_tokens.clone()),
            vec![Node::if_then(
                Expr::and(
                    token_is_keyword(
                        haystack,
                        self.tok_types,
                        self.tok_starts,
                        self.tok_lens,
                        t.clone(),
                        b"type",
                    ),
                    Expr::and(
                        token_is_ident(self.tok_types, Expr::add(t.clone(), Expr::u32(1))),
                        token_is_keyword(
                            haystack,
                            self.tok_types,
                            self.tok_starts,
                            self.tok_lens,
                            Expr::add(t.clone(), Expr::u32(2)),
                            b"interface",
                        ),
                    ),
                ),
                self.nodes(
                    Expr::add(t.clone(), Expr::u32(3)),
                    Expr::u32(GO_DECL_INTERFACE),
                    Expr::add(t.clone(), Expr::u32(1)),
                ),
            )],
        )
    }
}
