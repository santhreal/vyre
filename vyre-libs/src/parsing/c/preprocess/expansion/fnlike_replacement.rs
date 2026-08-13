//! Function-like macro replacement: the argument scan, the replacement-token
//! decode, and the three-way split on `#`, `##`, and plain tokens.

use crate::parsing::c::lex::tokens::{TOK_HASH, TOK_HASHHASH};
use vyre_foundation::ir::{Expr, Node};

use super::arg_scan::emit_function_like_argument_scan;

/// Branch bodies for the function-like replacement walk.
///
/// The token-only and materialized kernels share the argument scan, the
/// skip-one-token guard, the replacement-token decode, and the three-way split
/// on `#`, `##`, and everything else. They disagree only on what each of those
/// three branches emits.
pub(super) struct FunctionLikeReplacementSpec<'a> {
    /// Input token-type stream, used by the argument scan.
    pub(super) in_tok_types: &'a str,
    /// Replacement token-type table.
    pub(super) macro_vals: &'a str,
    /// Replacement parameter-index table.
    pub(super) macro_replacement_params: &'a str,
    /// Workgroup argument start bounds.
    pub(super) macro_arg_starts: &'a str,
    /// Workgroup argument end bounds.
    pub(super) macro_arg_ends: &'a str,
    /// Token count bounding the argument scan.
    pub(super) num_tokens: Expr,
    /// `#param` stringification.
    pub(super) stringify: Vec<Node>,
    /// `lhs ## rhs` token paste.
    pub(super) paste: Vec<Node>,
    /// Any other replacement token.
    pub(super) regular: Vec<Node>,
}

/// Build the function-like replacement walk.
pub(super) fn emit_function_like_replacement_walk(
    spec: FunctionLikeReplacementSpec<'_>,
) -> Vec<Node> {
    let mut nodes = emit_function_like_argument_scan(
        spec.in_tok_types,
        spec.macro_arg_starts,
        spec.macro_arg_ends,
        spec.num_tokens,
    );
    nodes.extend([
        Node::let_bind("named_skip_repl", Expr::u32(0)),
        Node::loop_for(
            "named_repl_i",
            Expr::u32(0),
            Expr::var("named_repl_size"),
            vec![Node::if_then_else(
                Expr::eq(Expr::var("named_skip_repl"), Expr::u32(1)),
                vec![Node::assign("named_skip_repl", Expr::u32(0))],
                {
                    let mut repl = vec![
                        Node::let_bind(
                            "named_repl_offset",
                            Expr::add(Expr::var("named_macro_idx"), Expr::var("named_repl_i")),
                        ),
                        Node::let_bind(
                            "named_repl_param",
                            Expr::load(
                                spec.macro_replacement_params,
                                Expr::var("named_repl_offset"),
                            ),
                        ),
                        Node::let_bind(
                            "named_repl_tok",
                            Expr::load(spec.macro_vals, Expr::var("named_repl_offset")),
                        ),
                    ];
                    repl.push(Node::if_then_else(
                        Expr::and(
                            Expr::eq(Expr::var("named_repl_tok"), Expr::u32(TOK_HASH)),
                            Expr::lt(
                                Expr::add(Expr::var("named_repl_i"), Expr::u32(1)),
                                Expr::var("named_repl_size"),
                            ),
                        ),
                        spec.stringify,
                        vec![Node::if_then_else(
                            Expr::eq(Expr::var("named_repl_tok"), Expr::u32(TOK_HASHHASH)),
                            spec.paste,
                            spec.regular,
                        )],
                    ));
                    repl
                },
            )],
        ),
        Node::assign(
            "named_i",
            Expr::add(Expr::var("macro_close_idx"), Expr::u32(1)),
        ),
    ]);
    nodes
}
