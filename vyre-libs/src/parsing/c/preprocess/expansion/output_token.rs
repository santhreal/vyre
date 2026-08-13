//! Append one token to the expansion output buffer.

use vyre_foundation::ir::{Expr, Node};

pub(super) fn emit_one_output_token(
    out_tok_types: &str,
    token: Expr,
    max_out_tokens: u32,
) -> Vec<Node> {
    vec![
        Node::if_then(
            Expr::gt(
                Expr::add(Expr::var("named_out_idx"), Expr::u32(1)),
                Expr::u32(max_out_tokens),
            ),
            vec![Node::trap(
                Expr::add(Expr::var("named_out_idx"), Expr::u32(1)),
                "named-macro-expansion-output-overflow",
            )],
        ),
        Node::store(out_tok_types, Expr::var("named_out_idx"), token),
        Node::assign(
            "named_out_idx",
            Expr::add(Expr::var("named_out_idx"), Expr::u32(1)),
        ),
    ]
}
