use crate::parsing::c::lex::tokens::*;
use vyre::ir::{Expr, Node};

pub(super) fn emit_forward_matching_paren_scan(
    tok_types: &str,
    scan_index_name: &str,
    scan_start: Expr,
    scan_end: Expr,
    scan_token_name: &str,
    depth_name: &str,
    out_match_name: &str,
    guard: Option<Expr>,
) -> Vec<Node> {
    let loop_node = Node::loop_for(
        scan_index_name,
        scan_start,
        scan_end,
        vec![
            Node::let_bind(
                scan_token_name,
                Expr::load(tok_types, Expr::var(scan_index_name)),
            ),
            Node::if_then(
                Expr::eq(Expr::var(scan_token_name), Expr::u32(TOK_LPAREN)),
                vec![Node::assign(
                    depth_name,
                    Expr::add(Expr::var(depth_name), Expr::u32(1)),
                )],
            ),
            Node::if_then(
                Expr::and(
                    Expr::eq(Expr::var(out_match_name), Expr::u32(u32::MAX)),
                    Expr::eq(Expr::var(scan_token_name), Expr::u32(TOK_RPAREN)),
                ),
                vec![Node::if_then_else(
                    Expr::eq(Expr::var(depth_name), Expr::u32(1)),
                    vec![Node::assign(out_match_name, Expr::var(scan_index_name))],
                    vec![Node::assign(
                        depth_name,
                        Expr::sub(Expr::var(depth_name), Expr::u32(1)),
                    )],
                )],
            ),
        ],
    );

    let mut nodes = vec![
        Node::let_bind(out_match_name, Expr::u32(u32::MAX)),
        Node::let_bind(depth_name, Expr::u32(1)),
    ];
    if let Some(guard) = guard {
        nodes.push(Node::if_then(guard, vec![loop_node]));
    } else {
        nodes.push(loop_node);
    }
    nodes
}
