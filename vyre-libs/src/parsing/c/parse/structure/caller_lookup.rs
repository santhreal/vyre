use super::*;

pub(super) fn emit_enclosing_function_lookup(
    functions: &str,
    num_functions: Expr,
    token_idx: Expr,
) -> Vec<Node> {
    vec![
        Node::let_bind("caller_id", Expr::u32(u32::MAX)),
        Node::loop_for(
            "caller_fn_scan",
            Expr::u32(0),
            num_functions,
            vec![
                Node::let_bind(
                    "fn_rec_base",
                    Expr::mul(Expr::var("caller_fn_scan"), Expr::u32(3)),
                ),
                Node::let_bind(
                    "fn_body_end",
                    Expr::load(functions, Expr::add(Expr::var("fn_rec_base"), Expr::u32(2))),
                ),
                Node::let_bind(
                    "fn_body_start",
                    Expr::load(functions, Expr::add(Expr::var("fn_rec_base"), Expr::u32(1))),
                ),
                Node::if_then(
                    Expr::and(
                        Expr::eq(Expr::var("caller_id"), Expr::u32(u32::MAX)),
                        Expr::and(
                            Expr::ge(token_idx.clone(), Expr::var("fn_body_start")),
                            Expr::le(token_idx.clone(), Expr::var("fn_body_end")),
                        ),
                    ),
                    vec![Node::assign("caller_id", Expr::var("caller_fn_scan"))],
                ),
            ],
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn function_record_body_loads_use_alignment_safe_order() {
        let nodes = emit_enclosing_function_lookup("functions", Expr::u32(1), Expr::u32(0));
        let loop_body = match &nodes[1] {
            Node::Loop { body, .. } => body,
            other => panic!(
                "Fix: caller lookup must contain one bounded function-record loop, got {other:?}."
            ),
        };
        let load_offset = |node: &Node, expected_name: &str| {
            match node {
            Node::Let {
                name,
                value:
                    Expr::Load {
                        buffer,
                        index,
                    },
            } if name.as_str() == expected_name && buffer.as_str() == "functions" => match index.as_ref() {
                Expr::BinOp {
                    op: vyre_foundation::ir::BinOp::Add,
                    right,
                    ..
                } => match right.as_ref() {
                    Expr::LitU32(offset) => *offset,
                    other => panic!("Fix: function-record load offset must be a u32 literal, got {other:?}."),
                },
                other => panic!("Fix: function-record load must index record base plus field offset, got {other:?}."),
            },
            other => panic!("Fix: expected `{expected_name}` function-record load, got {other:?}."),
        }
        };

        assert_eq!(load_offset(&loop_body[1], "fn_body_end"), 2);
        assert_eq!(load_offset(&loop_body[2], "fn_body_start"), 1);
    }
}
