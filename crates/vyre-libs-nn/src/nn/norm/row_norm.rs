//! Shared row-wise normalization body construction.

use vyre_foundation::ir::{DataType, Expr, Node};

pub(super) fn row_sum_squares_body(
    input: &str,
    total: u32,
    row_len: u32,
    output: &str,
    output_dtype: DataType,
    result_expr: Expr,
) -> Vec<Node> {
    let index = Expr::var("index");
    let row_start = Expr::mul(
        Expr::div(index.clone(), Expr::u32(row_len)),
        Expr::u32(row_len),
    );
    vec![
        Node::let_bind("index", Expr::LogicalIndex { axis: 0 }),
        Node::if_then(
            Expr::lt(index.clone(), Expr::u32(total)),
            vec![
                Node::let_bind("row_start", row_start),
                Node::let_bind("sum_squares", Expr::f32(0.0)),
                Node::loop_for(
                    "offset",
                    Expr::u32(0),
                    Expr::u32(row_len),
                    vec![
                        Node::let_bind(
                            "row_elem",
                            Expr::cast(
                                DataType::F32,
                                Expr::load(
                                    input,
                                    Expr::add(Expr::var("row_start"), Expr::var("offset")),
                                ),
                            ),
                        ),
                        Node::assign(
                            "sum_squares",
                            Expr::add(
                                Expr::var("sum_squares"),
                                Expr::mul(Expr::var("row_elem"), Expr::var("row_elem")),
                            ),
                        ),
                    ],
                ),
                Node::Store {
                    buffer: output.into(),
                    index,
                    value: Expr::cast(output_dtype, result_expr),
                },
            ],
        ),
    ]
}
