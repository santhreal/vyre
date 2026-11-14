//! MMA-oriented tiled matmul body for full F16 M16N8-aligned tiles.

use vyre_foundation::ir::{DataType, Expr, Node};

use super::mma_fragment::matmul_mma_fragment;
use crate::builder::matrix_tile::{bind_output_tile_coordinates, OutputTileCoordNames};
use crate::builder::matrix_tile::{MatrixShape, TileShape};

/// Build the MMA-oriented body for a full M16N8-aligned tile.
///
/// Operands stay in `dtype` in memory and accumulate in f32, which is the
/// `MatrixMma { F16, F16, F32 }` form this fragment sequence targets. The IR
/// `Fma` takes three f32 operands, so each operand load is cast on the way in
/// and each accumulator is cast back to `dtype` on the way out.
pub(super) fn cooperative_matmul_body_mma(
    a: &str,
    b: &str,
    bias: Option<&str>,
    out: &str,
    shape: MatrixShape,
    tile: TileShape,
    dtype: DataType,
) -> Vec<Node> {
    let is_f16 = matches!(dtype, DataType::F16);
    let is_bf16 = matches!(dtype, DataType::BF16);
    let to_f32 = |value: Expr| Expr::cast(DataType::F32, value);
    let from_f32 = move |value: Expr| {
        let out_dtype = if is_f16 {
            DataType::F16
        } else if is_bf16 {
            DataType::BF16
        } else {
            DataType::F32
        };
        Expr::cast(out_dtype, value)
    };
    let col_in_bounds = || Expr::lt(Expr::var("col"), Expr::u32(shape.n));
    let row_in_bounds =
        |r: &'static str| Expr::and(Expr::lt(Expr::var(r), Expr::u32(shape.m)), col_in_bounds());
    const ROWS: [(&str, &str, &str); 4] = [
        ("row0", "acc0", "mma_c0"),
        ("row1", "acc1", "mma_c1"),
        ("row2", "acc2", "mma_c2"),
        ("row3", "acc3", "mma_c3"),
    ];
    let mut body = bind_output_tile_coordinates(
        shape,
        tile,
        OutputTileCoordNames {
            lane_row: "lane_row",
            lane_col: "lane_col",
            row: "row0",
            col: "col",
        },
    );
    body.extend([
        Node::let_bind("row1", Expr::add(Expr::var("row0"), Expr::u32(4))),
        Node::let_bind("row2", Expr::add(Expr::var("row0"), Expr::u32(8))),
        Node::let_bind("row3", Expr::add(Expr::var("row0"), Expr::u32(12))),
        Node::let_bind("acc0", Expr::f32(0.0)),
        Node::let_bind("acc1", Expr::f32(0.0)),
        Node::let_bind("acc2", Expr::f32(0.0)),
        Node::let_bind("acc3", Expr::f32(0.0)),
    ]);
    if let Some(bias) = bias {
        body.push(Node::if_then(
            col_in_bounds(),
            ROWS.iter()
                .map(|&(_, acc, _)| Node::assign(acc, to_f32(Expr::load(bias, Expr::var("col")))))
                .collect(),
        ));
    }
    body.push(Node::loop_for(
        "kk",
        Expr::u32(0),
        Expr::u32(shape.k),
        vec![Node::let_bind(
            "b_val",
            to_f32(Expr::load(
                b,
                Expr::add(
                    Expr::mul(Expr::var("kk"), Expr::u32(shape.n)),
                    Expr::var("col"),
                ),
            )),
        )]
        .into_iter()
        .chain(matmul_mma_fragment(
            to_f32(Expr::load(
                a,
                Expr::add(
                    Expr::mul(Expr::var("row0"), Expr::u32(shape.k)),
                    Expr::var("kk"),
                ),
            )),
            to_f32(Expr::load(
                a,
                Expr::add(
                    Expr::mul(Expr::var("row1"), Expr::u32(shape.k)),
                    Expr::var("kk"),
                ),
            )),
            to_f32(Expr::load(
                a,
                Expr::add(
                    Expr::mul(Expr::var("row2"), Expr::u32(shape.k)),
                    Expr::var("kk"),
                ),
            )),
            to_f32(Expr::load(
                a,
                Expr::add(
                    Expr::mul(Expr::var("row3"), Expr::u32(shape.k)),
                    Expr::var("kk"),
                ),
            )),
            Expr::var("b_val"),
            Expr::var("b_val"),
            Expr::var("acc0"),
            Expr::var("acc1"),
            Expr::var("acc2"),
            Expr::var("acc3"),
        ))
        .chain(ROWS.iter().map(|&(r, acc, mma)| {
            Node::if_then(row_in_bounds(r), vec![Node::assign(acc, Expr::var(mma))])
        }))
        .collect(),
    ));
    body.extend(ROWS.iter().map(|&(r, acc, _)| {
        Node::if_then(
            row_in_bounds(r),
            vec![Node::Store {
                buffer: out.into(),
                index: Expr::add(
                    Expr::mul(Expr::var(r), Expr::u32(shape.n)),
                    Expr::var("col"),
                ),
                value: from_f32(Expr::var(acc)),
            }],
        )
    }));
    body
}
