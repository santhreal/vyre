//! Cooperative inner kernel body shared by the plain and bias-fused
//! tiled matmul variants.

use vyre_foundation::ir::{Expr, Node};

use vyre_libs_builder::builder::matrix_tile::{
    bind_output_tile_coordinates, cooperative_slab_loop, in_output_bounds, MatrixShape,
    OutputTileCoordNames, TileArithmetic, TileShape,
};

/// Build the cooperative staging body for a tiled matmul.
///
/// `zero` is the element type's own zero. It seeds the accumulator and pads the
/// lanes of a tile that the source matrix does not cover, so a partially filled
/// tile contributes nothing to the product.
///
/// A bias is added into the accumulator before the first slab rather than at
/// the store, which is what keeps the summation order the same as the untiled
/// variant this replaces.
pub(crate) fn cooperative_matmul_body(
    a: &str,
    b: &str,
    bias: Option<&str>,
    out: &str,
    shape: MatrixShape,
    tile: TileShape,
    a_tile_name: &str,
    b_tile_name: &str,
    zero: &Expr,
) -> Vec<Node> {
    let row = Expr::var("row");
    let col = Expr::var("col");
    let in_bounds = in_output_bounds(row.clone(), col.clone(), shape);
    let out_index = Expr::add(Expr::mul(row, Expr::u32(shape.n)), col.clone());

    let mut body = bind_output_tile_coordinates(
        shape,
        tile,
        OutputTileCoordNames {
            lane_row: "local_row",
            lane_col: "local_col",
            row: "row",
            col: "col",
        },
    );
    body.push(Node::let_bind("acc", zero.clone()));
    if let Some(bias) = bias {
        body.push(Node::if_then(
            in_bounds.clone(),
            vec![Node::assign("acc", Expr::load(bias, col))],
        ));
    }
    body.push(cooperative_slab_loop(
        a,
        b,
        a_tile_name,
        b_tile_name,
        shape,
        tile,
        &in_bounds,
        &TileArithmetic {
            pad: zero.clone(),
            combine: &|left, right| Expr::mul(left, right),
            accumulate: &|acc, value| Expr::add(acc, value),
        },
    ));
    body.push(Node::if_then(
        in_bounds,
        vec![Node::Store {
            buffer: out.into(),
            index: out_index,
            value: Expr::var("acc"),
        }],
    ));
    body
}
