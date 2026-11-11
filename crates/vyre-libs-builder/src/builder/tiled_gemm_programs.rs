//! Tiled contraction program assembly: register tiles and cooperative shared-memory tiles.
//!
//! A tile stages each operand element once and serves every accumulator in its
//! tile row or column from that staged value, so the read count per
//! contraction step falls from two per output to one per tile edge.

use vyre_foundation::composition::wrap_region;
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

use super::gemm_algebra::ContractionSemiring;
use super::ContractionEpilogue;
use crate::plumbing::operand::element_zero::element_zero;
use crate::plumbing::operand::tensor_ref::{element_count, TensorRefError};

use super::contraction_buffers::{
    matmul_2d_counts, matmul_2d_operands, projection_counts, projection_operands,
};
use crate::builder::matrix_tile::{
    bind_output_tile_coordinates, cooperative_slab_loop, in_output_bounds, output_tile_shape,
    padded_tile_lane_count, MatrixShape, OutputTileCoordNames, TileArithmetic, TileShape,
};

/// Assemble a row-batched affine projection where one invocation accumulates a
/// `tile_rows x tile_columns` tile of outputs.
///
/// One staged left value serves every accumulator in its tile row and one
/// staged right value serves every accumulator in its tile column, so the tile
/// reads `tile_rows + tile_columns` elements per contraction step where the
/// untiled program reads two per output. Tile coordinates are clamped into the
/// declared extents, which is how a tile that overhangs the output stays
/// inside it: an overhanging position recomputes and rewrites the boundary
/// element instead of reaching past the declared shape.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_batched_rows_register_tiled(
    generator: &'static str,
    x: &str,
    w: &str,
    bias: Option<&str>,
    out: &str,
    rows: u32,
    in_dim: u32,
    out_dim: u32,
    tile_rows: u32,
    tile_columns: u32,
    dtype: &DataType,
    acc_dtype: &DataType,
    weight_out_in: bool,
    workgroup_size: [u32; 3],
) -> Result<Program, TensorRefError> {
    if rows == 0 || in_dim == 0 || out_dim == 0 || tile_rows == 0 || tile_columns == 0 {
        return Err(TensorRefError::ShapeMismatch {
            name: out.to_string(),
            found: vec![rows, in_dim, out_dim, tile_rows, tile_columns],
            expected: vec![1, 1, 1, 1, 1],
            op: "vyre-libs::builder::gemm::register_tiled",
        });
    }

    let (input_count, weight_count, output_count) =
        projection_counts(x, w, out, rows, in_dim, out_dim)?;
    let row_tiles = rows.div_ceil(tile_rows);
    let column_tiles = out_dim.div_ceil(tile_columns);
    let tile_count = row_tiles.checked_mul(column_tiles).ok_or_else(|| {
        TensorRefError::ElementCountOverflow {
            name: out.to_string(),
            shape: vec![row_tiles, column_tiles],
        }
    })?;

    let row_name = |r: u32| format!("row_{r}");
    let column_name = |c: u32| format!("column_{c}");
    let left_name = |r: u32| format!("left_{r}");
    let right_name = |c: u32| format!("right_{c}");
    let acc_name = |r: u32, c: u32| format!("accumulator_{r}_{c}");

    let mut tile_body = vec![
        Node::let_bind(
            "tile_row",
            Expr::div(Expr::var("tile"), Expr::u32(column_tiles)),
        ),
        Node::let_bind(
            "tile_column",
            Expr::rem(Expr::var("tile"), Expr::u32(column_tiles)),
        ),
        Node::let_bind(
            "row_origin",
            Expr::mul(Expr::var("tile_row"), Expr::u32(tile_rows)),
        ),
        Node::let_bind(
            "column_origin",
            Expr::mul(Expr::var("tile_column"), Expr::u32(tile_columns)),
        ),
    ];

    for r in 0..tile_rows {
        tile_body.push(Node::let_bind(
            row_name(r),
            Expr::min(
                Expr::add(Expr::var("row_origin"), Expr::u32(r)),
                Expr::u32(rows - 1),
            ),
        ));
    }
    for c in 0..tile_columns {
        tile_body.push(Node::let_bind(
            column_name(c),
            Expr::min(
                Expr::add(Expr::var("column_origin"), Expr::u32(c)),
                Expr::u32(out_dim - 1),
            ),
        ));
    }

    let zero = element_zero(acc_dtype).unwrap_or_else(|| Expr::u32(0));
    for r in 0..tile_rows {
        for c in 0..tile_columns {
            let initial = bias.map_or_else(
                || zero.clone(),
                |name| {
                    Expr::cast(
                        acc_dtype.clone(),
                        Expr::load(name, Expr::var(column_name(c))),
                    )
                },
            );
            tile_body.push(Node::let_bind(acc_name(r, c), initial));
        }
    }

    let mut inner_body = Vec::with_capacity(
        (tile_rows as usize + tile_columns as usize) + (tile_rows as usize * tile_columns as usize),
    );
    for r in 0..tile_rows {
        inner_body.push(Node::let_bind(
            left_name(r),
            Expr::cast(
                acc_dtype.clone(),
                Expr::load(
                    x,
                    Expr::add(
                        Expr::mul(Expr::var(row_name(r)), Expr::u32(in_dim)),
                        Expr::var("inner"),
                    ),
                ),
            ),
        ));
    }
    for c in 0..tile_columns {
        let weight_index = if weight_out_in {
            Expr::add(
                Expr::mul(Expr::var(column_name(c)), Expr::u32(in_dim)),
                Expr::var("inner"),
            )
        } else {
            Expr::add(
                Expr::mul(Expr::var("inner"), Expr::u32(out_dim)),
                Expr::var(column_name(c)),
            )
        };
        inner_body.push(Node::let_bind(
            right_name(c),
            Expr::cast(acc_dtype.clone(), Expr::load(w, weight_index)),
        ));
    }
    for r in 0..tile_rows {
        for c in 0..tile_columns {
            inner_body.push(Node::assign(
                acc_name(r, c),
                Expr::add(
                    Expr::var(acc_name(r, c)),
                    Expr::mul(Expr::var(left_name(r)), Expr::var(right_name(c))),
                ),
            ));
        }
    }
    tile_body.push(Node::loop_for(
        "inner",
        Expr::u32(0),
        Expr::u32(in_dim),
        inner_body,
    ));

    // Every coordinate is clamped into the declared extents, so an overhanging
    // tile position accumulates and rewrites the boundary element rather than
    // reaching past it. Clamping is what bounds the write; a second predicate
    // on the unclamped coordinate would guard a store that is already correct.
    for r in 0..tile_rows {
        for c in 0..tile_columns {
            tile_body.push(Node::Store {
                buffer: out.into(),
                index: Expr::add(
                    Expr::mul(Expr::var(row_name(r)), Expr::u32(out_dim)),
                    Expr::var(column_name(c)),
                ),
                value: Expr::cast(dtype.clone(), Expr::var(acc_name(r, c))),
            });
        }
    }

    let body = vec![
        Node::let_bind("tile", Expr::LogicalIndex { axis: 0 }),
        Node::if_then(
            Expr::lt(Expr::var("tile"), Expr::u32(tile_count)),
            tile_body,
        ),
    ];

    let (mut buffers, output_slot) =
        projection_operands(x, input_count, w, weight_count, bias, out_dim, dtype);
    buffers.push(BufferDecl::output(out, output_slot, dtype.clone()).with_count(output_count));

    let region = wrap_region(generator, body, None);

    Ok(Program::wrapped(buffers, workgroup_size, vec![region]))
}

/// Assemble 2D GEMM with 2D cooperative shared-memory tiling.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_matmul_2d_cooperative(
    generator: &'static str,
    a: &str,
    b: &str,
    bias: Option<&str>,
    out: &str,
    m: u32,
    k: u32,
    n: u32,
    tile: u32,
    a_tile_name: &str,
    b_tile_name: &str,
    dtype: &DataType,
    semiring: &ContractionSemiring,
    epilogue: &ContractionEpilogue,
    workgroup_size: [u32; 3],
) -> Result<Program, TensorRefError> {
    let (a_count, b_count, out_count) = matmul_2d_counts(a, b, out, m, k, n)?;

    let shape = MatrixShape { m, k, n };
    let (out_cols, out_rows, lanes) = output_tile_shape(workgroup_size)?;
    let k_tile = tile;
    let a_tile_count = element_count(a_tile_name, &[out_rows, k_tile])?;
    let b_tile_count = element_count(b_tile_name, &[k_tile, out_cols])?;
    let padded_out_count = padded_tile_lane_count(m, n, out_rows, out_cols, lanes)?;
    let tile_shape = TileShape {
        k_tile,
        out_rows,
        out_cols,
        x_lanes: workgroup_size[0].max(1),
        y_lanes: workgroup_size[1].max(1),
        lanes,
        a_values: a_tile_count,
        b_values: b_tile_count,
    };

    let row = Expr::var("row");
    let col = Expr::var("col");
    let in_bounds = in_output_bounds(row.clone(), col.clone(), shape);

    let store_value = match epilogue {
        ContractionEpilogue::None => Expr::var("acc"),
        ContractionEpilogue::Bias { buffer, .. } => {
            Expr::add(Expr::var("acc"), Expr::load(buffer, col.clone()))
        }
        ContractionEpilogue::Activation {
            bias: Some(bias_buf),
            activation,
        } => activation(Expr::add(
            Expr::var("acc"),
            Expr::load(bias_buf, col.clone()),
        )),
        ContractionEpilogue::Activation {
            bias: None,
            activation,
        } => activation(Expr::var("acc")),
        ContractionEpilogue::QuantizedScale {
            row_scales,
            batch_scales,
        } => Expr::mul(
            Expr::mul(Expr::var("acc"), Expr::load(row_scales, row.clone())),
            Expr::load(batch_scales, Expr::u32(0)),
        ),
    };

    let out_index = Expr::add(Expr::mul(row.clone(), Expr::u32(n)), col.clone());

    let mut body = bind_output_tile_coordinates(
        shape,
        tile_shape,
        OutputTileCoordNames {
            lane_row: "local_row",
            lane_col: "local_col",
            row: "row",
            col: "col",
        },
    );
    body.push(Node::let_bind("acc", semiring.identity_expr(dtype)));
    body.push(cooperative_slab_loop(
        a,
        b,
        a_tile_name,
        b_tile_name,
        shape,
        tile_shape,
        &in_bounds,
        &TileArithmetic {
            pad: semiring.identity_expr(dtype),
            combine: &|left, right| semiring.combine_expr(left, right),
            accumulate: &|acc, value| semiring.accumulate_expr(acc, value),
        },
    ));
    body.push(Node::if_then(
        in_bounds,
        vec![Node::Store {
            buffer: out.into(),
            index: out_index,
            value: store_value,
        }],
    ));

    let (mut buffers, next_slot) =
        matmul_2d_operands(a, a_count, b, b_count, bias, epilogue, dtype, m, n);
    buffers.push(BufferDecl::workgroup(
        a_tile_name,
        a_tile_count,
        dtype.clone(),
    ));
    buffers.push(BufferDecl::workgroup(
        b_tile_name,
        b_tile_count,
        dtype.clone(),
    ));
    let mut out_decl =
        BufferDecl::output(out, next_slot, dtype.clone()).with_count(padded_out_count);
    if let Some(element_size) = dtype.size_bytes() {
        let logical_output_bytes = (out_count as u64).saturating_mul(element_size as u64);
        out_decl = out_decl.with_output_byte_range(0..logical_output_bytes);
    }
    buffers.push(out_decl);

    let region = wrap_region(generator, body, None);

    Ok(Program::wrapped(buffers, [lanes, 1, 1], vec![region]))
}
