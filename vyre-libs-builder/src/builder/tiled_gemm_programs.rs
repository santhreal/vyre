//! Tiled contraction program assembly: register tiles and cooperative shared-memory tiles.
//!
//! A tile stages each operand element once and serves every accumulator in its
//! tile row or column from that staged value, so the read count per
//! contraction step falls from two per output to one per tile edge.

use vyre_foundation::composition::{wrap_anonymous_region, wrap_region};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use super::gemm_algebra::ContractionSemiring;
use super::ContractionEpilogue;
use crate::plumbing::operand::element_zero::element_zero;
use crate::plumbing::operand::tensor_ref::TensorRefError;

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

    let input_count =
        rows.checked_mul(in_dim)
            .ok_or_else(|| TensorRefError::ElementCountOverflow {
                name: x.to_string(),
                shape: vec![rows, in_dim],
            })?;
    let output_count =
        rows.checked_mul(out_dim)
            .ok_or_else(|| TensorRefError::ElementCountOverflow {
                name: out.to_string(),
                shape: vec![rows, out_dim],
            })?;
    let weight_count =
        in_dim
            .checked_mul(out_dim)
            .ok_or_else(|| TensorRefError::ElementCountOverflow {
                name: w.to_string(),
                shape: vec![in_dim, out_dim],
            })?;
    let row_tiles = rows.div_ceil(tile_rows);
    let column_tiles = out_dim.div_ceil(tile_columns);
    let tile_count =
        row_tiles
            .checked_mul(column_tiles)
            .ok_or_else(|| TensorRefError::ElementCountOverflow {
                name: out.to_string(),
                shape: vec![row_tiles, column_tiles],
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

    let mut buffers = vec![
        BufferDecl::storage(x, 0, BufferAccess::ReadOnly, dtype.clone()).with_count(input_count),
        BufferDecl::storage(w, 1, BufferAccess::ReadOnly, dtype.clone()).with_count(weight_count),
    ];
    let output_slot = if let Some(name) = bias {
        buffers.push(
            BufferDecl::storage(name, 2, BufferAccess::ReadOnly, dtype.clone()).with_count(out_dim),
        );
        3
    } else {
        2
    };
    buffers.push(BufferDecl::output(out, output_slot, dtype.clone()).with_count(output_count));

    let region = if generator.starts_with("anonymous::") {
        wrap_anonymous_region(generator, body)
    } else {
        wrap_region(generator, body, None)
    };

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
    let a_count = m
        .checked_mul(k)
        .ok_or_else(|| TensorRefError::ElementCountOverflow {
            name: a.to_string(),
            shape: vec![m, k],
        })?;
    let b_count = k
        .checked_mul(n)
        .ok_or_else(|| TensorRefError::ElementCountOverflow {
            name: b.to_string(),
            shape: vec![k, n],
        })?;
    let out_count = m
        .checked_mul(n)
        .ok_or_else(|| TensorRefError::ElementCountOverflow {
            name: out.to_string(),
            shape: vec![m, n],
        })?;

    let out_cols = workgroup_size[0].max(1);
    let out_rows = workgroup_size[1]
        .max(1)
        .saturating_mul(workgroup_size[2].max(1));
    let lanes =
        out_cols
            .checked_mul(out_rows)
            .ok_or_else(|| TensorRefError::ElementCountOverflow {
                name: "workgroup_lanes".to_string(),
                shape: vec![out_rows, out_cols],
            })?;
    let k_tile = tile;

    let a_tile_count =
        out_rows
            .checked_mul(k_tile)
            .ok_or_else(|| TensorRefError::ElementCountOverflow {
                name: a_tile_name.to_string(),
                shape: vec![out_rows, k_tile],
            })?;
    let b_tile_count =
        k_tile
            .checked_mul(out_cols)
            .ok_or_else(|| TensorRefError::ElementCountOverflow {
                name: b_tile_name.to_string(),
                shape: vec![k_tile, out_cols],
            })?;

    let row_tiles = m.div_ceil(out_rows);
    let col_tiles = n.div_ceil(out_cols);
    let total_tiles =
        row_tiles
            .checked_mul(col_tiles)
            .ok_or_else(|| TensorRefError::ElementCountOverflow {
                name: "matmul_tiled_tiles".to_string(),
                shape: vec![row_tiles, col_tiles],
            })?;
    let padded_out_count =
        total_tiles
            .checked_mul(lanes)
            .ok_or_else(|| TensorRefError::ElementCountOverflow {
                name: "matmul_tiled_launch_lanes".to_string(),
                shape: vec![row_tiles, col_tiles, lanes],
            })?;

    let k_tile_count = k.div_ceil(k_tile);
    let load_passes = a_tile_count.max(b_tile_count).div_ceil(lanes).max(1);

    let local = Expr::var("local");
    let row = Expr::var("row");
    let col = Expr::var("col");

    let local_expr = Expr::add(
        Expr::add(
            Expr::LogicalWithinTileId { axis: 0 },
            Expr::mul(Expr::LogicalWithinTileId { axis: 1 }, Expr::u32(out_cols)),
        ),
        Expr::mul(
            Expr::LogicalWithinTileId { axis: 2 },
            Expr::u32(out_cols.saturating_mul(workgroup_size[1].max(1))),
        ),
    );

    let in_bounds = Expr::and(
        Expr::lt(row.clone(), Expr::u32(m)),
        Expr::lt(col.clone(), Expr::u32(n)),
    );

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

    let body = vec![
        Node::let_bind("local", local_expr),
        Node::let_bind("tile_block", Expr::LogicalTileId { axis: 0 }),
        Node::let_bind("tile_cols", Expr::u32(col_tiles)),
        Node::let_bind(
            "tile_row_base",
            Expr::mul(
                Expr::div(Expr::var("tile_block"), Expr::var("tile_cols")),
                Expr::u32(out_rows),
            ),
        ),
        Node::let_bind(
            "tile_col_base",
            Expr::mul(
                Expr::rem(Expr::var("tile_block"), Expr::var("tile_cols")),
                Expr::u32(out_cols),
            ),
        ),
        Node::let_bind("local_row", Expr::div(local.clone(), Expr::u32(out_cols))),
        Node::let_bind("local_col", Expr::rem(local.clone(), Expr::u32(out_cols))),
        Node::let_bind(
            "row",
            Expr::add(Expr::var("tile_row_base"), Expr::var("local_row")),
        ),
        Node::let_bind(
            "col",
            Expr::add(Expr::var("tile_col_base"), Expr::var("local_col")),
        ),
        Node::let_bind("acc", semiring.identity_expr(dtype)),
        Node::loop_for(
            "tile_idx",
            Expr::u32(0),
            Expr::u32(k_tile_count),
            vec![
                Node::let_bind(
                    "k_base",
                    Expr::mul(Expr::var("tile_idx"), Expr::u32(k_tile)),
                ),
                Node::loop_for(
                    "load_pass",
                    Expr::u32(0),
                    Expr::u32(load_passes),
                    vec![
                        Node::let_bind(
                            "a_linear",
                            Expr::add(
                                local.clone(),
                                Expr::mul(Expr::var("load_pass"), Expr::u32(lanes)),
                            ),
                        ),
                        Node::if_then(
                            Expr::lt(Expr::var("a_linear"), Expr::u32(a_tile_count)),
                            vec![
                                Node::let_bind(
                                    "a_local_row",
                                    Expr::div(Expr::var("a_linear"), Expr::u32(k_tile)),
                                ),
                                Node::let_bind(
                                    "a_local_k",
                                    Expr::rem(Expr::var("a_linear"), Expr::u32(k_tile)),
                                ),
                                Node::let_bind(
                                    "a_row",
                                    Expr::add(Expr::var("tile_row_base"), Expr::var("a_local_row")),
                                ),
                                Node::let_bind(
                                    "a_k",
                                    Expr::add(Expr::var("k_base"), Expr::var("a_local_k")),
                                ),
                                Node::Store {
                                    buffer: a_tile_name.into(),
                                    index: Expr::var("a_linear"),
                                    value: semiring.identity_expr(dtype),
                                },
                                Node::if_then(
                                    Expr::and(
                                        Expr::lt(Expr::var("a_row"), Expr::u32(m)),
                                        Expr::lt(Expr::var("a_k"), Expr::u32(k)),
                                    ),
                                    vec![Node::Store {
                                        buffer: a_tile_name.into(),
                                        index: Expr::var("a_linear"),
                                        value: Expr::load(
                                            a,
                                            Expr::add(
                                                Expr::mul(Expr::var("a_row"), Expr::u32(k)),
                                                Expr::var("a_k"),
                                            ),
                                        ),
                                    }],
                                ),
                            ],
                        ),
                        Node::let_bind(
                            "b_linear",
                            Expr::add(
                                local.clone(),
                                Expr::mul(Expr::var("load_pass"), Expr::u32(lanes)),
                            ),
                        ),
                        Node::if_then(
                            Expr::lt(Expr::var("b_linear"), Expr::u32(b_tile_count)),
                            vec![
                                Node::let_bind(
                                    "b_local_k",
                                    Expr::div(Expr::var("b_linear"), Expr::u32(out_cols)),
                                ),
                                Node::let_bind(
                                    "b_local_col",
                                    Expr::rem(Expr::var("b_linear"), Expr::u32(out_cols)),
                                ),
                                Node::let_bind(
                                    "b_k",
                                    Expr::add(Expr::var("k_base"), Expr::var("b_local_k")),
                                ),
                                Node::let_bind(
                                    "b_col",
                                    Expr::add(Expr::var("tile_col_base"), Expr::var("b_local_col")),
                                ),
                                Node::Store {
                                    buffer: b_tile_name.into(),
                                    index: Expr::var("b_linear"),
                                    value: semiring.identity_expr(dtype),
                                },
                                Node::if_then(
                                    Expr::and(
                                        Expr::lt(Expr::var("b_k"), Expr::u32(k)),
                                        Expr::lt(Expr::var("b_col"), Expr::u32(n)),
                                    ),
                                    vec![Node::Store {
                                        buffer: b_tile_name.into(),
                                        index: Expr::var("b_linear"),
                                        value: Expr::load(
                                            b,
                                            Expr::add(
                                                Expr::mul(Expr::var("b_k"), Expr::u32(n)),
                                                Expr::var("b_col"),
                                            ),
                                        ),
                                    }],
                                ),
                            ],
                        ),
                    ],
                ),
                Node::logical_barrier(vyre_foundation::ir::MemoryOrdering::SeqCst),
                Node::loop_for(
                    "tile_k",
                    Expr::u32(0),
                    Expr::u32(k_tile),
                    vec![Node::if_then(
                        in_bounds.clone(),
                        vec![Node::assign(
                            "acc",
                            semiring.accumulate_expr(
                                Expr::var("acc"),
                                semiring.combine_expr(
                                    Expr::load(
                                        a_tile_name,
                                        Expr::add(
                                            Expr::mul(Expr::var("local_row"), Expr::u32(k_tile)),
                                            Expr::var("tile_k"),
                                        ),
                                    ),
                                    Expr::load(
                                        b_tile_name,
                                        Expr::add(
                                            Expr::mul(Expr::var("tile_k"), Expr::u32(out_cols)),
                                            Expr::var("local_col"),
                                        ),
                                    ),
                                ),
                            ),
                        )],
                    )],
                ),
                Node::logical_barrier(vyre_foundation::ir::MemoryOrdering::SeqCst),
            ],
        ),
        Node::if_then(
            in_bounds,
            vec![Node::Store {
                buffer: out.into(),
                index: out_index,
                value: store_value,
            }],
        ),
    ];

    let mut buffers = vec![
        BufferDecl::storage(a, 0, BufferAccess::ReadOnly, dtype.clone()).with_count(a_count),
        BufferDecl::storage(b, 1, BufferAccess::ReadOnly, dtype.clone()).with_count(b_count),
    ];
    let mut next_slot = 2;
    if let Some(bias_name) = bias {
        buffers.push(
            BufferDecl::storage(bias_name, next_slot, BufferAccess::ReadOnly, dtype.clone())
                .with_count(n),
        );
        next_slot += 1;
    }
    if let ContractionEpilogue::QuantizedScale {
        row_scales,
        batch_scales,
    } = epilogue
    {
        buffers.push(
            BufferDecl::storage(row_scales, next_slot, BufferAccess::ReadOnly, dtype.clone())
                .with_count(m),
        );
        buffers.push(
            BufferDecl::storage(
                batch_scales,
                next_slot + 1,
                BufferAccess::ReadOnly,
                dtype.clone(),
            )
            .with_count(1),
        );
        next_slot += 2;
    }
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

    let region = if generator.starts_with("anonymous::") {
        wrap_anonymous_region(generator, body)
    } else {
        wrap_region(generator, body, None)
    };

    Ok(Program::wrapped(buffers, [lanes, 1, 1], vec![region]))
}
