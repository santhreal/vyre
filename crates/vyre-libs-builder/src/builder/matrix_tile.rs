//! Cooperative matrix tiling: the geometry of an output tile and the staging
//! loop that fills it.
//!
//! A workgroup stages one `k_tile` slab of each operand into workgroup memory,
//! accumulates every output the tile owns against that slab, and moves to the
//! next slab. Each operand element is read once per tile edge instead of once
//! per output, which is the whole reason a tiled contraction is faster than
//! the linear one.
//!
//! The staging loop carries no arithmetic of its own. A caller supplies the
//! pad value an uncovered lane takes, and the two operations that combine a
//! pair of staged values and fold the result into the accumulator, so a plain
//! product and a semiring contraction share one loop.

use vyre_foundation::ir::{Expr, MemoryOrdering, Node};

use crate::plumbing::operand::tensor_ref::{element_count, TensorRefError};

/// 2D matrix shape `(m, k, n)` for matrix multiplication.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct MatrixShape {
    /// Left matrix rows and output rows.
    pub m: u32,
    /// Shared contraction dimension.
    pub k: u32,
    /// Right matrix columns and output columns.
    pub n: u32,
}

/// Geometry of the output tile one workgroup owns.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct TileShape {
    /// Contraction elements staged per slab.
    pub k_tile: u32,
    /// Output rows the tile covers.
    pub out_rows: u32,
    /// Output columns the tile covers.
    pub out_cols: u32,
    /// Workgroup lanes along the column axis.
    pub x_lanes: u32,
    /// Workgroup lanes along the row axis.
    pub y_lanes: u32,
    /// Lanes in the workgroup.
    pub lanes: u32,
    /// Left-operand elements staged per slab.
    pub a_values: u32,
    /// Right-operand elements staged per slab.
    pub b_values: u32,
}

/// Output columns, output rows and lane count a workgroup shape implies.
///
/// The x axis walks output columns and the remaining two axes walk output
/// rows, so a workgroup of `[x, y, z]` owns `x` columns of `y * z` rows.
pub fn output_tile_shape(workgroup: [u32; 3]) -> Result<(u32, u32, u32), TensorRefError> {
    let out_cols = workgroup[0].max(1);
    let out_rows = workgroup[1].max(1).saturating_mul(workgroup[2].max(1));
    let lanes = element_count("matmul_workgroup", &[out_rows, out_cols])?;
    Ok((out_cols, out_rows, lanes))
}

/// Lanes the launch declares once every partially covered tile is padded to a
/// whole one.
pub fn padded_tile_lane_count(
    m: u32,
    n: u32,
    out_rows: u32,
    out_cols: u32,
    lanes: u32,
) -> Result<u32, TensorRefError> {
    let row_tiles = m.div_ceil(out_rows);
    let col_tiles = n.div_ceil(out_cols);
    element_count("matmul_tiled_launch_lanes", &[row_tiles, col_tiles, lanes])
}

/// True where `(row, col)` names an element the output declares.
#[must_use]
pub fn in_output_bounds(row: Expr, col: Expr, shape: MatrixShape) -> Expr {
    Expr::and(
        Expr::lt(row, Expr::u32(shape.m)),
        Expr::lt(col, Expr::u32(shape.n)),
    )
}

/// Variable names a caller wants the output-tile coordinates bound to.
pub struct OutputTileCoordNames {
    /// Lane row within the tile.
    pub lane_row: &'static str,
    /// Lane column within the tile.
    pub lane_col: &'static str,
    /// Output row the lane owns.
    pub row: &'static str,
    /// Output column the lane owns.
    pub col: &'static str,
}

/// Bind the lane's position in its tile and the output element it owns.
///
/// The bindings are `local`, `tile_block`, `tile_cols`, `tile_row_base`,
/// `tile_col_base` and the four `names` carry, in that order.
#[must_use]
pub fn bind_output_tile_coordinates(
    shape: MatrixShape,
    tile: TileShape,
    names: OutputTileCoordNames,
) -> Vec<Node> {
    let local = Expr::var("local");
    vec![
        Node::let_bind(
            "local",
            Expr::add(
                Expr::add(
                    Expr::LogicalWithinTileId { axis: 0 },
                    Expr::mul(
                        Expr::LogicalWithinTileId { axis: 1 },
                        Expr::u32(tile.x_lanes),
                    ),
                ),
                Expr::mul(
                    Expr::LogicalWithinTileId { axis: 2 },
                    Expr::u32(tile.x_lanes.saturating_mul(tile.y_lanes)),
                ),
            ),
        ),
        Node::let_bind("tile_block", Expr::LogicalTileId { axis: 0 }),
        Node::let_bind("tile_cols", Expr::u32(shape.n.div_ceil(tile.out_cols))),
        Node::let_bind(
            "tile_row_base",
            Expr::mul(
                Expr::div(Expr::var("tile_block"), Expr::var("tile_cols")),
                Expr::u32(tile.out_rows),
            ),
        ),
        Node::let_bind(
            "tile_col_base",
            Expr::mul(
                Expr::rem(Expr::var("tile_block"), Expr::var("tile_cols")),
                Expr::u32(tile.out_cols),
            ),
        ),
        Node::let_bind(
            names.lane_row,
            Expr::div(local.clone(), Expr::u32(tile.out_cols)),
        ),
        Node::let_bind(names.lane_col, Expr::rem(local, Expr::u32(tile.out_cols))),
        Node::let_bind(
            names.row,
            Expr::add(Expr::var("tile_row_base"), Expr::var(names.lane_row)),
        ),
        Node::let_bind(
            names.col,
            Expr::add(Expr::var("tile_col_base"), Expr::var(names.lane_col)),
        ),
    ]
}

/// The arithmetic a staged pair of values goes through.
pub struct TileArithmetic<'a> {
    /// Value an uncovered staging lane holds, so a partial slab contributes
    /// the identity rather than whatever the buffer last held.
    pub pad: Expr,
    /// Combine one staged left value with one staged right value.
    pub combine: &'a dyn Fn(Expr, Expr) -> Expr,
    /// Fold a combined value into the accumulator.
    pub accumulate: &'a dyn Fn(Expr, Expr) -> Expr,
}

/// The staging loop: for each `k_tile` slab, fill both workgroup tiles, then
/// accumulate the slab into `acc`.
///
/// The caller owns `acc` and the output store. Both barriers are inside the
/// slab loop, so a lane never reads a tile another lane is still filling and
/// never refills one another lane is still reading.
#[must_use]
pub fn cooperative_slab_loop(
    a: &str,
    b: &str,
    a_tile_name: &str,
    b_tile_name: &str,
    shape: MatrixShape,
    tile: TileShape,
    in_bounds: &Expr,
    arithmetic: &TileArithmetic<'_>,
) -> Node {
    let slab_count = shape.k.div_ceil(tile.k_tile);
    let load_passes = tile.a_values.max(tile.b_values).div_ceil(tile.lanes).max(1);
    let local = Expr::var("local");

    Node::loop_for(
        "tile_idx",
        Expr::u32(0),
        Expr::u32(slab_count),
        vec![
            Node::let_bind(
                "k_base",
                Expr::mul(Expr::var("tile_idx"), Expr::u32(tile.k_tile)),
            ),
            Node::loop_for(
                "load_pass",
                Expr::u32(0),
                Expr::u32(load_passes),
                stage_operand(
                    StagedOperand {
                        source: a,
                        tile_name: a_tile_name,
                        linear: "a_linear",
                        local_major: "a_local_row",
                        local_minor: "a_local_k",
                        major: "a_row",
                        minor: "a_k",
                        major_base: "tile_row_base",
                        minor_base: "k_base",
                        values: tile.a_values,
                        minor_extent: tile.k_tile,
                        major_limit: shape.m,
                        minor_limit: shape.k,
                        row_stride: shape.k,
                    },
                    tile.lanes,
                    &local,
                    &arithmetic.pad,
                )
                .into_iter()
                .chain(stage_operand(
                    StagedOperand {
                        source: b,
                        tile_name: b_tile_name,
                        linear: "b_linear",
                        local_major: "b_local_k",
                        local_minor: "b_local_col",
                        major: "b_k",
                        minor: "b_col",
                        major_base: "k_base",
                        minor_base: "tile_col_base",
                        values: tile.b_values,
                        minor_extent: tile.out_cols,
                        major_limit: shape.k,
                        minor_limit: shape.n,
                        row_stride: shape.n,
                    },
                    tile.lanes,
                    &local,
                    &arithmetic.pad,
                ))
                .collect(),
            ),
            Node::logical_barrier(MemoryOrdering::SeqCst),
            Node::loop_for(
                "tile_k",
                Expr::u32(0),
                Expr::u32(tile.k_tile),
                vec![Node::if_then(
                    in_bounds.clone(),
                    vec![Node::assign(
                        "acc",
                        (arithmetic.accumulate)(
                            Expr::var("acc"),
                            (arithmetic.combine)(
                                Expr::load(
                                    a_tile_name,
                                    Expr::add(
                                        Expr::mul(Expr::var("local_row"), Expr::u32(tile.k_tile)),
                                        Expr::var("tile_k"),
                                    ),
                                ),
                                Expr::load(
                                    b_tile_name,
                                    Expr::add(
                                        Expr::mul(Expr::var("tile_k"), Expr::u32(tile.out_cols)),
                                        Expr::var("local_col"),
                                    ),
                                ),
                            ),
                        ),
                    )],
                )],
            ),
            Node::logical_barrier(MemoryOrdering::SeqCst),
        ],
    )
}

/// One operand's staging pass: which buffer it reads, which workgroup tile it
/// fills, and the extents that decide whether a lane covers real data.
struct StagedOperand<'a> {
    source: &'a str,
    tile_name: &'a str,
    linear: &'static str,
    local_major: &'static str,
    local_minor: &'static str,
    major: &'static str,
    minor: &'static str,
    major_base: &'static str,
    minor_base: &'static str,
    values: u32,
    minor_extent: u32,
    major_limit: u32,
    minor_limit: u32,
    row_stride: u32,
}

/// Stage one operand slab: pad every lane of the tile, then overwrite the
/// lanes the source matrix covers.
///
/// The pad store is unconditional because a tile that overhangs the matrix
/// keeps whatever the previous slab wrote there otherwise, and that value
/// would enter the next accumulation.
fn stage_operand(operand: StagedOperand<'_>, lanes: u32, local: &Expr, pad: &Expr) -> Vec<Node> {
    let linear = Expr::var(operand.linear);
    vec![
        Node::let_bind(
            operand.linear,
            Expr::add(
                local.clone(),
                Expr::mul(Expr::var("load_pass"), Expr::u32(lanes)),
            ),
        ),
        Node::if_then(
            Expr::lt(linear.clone(), Expr::u32(operand.values)),
            vec![
                Node::let_bind(
                    operand.local_major,
                    Expr::div(linear.clone(), Expr::u32(operand.minor_extent)),
                ),
                Node::let_bind(
                    operand.local_minor,
                    Expr::rem(linear.clone(), Expr::u32(operand.minor_extent)),
                ),
                Node::let_bind(
                    operand.major,
                    Expr::add(
                        Expr::var(operand.major_base),
                        Expr::var(operand.local_major),
                    ),
                ),
                Node::let_bind(
                    operand.minor,
                    Expr::add(
                        Expr::var(operand.minor_base),
                        Expr::var(operand.local_minor),
                    ),
                ),
                Node::Store {
                    buffer: operand.tile_name.into(),
                    index: linear.clone(),
                    value: pad.clone(),
                },
                Node::if_then(
                    Expr::and(
                        Expr::lt(Expr::var(operand.major), Expr::u32(operand.major_limit)),
                        Expr::lt(Expr::var(operand.minor), Expr::u32(operand.minor_limit)),
                    ),
                    vec![Node::Store {
                        buffer: operand.tile_name.into(),
                        index: linear,
                        value: Expr::load(
                            operand.source,
                            Expr::add(
                                Expr::mul(Expr::var(operand.major), Expr::u32(operand.row_stride)),
                                Expr::var(operand.minor),
                            ),
                        ),
                    }],
                ),
            ],
        ),
    ]
}
