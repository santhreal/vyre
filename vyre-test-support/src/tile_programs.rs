//! Programs built out of the tile statements.
//!
//! `Node::TileDecl`, `TileLoad`, `TileMatmul` and `TileStore` only mean
//! anything together, so a case that exercises one builds the whole `C = A x B`
//! shape. The lowering contract and the reference contract each built that
//! shape from scratch, differing only in element type, extents, and buffer
//! sizes, so a statement whose operand order changed had two places to be
//! updated and one of them would keep compiling.

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, Expr, Ident, Layout, Node, Program, Residency, SubgroupReduceOp, Tile,
};
use vyre_spec::DataType;

/// One tile operand: the tile and the element count of the buffer it moves
/// through.
#[derive(Clone, Debug)]
pub struct TileOperand {
    /// Element type, extents, layout and residency of the tile.
    pub tile: Tile,
    /// Element count of the buffer the tile is loaded from or stored to.
    pub elements: u32,
}

impl TileOperand {
    /// A tile of `element` with `extents`, loaded from a buffer of `elements`.
    #[must_use]
    pub fn new(
        element: DataType,
        extents: impl Into<Vec<u32>>,
        layout: Layout,
        residency: Residency,
        elements: u32,
    ) -> Self {
        Self {
            tile: Tile::new(element, extents, layout, residency),
            elements,
        }
    }
}

/// `out = left x right`, as tile statements over three buffers.
///
/// Buffers are `a` (read-only), `b` (read-only) and `out`, each declared with
/// its operand's element type and count. Each tile is loaded under its own
/// layout, because a load layout that disagreed with the tile would be testing
/// the transpose rather than the matmul.
#[must_use]
pub fn tile_matmul_program(
    workgroup: [u32; 3],
    left: &TileOperand,
    right: &TileOperand,
    accumulator: &TileOperand,
) -> Program {
    let left_layout = left.tile.layout.clone();
    let right_layout = right.tile.layout.clone();
    Program::wrapped(
        vec![
            BufferDecl::storage("a", 0, BufferAccess::ReadOnly, left.tile.element.clone())
                .with_count(left.elements),
            BufferDecl::storage("b", 1, BufferAccess::ReadOnly, right.tile.element.clone())
                .with_count(right.elements),
            BufferDecl::output("out", 2, accumulator.tile.element.clone())
                .with_count(accumulator.elements),
        ],
        workgroup,
        vec![
            Node::tile_decl("c", accumulator.tile.clone()),
            Node::tile_load(
                "t_a",
                left.tile.clone(),
                "a",
                vec![Expr::u32(0), Expr::u32(0)],
                left_layout,
            ),
            Node::tile_load(
                "t_b",
                right.tile.clone(),
                "b",
                vec![Expr::u32(0), Expr::u32(0)],
                right_layout,
            ),
            Node::tile_matmul("c", "t_a", "t_b"),
            Node::tile_store("out", vec![Expr::u32(0), Expr::u32(0)], "c"),
        ],
    )
}

/// The canonical matrix-fragment matmul: `c[16,8] = a[16,16] x b[16,8]`, F16
/// inputs accumulating in F32, on a full 32-lane workgroup.
///
/// This is the one shape a matrix-fragment case is about, and it is the
/// smallest tile matmul whose operands are wide enough for a fragment to be
/// packed out of more than one word. Every fragment case asserted against
/// exactly this program and each one restated all three operands, so an extent
/// or buffer count that changed had four places to change and each of them kept
/// compiling against the old shape.
///
/// `input_residency` is what decides whether a fragment is formed at all, and
/// `left_layout` is what decides whether the left operand is read transposed.
/// Those are the two facts a case varies, so they are the two arguments. The
/// accumulator stays register-resident row-major: a fragment accumulator that
/// is not register-resident is a different contract with a different owner.
#[must_use]
pub fn fragment_matmul_program(input_residency: Residency, left_layout: Layout) -> Program {
    tile_matmul_program(
        [32, 1, 1],
        &TileOperand::new(
            DataType::F16,
            vec![16, 16],
            left_layout,
            input_residency,
            256,
        ),
        &TileOperand::new(
            DataType::F16,
            vec![16, 8],
            Layout::ColumnMajor,
            input_residency,
            128,
        ),
        &TileOperand::new(
            DataType::F32,
            vec![16, 8],
            Layout::RowMajor,
            Residency::Register,
            128,
        ),
    )
}

/// One tile program with the buffers it reads and the output it must produce.
pub struct TileCase {
    /// Case name, reported by whichever suite failed on it.
    pub name: &'static str,
    /// The program under test.
    pub program: Program,
    /// One `f32` buffer per read-only binding, in slot order.
    pub inputs: Vec<Vec<f32>>,
    /// The `out` buffer contents the program must produce.
    pub expected: Vec<f32>,
}

/// Every tile program whose result is fixed by the tile statements' semantics.
///
/// The lowering contract claims that a lowered tile program reproduces the
/// reference oracle index for index, and the reference contract claims the
/// oracle computes the stated values. Both claims were asserted against
/// programs each suite built for itself, so the two sides of the comparison
/// were never the same program: the lowering suite covered a load/store round
/// trip the oracle never ran, the oracle covered a column-major load and a
/// scaling elementwise the lowering suite never lowered, and the three cases
/// they did share had drifted apart in buffer names. One corpus makes the
/// comparison mean what it says, and a case added here is answered by both
/// sides at once.
///
/// Every case reads and writes `f32` so one decode covers all of them, and
/// every expected value is computed by hand from the tile statements rather
/// than recorded from a run.
#[must_use]
pub fn tile_cases() -> Vec<TileCase> {
    let square = |extents: Vec<u32>| {
        Tile::new(
            DataType::F32,
            extents,
            Layout::RowMajor,
            Residency::Register,
        )
    };
    vec![
        TileCase {
            name: "load_store_roundtrip",
            program: Program::wrapped(
                vec![
                    BufferDecl::storage("a", 0, BufferAccess::ReadOnly, DataType::F32)
                        .with_count(4),
                    BufferDecl::output("out", 1, DataType::F32).with_count(4),
                ],
                [1, 1, 1],
                vec![
                    Node::tile_load(
                        "t",
                        square(vec![2, 2]),
                        "a",
                        vec![Expr::u32(0), Expr::u32(0)],
                        Layout::RowMajor,
                    ),
                    Node::tile_store("out", vec![Expr::u32(0), Expr::u32(0)], "t"),
                ],
            ),
            inputs: vec![vec![1.0, 2.0, 3.0, 4.0]],
            expected: vec![1.0, 2.0, 3.0, 4.0],
        },
        TileCase {
            name: "matmul_2x2",
            program: tile_matmul_program(
                [1, 1, 1],
                &TileOperand::new(
                    DataType::F32,
                    vec![2, 2],
                    Layout::RowMajor,
                    Residency::Register,
                    4,
                ),
                &TileOperand::new(
                    DataType::F32,
                    vec![2, 2],
                    Layout::RowMajor,
                    Residency::Register,
                    4,
                ),
                &TileOperand::new(
                    DataType::F32,
                    vec![2, 2],
                    Layout::RowMajor,
                    Residency::Register,
                    4,
                ),
            ),
            inputs: vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]],
            expected: vec![19.0, 22.0, 43.0, 50.0],
        },
        TileCase {
            name: "reduce_axis_1",
            program: Program::wrapped(
                vec![
                    BufferDecl::storage("a", 0, BufferAccess::ReadOnly, DataType::F32)
                        .with_count(4),
                    BufferDecl::output("out", 1, DataType::F32).with_count(2),
                ],
                [1, 1, 1],
                vec![
                    Node::tile_load(
                        "t_a",
                        square(vec![2, 2]),
                        "a",
                        vec![Expr::u32(0), Expr::u32(0)],
                        Layout::RowMajor,
                    ),
                    Node::tile_reduce("max_per_row", "t_a", SubgroupReduceOp::Max, 1),
                    Node::tile_store("out", vec![Expr::u32(0)], "max_per_row"),
                ],
            ),
            inputs: vec![vec![1.0, 5.0, 2.0, 8.0]],
            expected: vec![5.0, 8.0],
        },
        TileCase {
            name: "broadcast_elementwise",
            program: Program::wrapped(
                vec![
                    BufferDecl::storage("a", 0, BufferAccess::ReadOnly, DataType::F32)
                        .with_count(4),
                    BufferDecl::output("out", 1, DataType::F32).with_count(4),
                ],
                [1, 1, 1],
                vec![
                    Node::tile_load(
                        "t_a",
                        square(vec![2, 2]),
                        "a",
                        vec![Expr::u32(0), Expr::u32(0)],
                        Layout::RowMajor,
                    ),
                    Node::tile_reduce("row_max", "t_a", SubgroupReduceOp::Max, 1),
                    Node::tile_elementwise(
                        "diff",
                        vec![Ident::from("t_a"), Ident::from("row_max")],
                        vec![Node::let_bind(
                            "diff",
                            Expr::sub(Expr::var("t_a"), Expr::var("row_max")),
                        )],
                    ),
                    Node::tile_store("out", vec![Expr::u32(0)], "diff"),
                ],
            ),
            inputs: vec![vec![10.0, 20.0, 30.0, 40.0]],
            expected: vec![-10.0, 0.0, -10.0, 0.0],
        },
        TileCase {
            name: "elementwise_scaling",
            program: Program::wrapped(
                vec![
                    BufferDecl::storage("a", 0, BufferAccess::ReadOnly, DataType::F32)
                        .with_count(4),
                    BufferDecl::output("out", 1, DataType::F32).with_count(4),
                ],
                [1, 1, 1],
                vec![
                    Node::tile_load(
                        "t_a",
                        square(vec![4]),
                        "a",
                        vec![Expr::u32(0)],
                        Layout::RowMajor,
                    ),
                    Node::tile_elementwise(
                        "scaled",
                        vec![Ident::from("t_a")],
                        vec![Node::let_bind(
                            "scaled",
                            Expr::mul(Expr::var("t_a"), Expr::f32(3.0)),
                        )],
                    ),
                    Node::tile_store("out", vec![Expr::u32(0)], "scaled"),
                ],
            ),
            inputs: vec![vec![2.0, 4.0, 6.0, 8.0]],
            expected: vec![6.0, 12.0, 18.0, 24.0],
        },
        TileCase {
            name: "column_major_load",
            program: Program::wrapped(
                vec![
                    BufferDecl::storage("a", 0, BufferAccess::ReadOnly, DataType::F32)
                        .with_count(4),
                    BufferDecl::output("out", 1, DataType::F32).with_count(4),
                ],
                [1, 1, 1],
                vec![
                    Node::tile_load(
                        "t_col",
                        Tile::new(
                            DataType::F32,
                            vec![2, 2],
                            Layout::ColumnMajor,
                            Residency::Register,
                        ),
                        "a",
                        vec![Expr::u32(0), Expr::u32(0)],
                        Layout::ColumnMajor,
                    ),
                    Node::tile_store("out", vec![Expr::u32(0)], "t_col"),
                ],
            ),
            inputs: vec![vec![1.0, 2.0, 3.0, 4.0]],
            expected: vec![1.0, 3.0, 2.0, 4.0],
        },
    ]
}
