//! Programs built out of the tile statements.
//!
//! `Node::TileDecl`, `TileLoad`, `TileMatmul` and `TileStore` only mean
//! anything together, so a case that exercises one builds the whole `C = A x B`
//! shape. The lowering contract and the reference contract each built that
//! shape from scratch, differing only in element type, extents, and buffer
//! sizes, so a statement whose operand order changed had two places to be
//! updated and one of them would keep compiling.

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, Expr, Layout, Node, Program, Residency, Tile,
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
