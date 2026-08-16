//! Character-cell grid expansion.
//!
//! One invocation derives one packed `u32` RGBA pixel from the cell that
//! covers it. A terminal surface is a small grid of cells and a large field
//! of pixels: 80x24 cells behind 800x456 pixels. Expanding the grid on the
//! host means building one rectangle per cell every frame and handing the
//! renderer tens of thousands of them; expanding it here means the host
//! writes one `u32` per cell that changed and nothing else.
//!
//! Category A composition  -  pure IR over existing expressions, specializing
//! the Tier 2.5 `packed_rgba_map` shape. No new IR variant, no target
//! lowering.

use vyre_foundation::composition::{wrap_anonymous_region, wrap_child_region};
use vyre_foundation::ir::Ident;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

const OP_ID: &str = "vyre-libs::visual::cell_grid";

/// Pixel dimensions a cell grid expands to.
///
/// Kept as one value so a caller cannot pass the four numbers in the wrong
/// order without saying which is which.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GridShape {
    /// Cells across.
    pub cols: u32,
    /// Cells down.
    pub rows: u32,
    /// Pixels across one cell.
    pub cell_width: u32,
    /// Pixels down one cell.
    pub cell_height: u32,
}

impl GridShape {
    /// Pixels across the whole surface.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.cols * self.cell_width
    }

    /// Pixels down the whole surface.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.rows * self.cell_height
    }

    /// Cells in the grid.
    #[must_use]
    pub const fn cell_count(&self) -> u32 {
        self.cols * self.rows
    }

    /// Pixels in the surface.
    #[must_use]
    pub const fn pixel_count(&self) -> u32 {
        self.width() * self.height()
    }

    pub(super) fn validated(self) -> Self {
        assert!(
            self.cols > 0 && self.rows > 0,
            "Fix: a cell grid needs at least one row and one column, got {}x{}",
            self.cols,
            self.rows
        );
        assert!(
            self.cell_width > 0 && self.cell_height > 0,
            "Fix: a cell needs a non-zero size, got {}x{} pixels",
            self.cell_width,
            self.cell_height
        );
        // Every product below is computed once, here, so an overflow is a
        // named build-time failure rather than a wrapped count that silently
        // sizes a buffer too small.
        let width = self
            .cols
            .checked_mul(self.cell_width)
            .expect("Fix: cols * cell_width overflows u32");
        let height = self
            .rows
            .checked_mul(self.cell_height)
            .expect("Fix: rows * cell_height overflows u32");
        width
            .checked_mul(height)
            .expect("Fix: the surface pixel count overflows u32");
        self.cols
            .checked_mul(self.rows)
            .expect("Fix: cols * rows overflows u32");
        self
    }
}

/// Bind `y`, `x`, `col`, `row` and `cell` for the pixel already bound as
/// `idx`. Shared by every op that expands a cell grid, so the mapping cannot
/// drift between them.
///
/// Every divisor is a build-time constant, so the Layer 1 strength-reduction
/// pass turns each division into mulhi plus a shift. Writing the division is
/// the correct thing to write.
pub(super) fn cell_lookup_nodes(shape: GridShape) -> Vec<Node> {
    let width = shape.width();
    vec![
        Node::let_bind("y", Expr::div(Expr::var("idx"), Expr::u32(width))),
        // x = idx - y * width, not idx % width. The remainder lowers to the
        // same division just performed, plus a multiply and a subtract, so
        // reusing y keeps one division for the pair.
        Node::let_bind(
            "x",
            Expr::sub(
                Expr::var("idx"),
                Expr::mul(Expr::var("y"), Expr::u32(width)),
            ),
        ),
        Node::let_bind(
            "col",
            Expr::div(Expr::var("x"), Expr::u32(shape.cell_width)),
        ),
        Node::let_bind(
            "row",
            Expr::div(Expr::var("y"), Expr::u32(shape.cell_height)),
        ),
        Node::let_bind(
            "cell",
            Expr::add(
                Expr::mul(Expr::var("row"), Expr::u32(shape.cols)),
                Expr::var("col"),
            ),
        ),
    ]
}

/// Build a Program that fills `output` with one packed RGBA pixel per pixel of
/// the surface, taking each pixel's colour from the cell that covers it.
///
/// `cells` is `[u32; cols * rows]` in row-major order, one packed RGBA colour
/// per cell. `output` is `[u32; width * height]`, also row-major.
#[must_use]
pub fn cell_grid_fill(cells: &str, output: &str, shape: GridShape) -> Program {
    let shape = shape.validated();
    let pixels = shape.pixel_count();
    let width = shape.width();

    Program::wrapped(
        vec![
            BufferDecl::storage(cells, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(shape.cell_count()),
            BufferDecl::storage(output, 1, BufferAccess::ReadWrite, DataType::U32)
                .with_count(pixels),
        ],
        super::PIXEL_WORKGROUP_SIZE,
        vec![wrap_anonymous_region(
            OP_ID,
            vec![wrap_child_region(
                crate::visual::packed_rgba_map::OP_ID,
                Ident::from(OP_ID),
                vec![
                    Node::let_bind("idx", Expr::gid_x()),
                    Node::if_then(Expr::lt(Expr::var("idx"), Expr::u32(pixels)), {
                        let mut body = cell_lookup_nodes(shape);
                        body.push(Node::let_bind(
                            "colour",
                            Expr::load(cells, Expr::var("cell")),
                        ));
                        body.push(Node::store(output, Expr::var("idx"), Expr::var("colour")));
                        body
                    }),
                ],
            )],
        )],
    )
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || {
            cell_grid_fill(
                "cells",
                "out",
                GridShape { cols: 2, rows: 2, cell_width: 2, cell_height: 2 },
            )
        },
        Some(|| {
            // Four cells behind a 4x4 surface: red, green on the top row,
            // blue, white on the bottom. Packing is little-endian RGBA, so
            // bits [7:0] are red and [31:24] are alpha.
            let cells = [0xFF00_00FFu32, 0xFF00_FF00, 0xFFFF_0000, 0xFFFF_FFFF];
            vec![vec![
                crate::visual::u32_word_bytes::u32_words_to_le_bytes(&cells),
                vec![0u8; 16 * 4],
            ]]
        }),
        Some(|| {
            // Each cell covers a 2x2 block, so every colour appears four
            // times, in a quadrant rather than a run.
            const R: u32 = 0xFF00_00FF;
            const G: u32 = 0xFF00_FF00;
            const B: u32 = 0xFFFF_0000;
            const W: u32 = 0xFFFF_FFFF;
            let expected = [
                R, R, G, G,
                R, R, G, G,
                B, B, W, W,
                B, B, W, W,
            ];
            vec![vec![crate::visual::u32_word_bytes::u32_words_to_le_bytes(&expected)]]
        }),
    )
    .with_category("visual")
}
