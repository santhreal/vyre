//! Glyph compositing over a character-cell grid.
//!
//! The other half of a terminal renderer. [`cell_grid_fill`] paints each cell
//! a flat colour; this op samples a glyph coverage atlas and blends the cell's
//! foreground over its background by that coverage, so one invocation still
//! produces exactly one packed RGBA pixel.
//!
//! The host uploads three `u32` per cell (glyph index, foreground, background)
//! and never touches a pixel. The atlas is uploaded once and changes only when
//! the font or the cell size does.
//!
//! Category A composition  -  pure IR over existing expressions, specializing
//! the Tier 2.5 `packed_rgba_map` shape.
//!
//! [`cell_grid_fill`]: crate::visual::cell_grid::cell_grid_fill

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use super::cell_grid::{cell_lookup_nodes, GridShape};

const OP_ID: &str = "vyre-libs::visual::glyph_grid";

/// Extract one 8-bit channel from a packed RGBA word.
///
/// The alpha channel needs no mask because nothing remains above it.
fn channel(pixel: &str, shift: u32) -> Expr {
    crate::builder::stencil::unpack_channel(pixel, shift)
}

/// Blend one channel of the foreground over the background by `coverage`.
///
/// `out = (fg * coverage + bg * (255 - coverage)) / 255`, with the division by
/// 255 taken as `(v + 128) * 257 >> 16`, the same approximation the Porter-Duff
/// composite op uses. The numerator peaks at `255 * 255`, since the two weights
/// sum to 255, so the result saturates at exactly 255 and needs no clamp.
fn blend_channel(shift: u32) -> Expr {
    crate::builder::stencil::blend_channel_coverage(
        channel("fg_px", shift),
        channel("bg_px", shift),
        Expr::var("cov"),
        Expr::var("inv_cov"),
    )
}

/// Build a Program that renders a grid of glyph cells into packed RGBA pixels.
///
/// `glyphs`, `fg` and `bg` are each `[u32; cols * rows]` in row-major order:
/// the glyph index, the foreground colour and the background colour of every
/// cell. `atlas` is `[u32; glyph_count * cell_width * cell_height]`, glyph
/// major, holding one 8-bit coverage value per texel. `output` is
/// `[u32; width * height]`.
///
/// # Panics
///
/// If the grid is degenerate, or if the atlas would be too large to index.
#[must_use]
pub fn glyph_grid_blend(
    glyphs: &str,
    fg: &str,
    bg: &str,
    atlas: &str,
    output: &str,
    shape: GridShape,
    glyph_count: u32,
) -> Program {
    let shape = shape.validated();
    assert!(
        glyph_count > 0,
        "Fix: a glyph atlas needs at least one glyph"
    );
    let cell_area = shape
        .cell_width
        .checked_mul(shape.cell_height)
        .expect("Fix: cell_width * cell_height overflows u32");
    let atlas_texels = glyph_count
        .checked_mul(cell_area)
        .expect("Fix: glyph_count * cell area overflows u32");
    let pixels = shape.pixel_count();
    let cells = shape.cell_count();

    let mut body = cell_lookup_nodes(shape);
    // Position within the cell, reusing the column and row
    // the lookup already resolved.
    body.push(Node::let_bind(
        "px",
        Expr::sub(
            Expr::var("x"),
            Expr::mul(Expr::var("col"), Expr::u32(shape.cell_width)),
        ),
    ));
    body.push(Node::let_bind(
        "py",
        Expr::sub(
            Expr::var("y"),
            Expr::mul(Expr::var("row"), Expr::u32(shape.cell_height)),
        ),
    ));
    body.push(Node::let_bind(
        "glyph",
        Expr::load(glyphs, Expr::var("cell")),
    ));
    // The atlas is glyph major, so a glyph's texels are
    // contiguous and a cell reads one cache line's worth
    // of them rather than striding the whole atlas.
    body.push(Node::let_bind(
        "texel",
        Expr::add(
            Expr::mul(Expr::var("glyph"), Expr::u32(cell_area)),
            crate::builder::stencil::flat_index(Expr::var("py"), shape.cell_width, Expr::var("px")),
        ),
    ));
    body.push(Node::let_bind(
        "cov",
        Expr::bitand(Expr::load(atlas, Expr::var("texel")), Expr::u32(0xFF)),
    ));
    body.push(Node::let_bind(
        "inv_cov",
        Expr::sub(Expr::u32(255), Expr::var("cov")),
    ));
    body.push(Node::let_bind("fg_px", Expr::load(fg, Expr::var("cell"))));
    body.push(Node::let_bind("bg_px", Expr::load(bg, Expr::var("cell"))));
    body.push(Node::let_bind("out_r", blend_channel(0)));
    body.push(Node::let_bind("out_g", blend_channel(8)));
    body.push(Node::let_bind("out_b", blend_channel(16)));
    body.push(Node::let_bind("out_a", blend_channel(24)));
    body.push(Node::let_bind(
        "packed",
        crate::builder::stencil::pack_rgba_named("out_r", "out_g", "out_b", "out_a"),
    ));
    body.push(Node::store(output, Expr::var("idx"), Expr::var("packed")));

    crate::visual::packed_rgba_map::build_pixel_pipeline(
        OP_ID,
        vec![
            BufferDecl::storage(glyphs, 0, BufferAccess::ReadOnly, DataType::U32).with_count(cells),
            BufferDecl::storage(fg, 1, BufferAccess::ReadOnly, DataType::U32).with_count(cells),
            BufferDecl::storage(bg, 2, BufferAccess::ReadOnly, DataType::U32).with_count(cells),
            BufferDecl::storage(atlas, 3, BufferAccess::ReadOnly, DataType::U32)
                .with_count(atlas_texels),
            BufferDecl::storage(output, 4, BufferAccess::ReadWrite, DataType::U32)
                .with_count(pixels),
        ],
        pixels,
        body,
    )
}

const EXPECTED_GLYPH_GRID_OUTPUT_BYTES: [u8; 16] = [
    0x00, 0x00, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0x80, 0x00, 0x7F, 0xFF, 0xFF, 0x00, 0x00, 0xFF,
];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || {
            glyph_grid_blend(
                "glyphs",
                "fg",
                "bg",
                "atlas",
                "out",
                GridShape { cols: 1, rows: 1, cell_width: 2, cell_height: 2 },
                2,
            )
        },
        Some(|| {
            // One cell showing glyph 1: red foreground over a blue background.
            // Glyph 0 is blank, glyph 1 covers three of the four texels, one
            // of them only half.
            let glyphs = [1u32];
            let fg = [0xFF00_00FFu32];
            let bg = [0xFFFF_0000u32];
            let atlas = [0u32, 0, 0, 0, 0, 255, 128, 255];
            vec![vec![
                crate::visual::u32_word_bytes::u32_words_to_le_bytes(&glyphs),
                crate::visual::u32_word_bytes::u32_words_to_le_bytes(&fg),
                crate::visual::u32_word_bytes::u32_words_to_le_bytes(&bg),
                crate::visual::u32_word_bytes::u32_words_to_le_bytes(&atlas),
                vec![0u8; 4 * 4],
            ]]
        }),
        Some(|| {
            vec![vec![EXPECTED_GLYPH_GRID_OUTPUT_BYTES.to_vec()]]
        }),
    )
    .with_category("visual")
}
