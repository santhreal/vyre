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

use vyre_foundation::ir::model::expr::GeneratorRef;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use super::cell_grid::{cell_lookup_nodes, GridShape};

const OP_ID: &str = "vyre-libs::visual::glyph_grid";

/// Extract one 8-bit channel from a packed RGBA word.
///
/// The alpha channel needs no mask because nothing remains above it.
fn channel(pixel: &str, shift: u32) -> Expr {
    let shifted = if shift == 0 {
        Expr::var(pixel)
    } else {
        Expr::shr(Expr::var(pixel), Expr::u32(shift))
    };
    if shift == 24 {
        shifted
    } else {
        Expr::bitand(shifted, Expr::u32(0xFF))
    }
}

/// Blend one channel of the foreground over the background by `coverage`.
///
/// `out = (fg * coverage + bg * (255 - coverage)) / 255`, with the division by
/// 255 taken as `(v + 128) * 257 >> 16`, the same approximation the Porter-Duff
/// composite op uses. The numerator peaks at `255 * 255`, since the two weights
/// sum to 255, so the result saturates at exactly 255 and needs no clamp.
fn blend_channel(shift: u32) -> Expr {
    super::wide_mul_shr_u32(
        Expr::add(
            Expr::add(
                Expr::mul(channel("fg_px", shift), Expr::var("cov")),
                Expr::mul(channel("bg_px", shift), Expr::var("inv_cov")),
            ),
            Expr::u32(128),
        ),
        Expr::u32(257),
        16,
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

    Program::wrapped(
        vec![
            BufferDecl::storage(glyphs, 0, BufferAccess::ReadOnly, DataType::U32).with_count(cells),
            BufferDecl::storage(fg, 1, BufferAccess::ReadOnly, DataType::U32).with_count(cells),
            BufferDecl::storage(bg, 2, BufferAccess::ReadOnly, DataType::U32).with_count(cells),
            BufferDecl::storage(atlas, 3, BufferAccess::ReadOnly, DataType::U32)
                .with_count(atlas_texels),
            BufferDecl::storage(output, 4, BufferAccess::ReadWrite, DataType::U32)
                .with_count(pixels),
        ],
        super::PIXEL_WORKGROUP_SIZE,
        vec![crate::region::wrap_anonymous(
            OP_ID,
            vec![crate::region::wrap_child(
                vyre_primitives::visual::packed_rgba_map::OP_ID,
                GeneratorRef {
                    name: OP_ID.to_string(),
                },
                vec![
                    Node::let_bind("idx", Expr::gid_x()),
                    Node::if_then(Expr::lt(Expr::var("idx"), Expr::u32(pixels)), {
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
                                Expr::add(
                                    Expr::mul(Expr::var("py"), Expr::u32(shape.cell_width)),
                                    Expr::var("px"),
                                ),
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
                            Expr::bitor(
                                Expr::bitor(
                                    Expr::var("out_r"),
                                    Expr::shl(Expr::var("out_g"), Expr::u32(8)),
                                ),
                                Expr::bitor(
                                    Expr::shl(Expr::var("out_b"), Expr::u32(16)),
                                    Expr::shl(Expr::var("out_a"), Expr::u32(24)),
                                ),
                            ),
                        ));
                        body.push(Node::store(output, Expr::var("idx"), Expr::var("packed")));
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
                crate::visual::byte_helpers::u32_words_to_le_bytes(&glyphs),
                crate::visual::byte_helpers::u32_words_to_le_bytes(&fg),
                crate::visual::byte_helpers::u32_words_to_le_bytes(&bg),
                crate::visual::byte_helpers::u32_words_to_le_bytes(&atlas),
                vec![0u8; 4 * 4],
            ]]
        }),
        Some(|| {
            // Coverage 0 leaves the background, 255 takes the foreground, and
            // 128 lands between them:
            //   r = (255*128 + 0*127 + 128) * 257 >> 16 = 128
            //   b = (0*128 + 255*127 + 128) * 257 >> 16 = 127
            //   a = (255*128 + 255*127 + 128) * 257 >> 16 = 255
            let expected = [0xFFFF_0000u32, 0xFF00_00FF, 0xFF7F_0080, 0xFF00_00FF];
            vec![vec![crate::visual::byte_helpers::u32_words_to_le_bytes(&expected)]]
        }),
    )
    .with_category("visual")
}
