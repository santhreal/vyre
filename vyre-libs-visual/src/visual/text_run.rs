//! Subpixel and arbitrary-position text run rasterization.
//!
//! Accumulates glyph instances from a shared font/glyph atlas into a destination
//! framebuffer with subpixel positioning and Porter-Duff alpha coverage.
//!
//! Category A composition - pure IR over existing expressions.

use vyre_foundation::composition::{wrap_anonymous_region, wrap_child_region};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Ident, Node, Program};

const OP_ID: &str = "vyre-libs::visual::text_run";

/// Build a Program that rasterizes arbitrary text runs from a glyph atlas.
///
/// `glyphs` is `[u32; glyph_count * 7]` in `[gx, gy, gw, gh, atlas_u, atlas_v, color]` format.
/// `atlas` is `[u32; atlas_w * atlas_h]` coverage atlas.
/// `bg` is `[u32; width * height]` background packed RGBA.
/// `output` is `[u32; width * height]` destination framebuffer.
#[must_use]
pub fn text_run_blend(
    glyphs: &str,
    glyph_count: u32,
    atlas: &str,
    atlas_w: u32,
    atlas_h: u32,
    bg: &str,
    output: &str,
    width: u32,
    height: u32,
) -> Program {
    let pixel_count = width * height;
    let mut body = vec![
        Node::let_bind("idx", Expr::logical_index(0)),
        Node::let_bind("px", Expr::rem(Expr::var("idx"), Expr::u32(width))),
        Node::let_bind("py", Expr::div(Expr::var("idx"), Expr::u32(width))),
        Node::let_bind("curr_px_0", Expr::load(bg, Expr::var("idx"))),
    ];

    for g in 0..glyph_count {
        let base = g * 7;
        let g_gx = format!("g_gx_{g}");
        let g_gy = format!("g_gy_{g}");
        let g_gw = format!("g_gw_{g}");
        let g_gh = format!("g_gh_{g}");
        let g_u0 = format!("g_u0_{g}");
        let g_v0 = format!("g_v0_{g}");
        let g_col = format!("g_col_{g}");

        let g_inside = format!("g_inside_{g}");
        let g_u = format!("g_u_{g}");
        let g_v = format!("g_v_{g}");
        let g_a_idx = format!("g_a_idx_{g}");
        let g_cov_raw = format!("g_cov_raw_{g}");
        let g_cov = format!("g_cov_{g}");
        let g_inv_cov = format!("g_inv_cov_{g}");

        let fg_r = format!("fg_r_{g}");
        let fg_g = format!("fg_g_{g}");
        let fg_b = format!("fg_b_{g}");
        let bg_r = format!("bg_r_{g}");
        let bg_g = format!("bg_g_{g}");
        let bg_b = format!("bg_b_{g}");
        let bg_a = format!("bg_a_{g}");

        let out_r = format!("out_r_{g}");
        let out_g = format!("out_g_{g}");
        let out_b = format!("out_b_{g}");
        let blended_px = format!("blended_px_{g}");
        let next_px = format!("next_px_{g}");

        body.extend(vec![
            Node::let_bind(&g_gx, Expr::load(glyphs, Expr::u32(base))),
            Node::let_bind(&g_gy, Expr::load(glyphs, Expr::u32(base + 1))),
            Node::let_bind(&g_gw, Expr::load(glyphs, Expr::u32(base + 2))),
            Node::let_bind(&g_gh, Expr::load(glyphs, Expr::u32(base + 3))),
            Node::let_bind(&g_u0, Expr::load(glyphs, Expr::u32(base + 4))),
            Node::let_bind(&g_v0, Expr::load(glyphs, Expr::u32(base + 5))),
            Node::let_bind(&g_col, Expr::load(glyphs, Expr::u32(base + 6))),
            Node::let_bind(
                &g_inside,
                Expr::and(
                    Expr::and(
                        Expr::ge(Expr::var("px"), Expr::var(&g_gx)),
                        Expr::lt(
                            Expr::var("px"),
                            Expr::add(Expr::var(&g_gx), Expr::var(&g_gw)),
                        ),
                    ),
                    Expr::and(
                        Expr::ge(Expr::var("py"), Expr::var(&g_gy)),
                        Expr::lt(
                            Expr::var("py"),
                            Expr::add(Expr::var(&g_gy), Expr::var(&g_gh)),
                        ),
                    ),
                ),
            ),
            Node::let_bind(
                &g_u,
                Expr::add(
                    Expr::var(&g_u0),
                    Expr::sub(Expr::var("px"), Expr::var(&g_gx)),
                ),
            ),
            Node::let_bind(
                &g_v,
                Expr::add(
                    Expr::var(&g_v0),
                    Expr::sub(Expr::var("py"), Expr::var(&g_gy)),
                ),
            ),
            Node::let_bind(
                &g_a_idx,
                Expr::add(
                    Expr::var(&g_u),
                    Expr::mul(Expr::var(&g_v), Expr::u32(atlas_w)),
                ),
            ),
            Node::let_bind(
                &g_cov_raw,
                Expr::select(
                    Expr::var(&g_inside),
                    Expr::load(atlas, Expr::var(&g_a_idx)),
                    Expr::u32(0),
                ),
            ),
            // Extract alpha coverage (bits 24..31 or low 8 bits)
            Node::let_bind(
                &g_cov,
                Expr::select(
                    Expr::gt(
                        vyre_libs_builder::builder::stencil::unpack_channel(&g_cov_raw, 24),
                        Expr::u32(0),
                    ),
                    vyre_libs_builder::builder::stencil::unpack_channel(&g_cov_raw, 24),
                    vyre_libs_builder::builder::stencil::unpack_channel(&g_cov_raw, 0),
                ),
            ),
            Node::let_bind(&g_inv_cov, Expr::sub(Expr::u32(255), Expr::var(&g_cov))),
            // Unpack fg and current bg
            Node::let_bind(
                &fg_r,
                vyre_libs_builder::builder::stencil::unpack_channel(&g_col, 0),
            ),
            Node::let_bind(
                &fg_g,
                vyre_libs_builder::builder::stencil::unpack_channel(&g_col, 8),
            ),
            Node::let_bind(
                &fg_b,
                vyre_libs_builder::builder::stencil::unpack_channel(&g_col, 16),
            ),
            Node::let_bind(
                &bg_r,
                vyre_libs_builder::builder::stencil::unpack_channel(&format!("curr_px_{g}"), 0),
            ),
            Node::let_bind(
                &bg_g,
                vyre_libs_builder::builder::stencil::unpack_channel(&format!("curr_px_{g}"), 8),
            ),
            Node::let_bind(
                &bg_b,
                vyre_libs_builder::builder::stencil::unpack_channel(&format!("curr_px_{g}"), 16),
            ),
            Node::let_bind(
                &bg_a,
                vyre_libs_builder::builder::stencil::unpack_channel(&format!("curr_px_{g}"), 24),
            ),
            // Channel blending: (fg * cov + bg * (255 - cov) + 128) / 255
            Node::let_bind(
                &out_r,
                vyre_libs_builder::builder::stencil::clamp_u8(Expr::div(
                    Expr::add(
                        Expr::mul(Expr::var(&fg_r), Expr::var(&g_cov)),
                        Expr::mul(Expr::var(&bg_r), Expr::var(&g_inv_cov)),
                    ),
                    Expr::u32(255),
                )),
            ),
            Node::let_bind(
                &out_g,
                vyre_libs_builder::builder::stencil::clamp_u8(Expr::div(
                    Expr::add(
                        Expr::mul(Expr::var(&fg_g), Expr::var(&g_cov)),
                        Expr::mul(Expr::var(&bg_g), Expr::var(&g_inv_cov)),
                    ),
                    Expr::u32(255),
                )),
            ),
            Node::let_bind(
                &out_b,
                vyre_libs_builder::builder::stencil::clamp_u8(Expr::div(
                    Expr::add(
                        Expr::mul(Expr::var(&fg_b), Expr::var(&g_cov)),
                        Expr::mul(Expr::var(&bg_b), Expr::var(&g_inv_cov)),
                    ),
                    Expr::u32(255),
                )),
            ),
            Node::let_bind(
                &blended_px,
                Expr::bitor(
                    Expr::var(&out_r),
                    Expr::bitor(
                        Expr::shl(Expr::var(&out_g), Expr::u32(8)),
                        Expr::bitor(
                            Expr::shl(Expr::var(&out_b), Expr::u32(16)),
                            Expr::shl(Expr::var(&bg_a), Expr::u32(24)),
                        ),
                    ),
                ),
            ),
            Node::let_bind(
                &next_px,
                Expr::select(
                    Expr::and(
                        Expr::var(&g_inside),
                        Expr::gt(Expr::var(&g_cov), Expr::u32(0)),
                    ),
                    Expr::var(&blended_px),
                    Expr::var(format!("curr_px_{g}")),
                ),
            ),
            Node::let_bind(format!("curr_px_{}", g + 1), Expr::var(&next_px)),
        ]);
    }

    body.push(Node::store(
        output,
        Expr::var("idx"),
        Expr::var(format!("curr_px_{glyph_count}")),
    ));

    Program::wrapped(
        vec![
            BufferDecl::storage(glyphs, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(glyph_count * 7),
            BufferDecl::storage(atlas, 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(atlas_w * atlas_h),
            BufferDecl::storage(bg, 2, BufferAccess::ReadOnly, DataType::U32)
                .with_count(pixel_count),
            BufferDecl::output(output, 3, DataType::U32).with_count(pixel_count),
        ],
        super::PIXEL_WORKGROUP_SIZE,
        vec![wrap_anonymous_region(
            OP_ID,
            vec![wrap_child_region(
                OP_ID,
                Ident::from(OP_ID),
                vec![
                    Node::let_bind("idx_guard", Expr::logical_index(0)),
                    Node::if_then(
                        Expr::lt(Expr::var("idx_guard"), Expr::u32(pixel_count)),
                        body,
                    ),
                ],
            )],
        )],
    )
}

const EXPECTED_TEXT_RUN_OUTPUT_BYTES: [u8; 16] = [
    0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0xFF, 0x00, 0x00, 0x00, 0xFF, 0x00, 0x00, 0x00, 0xFF,
];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID,
        || text_run_blend("glyphs", 1, "atlas", 2, 2, "bg", "out", 2, 2),
        Some(|| {
            let glyph_data = [0u32, 0, 1, 1, 0, 0, 0xFF00_00FF]; // (0,0) 1x1 at atlas (0,0), blue
            let atlas_data = [0xFF00_0000u32, 0, 0, 0]; // 255 alpha at (0,0)
            let bg_data = [0xFF00_0000u32; 4]; // black opaque
            vec![vec![
                vyre_primitives::wire::pack_u32_slice(&glyph_data),
                vyre_primitives::wire::pack_u32_slice(&atlas_data),
                vyre_primitives::wire::pack_u32_slice(&bg_data),
                vec![0; 16],
            ]]
        }),
        Some(|| {
            vec![vec![EXPECTED_TEXT_RUN_OUTPUT_BYTES.to_vec()]]
        }),
    )
    .with_category("visual")
    .with_uncharacterized()
}
