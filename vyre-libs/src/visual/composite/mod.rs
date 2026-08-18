//! Porter-Duff alpha compositing ("over" operation).
//!
//! `result = fg + bg * (1 - fg_alpha)`
//!
//! Category A composition  -  pure IR over existing expressions.
//! No Tier 2.5 primitives consumed.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

const OP_ID: &str = "vyre-libs::visual::composite";

/// Build a Program that composites `fg` over `bg` using Porter-Duff
/// "over" arithmetic, writing the result to `output`.
///
/// All buffers are `[u32; count]`  -  packed RGBA pixels.
#[must_use]
pub fn alpha_over(fg: &str, bg: &str, output: &str, count: u32) -> Program {
    let body = vec![
        // Load foreground and background pixels.
        Node::let_bind("fg_px", Expr::load(fg, Expr::var("idx"))),
        Node::let_bind("bg_px", Expr::load(bg, Expr::var("idx"))),
        Node::let_bind("fg_r", crate::builder::stencil::unpack_channel("fg_px", 0)),
        Node::let_bind("fg_g", crate::builder::stencil::unpack_channel("fg_px", 8)),
        Node::let_bind("fg_b", crate::builder::stencil::unpack_channel("fg_px", 16)),
        Node::let_bind("fg_a", crate::builder::stencil::unpack_channel("fg_px", 24)),
        Node::let_bind("bg_r", crate::builder::stencil::unpack_channel("bg_px", 0)),
        Node::let_bind("bg_g", crate::builder::stencil::unpack_channel("bg_px", 8)),
        Node::let_bind("bg_b", crate::builder::stencil::unpack_channel("bg_px", 16)),
        Node::let_bind("bg_a", crate::builder::stencil::unpack_channel("bg_px", 24)),
        // inv_a = 255 - fg_a
        Node::let_bind("inv_a", Expr::sub(Expr::u32(255), Expr::var("fg_a"))),
        // Porter-Duff over per channel.
        Node::let_bind(
            "out_r",
            crate::builder::stencil::blend_channel_porter_duff(
                Expr::var("fg_r"),
                Expr::var("bg_r"),
                Expr::var("inv_a"),
            ),
        ),
        Node::let_bind(
            "out_g",
            crate::builder::stencil::blend_channel_porter_duff(
                Expr::var("fg_g"),
                Expr::var("bg_g"),
                Expr::var("inv_a"),
            ),
        ),
        Node::let_bind(
            "out_b",
            crate::builder::stencil::blend_channel_porter_duff(
                Expr::var("fg_b"),
                Expr::var("bg_b"),
                Expr::var("inv_a"),
            ),
        ),
        Node::let_bind(
            "out_a",
            crate::builder::stencil::blend_channel_porter_duff(
                Expr::var("fg_a"),
                Expr::var("bg_a"),
                Expr::var("inv_a"),
            ),
        ),
        // Clamp to 255 and pack RGBA.
        Node::let_bind("cr", crate::builder::stencil::clamp_u8(Expr::var("out_r"))),
        Node::let_bind("cg", crate::builder::stencil::clamp_u8(Expr::var("out_g"))),
        Node::let_bind("cb", crate::builder::stencil::clamp_u8(Expr::var("out_b"))),
        Node::let_bind("ca", crate::builder::stencil::clamp_u8(Expr::var("out_a"))),
        Node::let_bind(
            "packed",
            crate::builder::stencil::pack_rgba_named("cr", "cg", "cb", "ca"),
        ),
        Node::store(output, Expr::var("idx"), Expr::var("packed")),
    ];

    crate::visual::packed_rgba_map::build_pixel_pipeline(
        OP_ID,
        vec![
            BufferDecl::storage(fg, 0, BufferAccess::ReadOnly, DataType::U32).with_count(count),
            BufferDecl::storage(bg, 1, BufferAccess::ReadOnly, DataType::U32).with_count(count),
            BufferDecl::storage(output, 2, BufferAccess::ReadWrite, DataType::U32)
                .with_count(count),
        ],
        count,
        body,
    )
}

const EXPECTED_ALPHA_OVER_OUTPUT_BYTES: [u8; 8] = [0xFF, 0x00, 0x7F, 0xFF, 0x00, 0xFF, 0x00, 0xFF];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || alpha_over("fg", "bg", "out", 2),
        Some(|| {
            // Pixel 0: semi-transparent red (128 alpha) over opaque blue.
            // Pixel 1: fully opaque green over opaque white.
            let fg = [0x8000_00FFu32, 0xFF00_FF00u32]; // RGBA: R=255 A=128; R=0 G=255 A=255
            let bg = [0xFF_FF0000u32, 0xFFFF_FFFFu32]; // RGBA: B=255 A=255; white A=255
            vec![vec![
                crate::visual::u32_word_bytes::u32_words_to_le_bytes(&fg),
                crate::visual::u32_word_bytes::u32_words_to_le_bytes(&bg),
                vec![0u8; 8],   // output
            ]]
        }),
        Some(|| {
            vec![vec![EXPECTED_ALPHA_OVER_OUTPUT_BYTES.to_vec()]]
        }),
    )
    .with_category("visual")
}
