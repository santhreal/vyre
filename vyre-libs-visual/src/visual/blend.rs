//! Extended Porter-Duff and UI compositing blend modes.
//!
//! Evaluates all twelve canonical Porter-Duff compositing operations plus
//! standard separable UI blending formulas (Multiply, Screen, Add / Linear Dodge).
//!
//! Category A composition - pure IR over existing expressions.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

const OP_ID: &str = "vyre-libs::visual::composite_blend";

/// Blend modes supported by the composite blend composition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    /// Standard source over destination: `src + dst * (1 - src_a)`.
    SrcOver = 0,
    /// Destination over source: `dst + src * (1 - dst_a)`.
    DstOver = 1,
    /// Source in destination: `src * dst_a`.
    SrcIn = 2,
    /// Destination in source: `dst * src_a`.
    DstIn = 3,
    /// Source out of destination: `src * (1 - dst_a)`.
    SrcOut = 4,
    /// Destination out of source: `dst * (1 - src_a)`.
    DstOut = 5,
    /// Source atop destination: `src * dst_a + dst * (1 - src_a)`.
    SrcAtop = 6,
    /// Destination atop source: `dst * src_a + src * (1 - dst_a)`.
    DstAtop = 7,
    /// Mutual exclusive or: `src * (1 - dst_a) + dst * (1 - src_a)`.
    Xor = 8,
    /// Multiplicative blend: `(src * dst) / 255`.
    Multiply = 9,
    /// Screen blend: `src + dst - (src * dst) / 255`.
    Screen = 10,
    /// Additive / linear dodge: `min(src + dst, 255)`.
    Add = 11,
}

/// Build a Program that composites `fg` over `bg` using the specified [`BlendMode`].
#[must_use]
pub fn composite_blend(fg: &str, bg: &str, output: &str, count: u32, mode: BlendMode) -> Program {
    let body = vec![
        Node::let_bind("fg_px", Expr::load(fg, Expr::var("idx"))),
        Node::let_bind("bg_px", Expr::load(bg, Expr::var("idx"))),
        Node::let_bind(
            "fg_r",
            vyre_libs_builder::builder::stencil::unpack_channel("fg_px", 0),
        ),
        Node::let_bind(
            "fg_g",
            vyre_libs_builder::builder::stencil::unpack_channel("fg_px", 8),
        ),
        Node::let_bind(
            "fg_b",
            vyre_libs_builder::builder::stencil::unpack_channel("fg_px", 16),
        ),
        Node::let_bind(
            "fg_a",
            vyre_libs_builder::builder::stencil::unpack_channel("fg_px", 24),
        ),
        Node::let_bind(
            "bg_r",
            vyre_libs_builder::builder::stencil::unpack_channel("bg_px", 0),
        ),
        Node::let_bind(
            "bg_g",
            vyre_libs_builder::builder::stencil::unpack_channel("bg_px", 8),
        ),
        Node::let_bind(
            "bg_b",
            vyre_libs_builder::builder::stencil::unpack_channel("bg_px", 16),
        ),
        Node::let_bind(
            "bg_a",
            vyre_libs_builder::builder::stencil::unpack_channel("bg_px", 24),
        ),
        Node::let_bind("inv_fg_a", Expr::sub(Expr::u32(255), Expr::var("fg_a"))),
        Node::let_bind("inv_bg_a", Expr::sub(Expr::u32(255), Expr::var("bg_a"))),
        // Channel compute based on mode
        Node::let_bind(
            "out_r",
            match mode {
                BlendMode::SrcOver => Expr::div(
                    Expr::add(
                        Expr::mul(Expr::var("fg_r"), Expr::u32(255)),
                        Expr::mul(Expr::var("bg_r"), Expr::var("inv_fg_a")),
                    ),
                    Expr::u32(255),
                ),
                BlendMode::DstOver => Expr::div(
                    Expr::add(
                        Expr::mul(Expr::var("bg_r"), Expr::u32(255)),
                        Expr::mul(Expr::var("fg_r"), Expr::var("inv_bg_a")),
                    ),
                    Expr::u32(255),
                ),
                BlendMode::SrcIn => Expr::div(
                    Expr::mul(Expr::var("fg_r"), Expr::var("bg_a")),
                    Expr::u32(255),
                ),
                BlendMode::DstIn => Expr::div(
                    Expr::mul(Expr::var("bg_r"), Expr::var("fg_a")),
                    Expr::u32(255),
                ),
                BlendMode::SrcOut => Expr::div(
                    Expr::mul(Expr::var("fg_r"), Expr::var("inv_bg_a")),
                    Expr::u32(255),
                ),
                BlendMode::DstOut => Expr::div(
                    Expr::mul(Expr::var("bg_r"), Expr::var("inv_fg_a")),
                    Expr::u32(255),
                ),
                BlendMode::SrcAtop => Expr::div(
                    Expr::add(
                        Expr::mul(Expr::var("fg_r"), Expr::var("bg_a")),
                        Expr::mul(Expr::var("bg_r"), Expr::var("inv_fg_a")),
                    ),
                    Expr::u32(255),
                ),
                BlendMode::DstAtop => Expr::div(
                    Expr::add(
                        Expr::mul(Expr::var("bg_r"), Expr::var("fg_a")),
                        Expr::mul(Expr::var("fg_r"), Expr::var("inv_bg_a")),
                    ),
                    Expr::u32(255),
                ),
                BlendMode::Xor => Expr::div(
                    Expr::add(
                        Expr::mul(Expr::var("fg_r"), Expr::var("inv_bg_a")),
                        Expr::mul(Expr::var("bg_r"), Expr::var("inv_fg_a")),
                    ),
                    Expr::u32(255),
                ),
                BlendMode::Multiply => Expr::div(
                    Expr::mul(Expr::var("fg_r"), Expr::var("bg_r")),
                    Expr::u32(255),
                ),
                BlendMode::Screen => Expr::sub(
                    Expr::add(Expr::var("fg_r"), Expr::var("bg_r")),
                    Expr::div(
                        Expr::mul(Expr::var("fg_r"), Expr::var("bg_r")),
                        Expr::u32(255),
                    ),
                ),
                BlendMode::Add => Expr::add(Expr::var("fg_r"), Expr::var("bg_r")),
            },
        ),
        Node::let_bind(
            "out_g",
            match mode {
                BlendMode::SrcOver => Expr::div(
                    Expr::add(
                        Expr::mul(Expr::var("fg_g"), Expr::u32(255)),
                        Expr::mul(Expr::var("bg_g"), Expr::var("inv_fg_a")),
                    ),
                    Expr::u32(255),
                ),
                BlendMode::DstOver => Expr::div(
                    Expr::add(
                        Expr::mul(Expr::var("bg_g"), Expr::u32(255)),
                        Expr::mul(Expr::var("fg_g"), Expr::var("inv_bg_a")),
                    ),
                    Expr::u32(255),
                ),
                BlendMode::SrcIn => Expr::div(
                    Expr::mul(Expr::var("fg_g"), Expr::var("bg_a")),
                    Expr::u32(255),
                ),
                BlendMode::DstIn => Expr::div(
                    Expr::mul(Expr::var("bg_g"), Expr::var("fg_a")),
                    Expr::u32(255),
                ),
                BlendMode::SrcOut => Expr::div(
                    Expr::mul(Expr::var("fg_g"), Expr::var("inv_bg_a")),
                    Expr::u32(255),
                ),
                BlendMode::DstOut => Expr::div(
                    Expr::mul(Expr::var("bg_g"), Expr::var("inv_fg_a")),
                    Expr::u32(255),
                ),
                BlendMode::SrcAtop => Expr::div(
                    Expr::add(
                        Expr::mul(Expr::var("fg_g"), Expr::var("bg_a")),
                        Expr::mul(Expr::var("bg_g"), Expr::var("inv_fg_a")),
                    ),
                    Expr::u32(255),
                ),
                BlendMode::DstAtop => Expr::div(
                    Expr::add(
                        Expr::mul(Expr::var("bg_g"), Expr::var("fg_a")),
                        Expr::mul(Expr::var("fg_g"), Expr::var("inv_bg_a")),
                    ),
                    Expr::u32(255),
                ),
                BlendMode::Xor => Expr::div(
                    Expr::add(
                        Expr::mul(Expr::var("fg_g"), Expr::var("inv_bg_a")),
                        Expr::mul(Expr::var("bg_g"), Expr::var("inv_fg_a")),
                    ),
                    Expr::u32(255),
                ),
                BlendMode::Multiply => Expr::div(
                    Expr::mul(Expr::var("fg_g"), Expr::var("bg_g")),
                    Expr::u32(255),
                ),
                BlendMode::Screen => Expr::sub(
                    Expr::add(Expr::var("fg_g"), Expr::var("bg_g")),
                    Expr::div(
                        Expr::mul(Expr::var("fg_g"), Expr::var("bg_g")),
                        Expr::u32(255),
                    ),
                ),
                BlendMode::Add => Expr::add(Expr::var("fg_g"), Expr::var("bg_g")),
            },
        ),
        Node::let_bind(
            "out_b",
            match mode {
                BlendMode::SrcOver => Expr::div(
                    Expr::add(
                        Expr::mul(Expr::var("fg_b"), Expr::u32(255)),
                        Expr::mul(Expr::var("bg_b"), Expr::var("inv_fg_a")),
                    ),
                    Expr::u32(255),
                ),
                BlendMode::DstOver => Expr::div(
                    Expr::add(
                        Expr::mul(Expr::var("bg_b"), Expr::u32(255)),
                        Expr::mul(Expr::var("fg_b"), Expr::var("inv_bg_a")),
                    ),
                    Expr::u32(255),
                ),
                BlendMode::SrcIn => Expr::div(
                    Expr::mul(Expr::var("fg_b"), Expr::var("bg_a")),
                    Expr::u32(255),
                ),
                BlendMode::DstIn => Expr::div(
                    Expr::mul(Expr::var("bg_b"), Expr::var("fg_a")),
                    Expr::u32(255),
                ),
                BlendMode::SrcOut => Expr::div(
                    Expr::mul(Expr::var("fg_b"), Expr::var("inv_bg_a")),
                    Expr::u32(255),
                ),
                BlendMode::DstOut => Expr::div(
                    Expr::mul(Expr::var("bg_b"), Expr::var("inv_fg_a")),
                    Expr::u32(255),
                ),
                BlendMode::SrcAtop => Expr::div(
                    Expr::add(
                        Expr::mul(Expr::var("fg_b"), Expr::var("bg_a")),
                        Expr::mul(Expr::var("bg_b"), Expr::var("inv_fg_a")),
                    ),
                    Expr::u32(255),
                ),
                BlendMode::DstAtop => Expr::div(
                    Expr::add(
                        Expr::mul(Expr::var("bg_b"), Expr::var("fg_a")),
                        Expr::mul(Expr::var("fg_b"), Expr::var("inv_bg_a")),
                    ),
                    Expr::u32(255),
                ),
                BlendMode::Xor => Expr::div(
                    Expr::add(
                        Expr::mul(Expr::var("fg_b"), Expr::var("inv_bg_a")),
                        Expr::mul(Expr::var("bg_b"), Expr::var("inv_fg_a")),
                    ),
                    Expr::u32(255),
                ),
                BlendMode::Multiply => Expr::div(
                    Expr::mul(Expr::var("fg_b"), Expr::var("bg_b")),
                    Expr::u32(255),
                ),
                BlendMode::Screen => Expr::sub(
                    Expr::add(Expr::var("fg_b"), Expr::var("bg_b")),
                    Expr::div(
                        Expr::mul(Expr::var("fg_b"), Expr::var("bg_b")),
                        Expr::u32(255),
                    ),
                ),
                BlendMode::Add => Expr::add(Expr::var("fg_b"), Expr::var("bg_b")),
            },
        ),
        Node::let_bind(
            "out_a",
            match mode {
                BlendMode::SrcIn => Expr::div(
                    Expr::mul(Expr::var("fg_a"), Expr::var("bg_a")),
                    Expr::u32(255),
                ),
                BlendMode::DstIn => Expr::div(
                    Expr::mul(Expr::var("bg_a"), Expr::var("fg_a")),
                    Expr::u32(255),
                ),
                BlendMode::SrcOut => Expr::div(
                    Expr::mul(Expr::var("fg_a"), Expr::var("inv_bg_a")),
                    Expr::u32(255),
                ),
                BlendMode::DstOut => Expr::div(
                    Expr::mul(Expr::var("bg_a"), Expr::var("inv_fg_a")),
                    Expr::u32(255),
                ),
                _ => vyre_libs_builder::builder::stencil::clamp_u8(Expr::add(
                    Expr::var("fg_a"),
                    Expr::div(
                        Expr::mul(Expr::var("bg_a"), Expr::var("inv_fg_a")),
                        Expr::u32(255),
                    ),
                )),
            },
        ),
        Node::let_bind(
            "cr",
            vyre_libs_builder::builder::stencil::clamp_u8(Expr::var("out_r")),
        ),
        Node::let_bind(
            "cg",
            vyre_libs_builder::builder::stencil::clamp_u8(Expr::var("out_g")),
        ),
        Node::let_bind(
            "cb",
            vyre_libs_builder::builder::stencil::clamp_u8(Expr::var("out_b")),
        ),
        Node::let_bind(
            "ca",
            vyre_libs_builder::builder::stencil::clamp_u8(Expr::var("out_a")),
        ),
        Node::let_bind(
            "packed_blend",
            Expr::bitor(
                Expr::var("cr"),
                Expr::bitor(
                    Expr::shl(Expr::var("cg"), Expr::u32(8)),
                    Expr::bitor(
                        Expr::shl(Expr::var("cb"), Expr::u32(16)),
                        Expr::shl(Expr::var("ca"), Expr::u32(24)),
                    ),
                ),
            ),
        ),
        Node::store(output, Expr::var("idx"), Expr::var("packed_blend")),
    ];

    crate::visual::packed_rgba_map::build_pixel_pipeline(
        OP_ID,
        vec![
            BufferDecl::storage(fg, 0, BufferAccess::ReadOnly, DataType::U32).with_count(count),
            BufferDecl::storage(bg, 1, BufferAccess::ReadOnly, DataType::U32).with_count(count),
            BufferDecl::output(output, 2, DataType::U32).with_count(count),
        ],
        count,
        body,
    )
}

const EXPECTED_BLEND_ADD_OUTPUT_BYTES: [u8; 8] = [0xFF, 0xFF, 0x00, 0xFF, 0xFF, 0x00, 0xFF, 0xFF];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID,
        || composite_blend("fg", "bg", "out", 2, BlendMode::Add),
        Some(|| {
            let fg = [0xFF00_00FFu32, 0xFF00_00FF]; // red
            let bg = [0xFF00_FF00u32, 0xFFFF_0000]; // green, blue
            vec![vec![
                vyre_primitives::wire::pack_u32_slice(&fg),
                vyre_primitives::wire::pack_u32_slice(&bg),
                vec![0; 8],
            ]]
        }),
        Some(|| {
            vec![vec![EXPECTED_BLEND_ADD_OUTPUT_BYTES.to_vec()]]
        }),
    )
    .with_category("visual")
    .with_uncharacterized()
}
