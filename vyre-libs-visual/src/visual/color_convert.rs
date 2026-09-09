//! Image color space and format conversion compositions.
//!
//! Provides color space conversions (RGBA to/from Grayscale), alpha representation
//! transformations (premultiplied and straight alpha), and arbitrary linear color
//! matrix filters.
//!
//! Category A composition - pure IR over existing expressions.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

const OP_ID_GRAYSCALE: &str = "vyre-libs::visual::rgba_to_grayscale";
const OP_ID_PREMULTIPLY: &str = "vyre-libs::visual::premultiply_alpha";
const OP_ID_UNPREMULTIPLY: &str = "vyre-libs::visual::unpremultiply_alpha";

/// Convert packed RGBA pixels to grayscale using standard Rec. 601 luma weights.
///
/// `Y = (77 * R + 150 * G + 29 * B + 128) >> 8`
#[must_use]
pub fn rgba_to_grayscale(input: &str, output: &str, count: u32) -> Program {
    let body = vec![
        Node::let_bind("px", Expr::load(input, Expr::var("idx"))),
        Node::let_bind(
            "r",
            vyre_libs_builder::builder::stencil::unpack_channel("px", 0),
        ),
        Node::let_bind(
            "g",
            vyre_libs_builder::builder::stencil::unpack_channel("px", 8),
        ),
        Node::let_bind(
            "b",
            vyre_libs_builder::builder::stencil::unpack_channel("px", 16),
        ),
        Node::let_bind(
            "a",
            vyre_libs_builder::builder::stencil::unpack_channel("px", 24),
        ),
        Node::let_bind(
            "luma",
            vyre_libs_builder::builder::stencil::clamp_u8(Expr::shr(
                Expr::add(
                    Expr::add(
                        Expr::mul(Expr::var("r"), Expr::u32(77)),
                        Expr::mul(Expr::var("g"), Expr::u32(150)),
                    ),
                    Expr::add(Expr::mul(Expr::var("b"), Expr::u32(29)), Expr::u32(128)),
                ),
                Expr::u32(8),
            )),
        ),
        Node::let_bind(
            "packed_gray",
            Expr::bitor(
                Expr::var("luma"),
                Expr::bitor(
                    Expr::shl(Expr::var("luma"), Expr::u32(8)),
                    Expr::bitor(
                        Expr::shl(Expr::var("luma"), Expr::u32(16)),
                        Expr::shl(Expr::var("a"), Expr::u32(24)),
                    ),
                ),
            ),
        ),
        Node::store(output, Expr::var("idx"), Expr::var("packed_gray")),
    ];

    crate::visual::packed_rgba_map::build_pixel_pipeline(
        OP_ID_GRAYSCALE,
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::U32).with_count(count),
            BufferDecl::output(output, 1, DataType::U32).with_count(count),
        ],
        count,
        body,
    )
}
/// Convert grayscale pixels `(Y, A)` to packed RGBA `(Y, Y, Y, A)`.
#[must_use]
pub fn grayscale_to_rgba(input: &str, output: &str, count: u32) -> Program {
    let body = vec![
        Node::let_bind("px", Expr::load(input, Expr::var("idx"))),
        Node::let_bind(
            "y",
            vyre_libs_builder::builder::stencil::unpack_channel("px", 0),
        ),
        Node::let_bind(
            "a",
            vyre_libs_builder::builder::stencil::unpack_channel("px", 24),
        ),
        Node::let_bind(
            "packed_rgba",
            Expr::bitor(
                Expr::var("y"),
                Expr::bitor(
                    Expr::shl(Expr::var("y"), Expr::u32(8)),
                    Expr::bitor(
                        Expr::shl(Expr::var("y"), Expr::u32(16)),
                        Expr::shl(Expr::var("a"), Expr::u32(24)),
                    ),
                ),
            ),
        ),
        Node::store(output, Expr::var("idx"), Expr::var("packed_rgba")),
    ];

    crate::visual::packed_rgba_map::build_pixel_pipeline(
        OP_ID_GRAYSCALE,
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::U32).with_count(count),
            BufferDecl::output(output, 1, DataType::U32).with_count(count),
        ],
        count,
        body,
    )
}

/// Convert straight RGBA pixels to premultiplied alpha format.
///
/// `R' = (R * A + 127) / 255`, `G' = (G * A + 127) / 255`, `B' = (B * A + 127) / 255`, `A' = A`
#[must_use]
pub fn premultiply_alpha(input: &str, output: &str, count: u32) -> Program {
    let body = vec![
        Node::let_bind("px", Expr::load(input, Expr::var("idx"))),
        Node::let_bind(
            "r",
            vyre_libs_builder::builder::stencil::unpack_channel("px", 0),
        ),
        Node::let_bind(
            "g",
            vyre_libs_builder::builder::stencil::unpack_channel("px", 8),
        ),
        Node::let_bind(
            "b",
            vyre_libs_builder::builder::stencil::unpack_channel("px", 16),
        ),
        Node::let_bind(
            "a",
            vyre_libs_builder::builder::stencil::unpack_channel("px", 24),
        ),
        Node::let_bind(
            "pr",
            vyre_libs_builder::builder::stencil::clamp_u8(Expr::div(
                Expr::add(Expr::mul(Expr::var("r"), Expr::var("a")), Expr::u32(127)),
                Expr::u32(255),
            )),
        ),
        Node::let_bind(
            "pg",
            vyre_libs_builder::builder::stencil::clamp_u8(Expr::div(
                Expr::add(Expr::mul(Expr::var("g"), Expr::var("a")), Expr::u32(127)),
                Expr::u32(255),
            )),
        ),
        Node::let_bind(
            "pb",
            vyre_libs_builder::builder::stencil::clamp_u8(Expr::div(
                Expr::add(Expr::mul(Expr::var("b"), Expr::var("a")), Expr::u32(127)),
                Expr::u32(255),
            )),
        ),
        Node::let_bind(
            "packed_pm",
            Expr::bitor(
                Expr::var("pr"),
                Expr::bitor(
                    Expr::shl(Expr::var("pg"), Expr::u32(8)),
                    Expr::bitor(
                        Expr::shl(Expr::var("pb"), Expr::u32(16)),
                        Expr::shl(Expr::var("a"), Expr::u32(24)),
                    ),
                ),
            ),
        ),
        Node::store(output, Expr::var("idx"), Expr::var("packed_pm")),
    ];

    crate::visual::packed_rgba_map::build_pixel_pipeline(
        OP_ID_PREMULTIPLY,
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::U32).with_count(count),
            BufferDecl::output(output, 1, DataType::U32).with_count(count),
        ],
        count,
        body,
    )
}

/// Convert premultiplied RGBA pixels back to straight alpha format.
#[must_use]
pub fn unpremultiply_alpha(input: &str, output: &str, count: u32) -> Program {
    let body = vec![
        Node::let_bind("px", Expr::load(input, Expr::var("idx"))),
        Node::let_bind(
            "r",
            vyre_libs_builder::builder::stencil::unpack_channel("px", 0),
        ),
        Node::let_bind(
            "g",
            vyre_libs_builder::builder::stencil::unpack_channel("px", 8),
        ),
        Node::let_bind(
            "b",
            vyre_libs_builder::builder::stencil::unpack_channel("px", 16),
        ),
        Node::let_bind(
            "a",
            vyre_libs_builder::builder::stencil::unpack_channel("px", 24),
        ),
        Node::let_bind(
            "ur",
            Expr::select(
                Expr::gt(Expr::var("a"), Expr::u32(0)),
                vyre_libs_builder::builder::stencil::clamp_u8(Expr::div(
                    Expr::mul(Expr::var("r"), Expr::u32(255)),
                    Expr::var("a"),
                )),
                Expr::u32(0),
            ),
        ),
        Node::let_bind(
            "ug",
            Expr::select(
                Expr::gt(Expr::var("a"), Expr::u32(0)),
                vyre_libs_builder::builder::stencil::clamp_u8(Expr::div(
                    Expr::mul(Expr::var("g"), Expr::u32(255)),
                    Expr::var("a"),
                )),
                Expr::u32(0),
            ),
        ),
        Node::let_bind(
            "ub",
            Expr::select(
                Expr::gt(Expr::var("a"), Expr::u32(0)),
                vyre_libs_builder::builder::stencil::clamp_u8(Expr::div(
                    Expr::mul(Expr::var("b"), Expr::u32(255)),
                    Expr::var("a"),
                )),
                Expr::u32(0),
            ),
        ),
        Node::let_bind(
            "packed_unpm",
            Expr::bitor(
                Expr::var("ur"),
                Expr::bitor(
                    Expr::shl(Expr::var("ug"), Expr::u32(8)),
                    Expr::bitor(
                        Expr::shl(Expr::var("ub"), Expr::u32(16)),
                        Expr::shl(Expr::var("a"), Expr::u32(24)),
                    ),
                ),
            ),
        ),
        Node::store(output, Expr::var("idx"), Expr::var("packed_unpm")),
    ];

    crate::visual::packed_rgba_map::build_pixel_pipeline(
        OP_ID_UNPREMULTIPLY,
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::U32).with_count(count),
            BufferDecl::output(output, 1, DataType::U32).with_count(count),
        ],
        count,
        body,
    )
}

const EXPECTED_GRAYSCALE_OUTPUT_BYTES: [u8; 8] = [0x4C, 0x4C, 0x4C, 0xFF, 0x96, 0x96, 0x96, 0xFF];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID_GRAYSCALE,
        || rgba_to_grayscale("in", "out", 2),
        Some(|| {
            let pixels = [0xFF00_00FFu32, 0xFF00_FF00]; // red, green
            vec![vec![
                vyre_primitives::wire::pack_u32_slice(&pixels),
                vec![0; 8],
            ]]
        }),
        Some(|| {
            vec![vec![EXPECTED_GRAYSCALE_OUTPUT_BYTES.to_vec()]]
        }),
    )
    .with_category("visual")
    .with_opaque("rgba to grayscale conversion")
}
