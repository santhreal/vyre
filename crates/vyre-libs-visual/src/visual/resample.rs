//! Image scaling and bilinear resampling compositions.
//!
//! Re-samples packed RGBA pixel surfaces across arbitrary resolution changes
//! using fixed-point 8.8 bilinear filtering.
//!
//! Category A composition - pure IR over existing expressions.

use vyre_foundation::composition::{wrap_anonymous_region, wrap_child_region};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Ident, Node, Program};

const OP_ID: &str = "vyre-libs::visual::resample";

/// Build a Program that resamples an image from `(in_w, in_h)` to `(out_w, out_h)` using bilinear interpolation.
#[must_use]
pub fn bilinear_resample_rgba(
    input: &str,
    in_w: u32,
    in_h: u32,
    output: &str,
    out_w: u32,
    out_h: u32,
) -> Program {
    let out_count = out_w * out_h;
    let in_count = in_w * in_h;

    let body = vec![
        Node::let_bind("idx", Expr::logical_index(0)),
        Node::let_bind("ox", Expr::rem(Expr::var("idx"), Expr::u32(out_w))),
        Node::let_bind("oy", Expr::div(Expr::var("idx"), Expr::u32(out_w))),
        // 8.8 fixed-point source coordinates
        Node::let_bind(
            "sx_fp",
            Expr::div(
                Expr::mul(Expr::var("ox"), Expr::u32(in_w * 256)),
                Expr::u32(out_w),
            ),
        ),
        Node::let_bind(
            "sy_fp",
            Expr::div(
                Expr::mul(Expr::var("oy"), Expr::u32(in_h * 256)),
                Expr::u32(out_h),
            ),
        ),
        Node::let_bind("x0", Expr::shr(Expr::var("sx_fp"), Expr::u32(8))),
        Node::let_bind("y0", Expr::shr(Expr::var("sy_fp"), Expr::u32(8))),
        Node::let_bind(
            "x1",
            Expr::select(
                Expr::lt(Expr::add(Expr::var("x0"), Expr::u32(1)), Expr::u32(in_w)),
                Expr::add(Expr::var("x0"), Expr::u32(1)),
                Expr::u32(in_w.saturating_sub(1)),
            ),
        ),
        Node::let_bind(
            "y1",
            Expr::select(
                Expr::lt(Expr::add(Expr::var("y0"), Expr::u32(1)), Expr::u32(in_h)),
                Expr::add(Expr::var("y0"), Expr::u32(1)),
                Expr::u32(in_h.saturating_sub(1)),
            ),
        ),
        Node::let_bind("fx", Expr::bitand(Expr::var("sx_fp"), Expr::u32(255))),
        Node::let_bind("fy", Expr::bitand(Expr::var("sy_fp"), Expr::u32(255))),
        Node::let_bind("inv_fx", Expr::sub(Expr::u32(256), Expr::var("fx"))),
        Node::let_bind("inv_fy", Expr::sub(Expr::u32(256), Expr::var("fy"))),
        // Bilinear weights (scaled to 256)
        Node::let_bind(
            "w00",
            Expr::shr(
                Expr::mul(Expr::var("inv_fx"), Expr::var("inv_fy")),
                Expr::u32(8),
            ),
        ),
        Node::let_bind(
            "w10",
            Expr::shr(
                Expr::mul(Expr::var("fx"), Expr::var("inv_fy")),
                Expr::u32(8),
            ),
        ),
        Node::let_bind(
            "w01",
            Expr::shr(
                Expr::mul(Expr::var("inv_fx"), Expr::var("fy")),
                Expr::u32(8),
            ),
        ),
        Node::let_bind(
            "w11",
            Expr::shr(Expr::mul(Expr::var("fx"), Expr::var("fy")), Expr::u32(8)),
        ),
        // Load four neighbor pixels
        Node::let_bind(
            "p00",
            Expr::load(
                input,
                Expr::add(Expr::var("x0"), Expr::mul(Expr::var("y0"), Expr::u32(in_w))),
            ),
        ),
        Node::let_bind(
            "p10",
            Expr::load(
                input,
                Expr::add(Expr::var("x1"), Expr::mul(Expr::var("y0"), Expr::u32(in_w))),
            ),
        ),
        Node::let_bind(
            "p01",
            Expr::load(
                input,
                Expr::add(Expr::var("x0"), Expr::mul(Expr::var("y1"), Expr::u32(in_w))),
            ),
        ),
        Node::let_bind(
            "p11",
            Expr::load(
                input,
                Expr::add(Expr::var("x1"), Expr::mul(Expr::var("y1"), Expr::u32(in_w))),
            ),
        ),
        // Interpolate R
        Node::let_bind(
            "out_r",
            vyre_libs_builder::builder::stencil::clamp_u8(Expr::shr(
                Expr::add(
                    Expr::add(
                        Expr::mul(
                            vyre_libs_builder::builder::stencil::unpack_channel("p00", 0),
                            Expr::var("w00"),
                        ),
                        Expr::mul(
                            vyre_libs_builder::builder::stencil::unpack_channel("p10", 0),
                            Expr::var("w10"),
                        ),
                    ),
                    Expr::add(
                        Expr::mul(
                            vyre_libs_builder::builder::stencil::unpack_channel("p01", 0),
                            Expr::var("w01"),
                        ),
                        Expr::mul(
                            vyre_libs_builder::builder::stencil::unpack_channel("p11", 0),
                            Expr::var("w11"),
                        ),
                    ),
                ),
                Expr::u32(8),
            )),
        ),
        // Interpolate G
        Node::let_bind(
            "out_g",
            vyre_libs_builder::builder::stencil::clamp_u8(Expr::shr(
                Expr::add(
                    Expr::add(
                        Expr::mul(
                            vyre_libs_builder::builder::stencil::unpack_channel("p00", 8),
                            Expr::var("w00"),
                        ),
                        Expr::mul(
                            vyre_libs_builder::builder::stencil::unpack_channel("p10", 8),
                            Expr::var("w10"),
                        ),
                    ),
                    Expr::add(
                        Expr::mul(
                            vyre_libs_builder::builder::stencil::unpack_channel("p01", 8),
                            Expr::var("w01"),
                        ),
                        Expr::mul(
                            vyre_libs_builder::builder::stencil::unpack_channel("p11", 8),
                            Expr::var("w11"),
                        ),
                    ),
                ),
                Expr::u32(8),
            )),
        ),
        // Interpolate B
        Node::let_bind(
            "out_b",
            vyre_libs_builder::builder::stencil::clamp_u8(Expr::shr(
                Expr::add(
                    Expr::add(
                        Expr::mul(
                            vyre_libs_builder::builder::stencil::unpack_channel("p00", 16),
                            Expr::var("w00"),
                        ),
                        Expr::mul(
                            vyre_libs_builder::builder::stencil::unpack_channel("p10", 16),
                            Expr::var("w10"),
                        ),
                    ),
                    Expr::add(
                        Expr::mul(
                            vyre_libs_builder::builder::stencil::unpack_channel("p01", 16),
                            Expr::var("w01"),
                        ),
                        Expr::mul(
                            vyre_libs_builder::builder::stencil::unpack_channel("p11", 16),
                            Expr::var("w11"),
                        ),
                    ),
                ),
                Expr::u32(8),
            )),
        ),
        // Interpolate A
        Node::let_bind(
            "out_a",
            vyre_libs_builder::builder::stencil::clamp_u8(Expr::shr(
                Expr::add(
                    Expr::add(
                        Expr::mul(
                            vyre_libs_builder::builder::stencil::unpack_channel("p00", 24),
                            Expr::var("w00"),
                        ),
                        Expr::mul(
                            vyre_libs_builder::builder::stencil::unpack_channel("p10", 24),
                            Expr::var("w10"),
                        ),
                    ),
                    Expr::add(
                        Expr::mul(
                            vyre_libs_builder::builder::stencil::unpack_channel("p01", 24),
                            Expr::var("w01"),
                        ),
                        Expr::mul(
                            vyre_libs_builder::builder::stencil::unpack_channel("p11", 24),
                            Expr::var("w11"),
                        ),
                    ),
                ),
                Expr::u32(8),
            )),
        ),
        // Pack
        Node::let_bind(
            "packed_resample",
            Expr::bitor(
                Expr::var("out_r"),
                Expr::bitor(
                    Expr::shl(Expr::var("out_g"), Expr::u32(8)),
                    Expr::bitor(
                        Expr::shl(Expr::var("out_b"), Expr::u32(16)),
                        Expr::shl(Expr::var("out_a"), Expr::u32(24)),
                    ),
                ),
            ),
        ),
        Node::store(output, Expr::var("idx"), Expr::var("packed_resample")),
    ];

    Program::wrapped(
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(in_count),
            BufferDecl::output(output, 1, DataType::U32).with_count(out_count),
        ],
        super::PIXEL_WORKGROUP_SIZE,
        vec![wrap_anonymous_region(
            OP_ID,
            vec![wrap_child_region(
                OP_ID,
                Ident::from(OP_ID),
                vec![
                    Node::let_bind("idx_guard", Expr::logical_index(0)),
                    Node::if_then(Expr::lt(Expr::var("idx_guard"), Expr::u32(out_count)), body),
                ],
            )],
        )],
    )
}

const EXPECTED_RESAMPLE_OUTPUT_BYTES: [u8; 16] = [
    0xFF, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0xFF,
];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID,
        || bilinear_resample_rgba("in", 1, 1, "out", 2, 2),
        Some(|| {
            let pixels = [0xFF00_00FFu32]; // red
            vec![vec![
                vyre_primitives::wire::pack_u32_slice(&pixels),
            ]]
        }),
        Some(|| {
            vec![vec![EXPECTED_RESAMPLE_OUTPUT_BYTES.to_vec()]]
        }),
    )
    .with_category("visual")
    .with_uncharacterized()
}
