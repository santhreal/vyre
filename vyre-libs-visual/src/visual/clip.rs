//! Scissor rectangle, rounded rectangle, and alpha mask clipping compositions.
//!
//! Restricts rendering to bounding rectangles, arbitrary alpha masks, or
//! rounded corner primitives.
//!
//! Category A composition - pure IR over existing expressions.

use vyre_foundation::composition::{wrap_anonymous_region, wrap_child_region};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Ident, Node, Program};

const OP_ID_SCISSOR: &str = "vyre-libs::visual::apply_scissor_rect";
const OP_ID_MASK: &str = "vyre-libs::visual::apply_clip_mask";

/// Build a Program that applies a rectangular scissor clip to a packed RGBA surface.
///
/// Pixels outside `[min_x, min_y, max_x, max_y]` are cleared to 0 (fully transparent).
#[must_use]
pub fn apply_scissor_rect(
    input: &str,
    output: &str,
    width: u32,
    height: u32,
    min_x: u32,
    min_y: u32,
    max_x: u32,
    max_y: u32,
) -> Program {
    let count = width * height;
    let body = vec![
        Node::let_bind("idx", Expr::logical_index(0)),
        Node::let_bind("px", Expr::rem(Expr::var("idx"), Expr::u32(width))),
        Node::let_bind("py", Expr::div(Expr::var("idx"), Expr::u32(width))),
        Node::let_bind("in_px", Expr::load(input, Expr::var("idx"))),
        Node::let_bind(
            "inside",
            Expr::and(
                Expr::and(
                    Expr::ge(Expr::var("px"), Expr::u32(min_x)),
                    Expr::lt(Expr::var("px"), Expr::u32(max_x)),
                ),
                Expr::and(
                    Expr::ge(Expr::var("py"), Expr::u32(min_y)),
                    Expr::lt(Expr::var("py"), Expr::u32(max_y)),
                ),
            ),
        ),
        Node::let_bind(
            "out_px",
            Expr::select(Expr::var("inside"), Expr::var("in_px"), Expr::u32(0)),
        ),
        Node::store(output, Expr::var("idx"), Expr::var("out_px")),
    ];

    Program::wrapped(
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::U32).with_count(count),
            BufferDecl::output(output, 1, DataType::U32).with_count(count),
        ],
        super::PIXEL_WORKGROUP_SIZE,
        vec![wrap_anonymous_region(
            OP_ID_SCISSOR,
            vec![wrap_child_region(
                OP_ID_SCISSOR,
                Ident::from(OP_ID_SCISSOR),
                vec![
                    Node::let_bind("idx_guard", Expr::logical_index(0)),
                    Node::if_then(Expr::lt(Expr::var("idx_guard"), Expr::u32(count)), body),
                ],
            )],
        )],
    )
}

/// Build a Program that modulates the alpha channel of `input` by a coverage `mask`.
#[must_use]
pub fn apply_clip_mask(input: &str, mask: &str, output: &str, count: u32) -> Program {
    let body = vec![
        Node::let_bind("in_px", Expr::load(input, Expr::var("idx"))),
        Node::let_bind("mask_px", Expr::load(mask, Expr::var("idx"))),
        Node::let_bind(
            "r",
            vyre_libs_builder::builder::stencil::unpack_channel("in_px", 0),
        ),
        Node::let_bind(
            "g",
            vyre_libs_builder::builder::stencil::unpack_channel("in_px", 8),
        ),
        Node::let_bind(
            "b",
            vyre_libs_builder::builder::stencil::unpack_channel("in_px", 16),
        ),
        Node::let_bind(
            "a",
            vyre_libs_builder::builder::stencil::unpack_channel("in_px", 24),
        ),
        Node::let_bind(
            "mask_a",
            vyre_libs_builder::builder::stencil::unpack_channel("mask_px", 24),
        ),
        Node::let_bind(
            "new_a",
            vyre_libs_builder::builder::stencil::clamp_u8(Expr::div(
                Expr::add(
                    Expr::mul(Expr::var("a"), Expr::var("mask_a")),
                    Expr::u32(127),
                ),
                Expr::u32(255),
            )),
        ),
        Node::let_bind(
            "out_px",
            Expr::bitor(
                Expr::var("r"),
                Expr::bitor(
                    Expr::shl(Expr::var("g"), Expr::u32(8)),
                    Expr::bitor(
                        Expr::shl(Expr::var("b"), Expr::u32(16)),
                        Expr::shl(Expr::var("new_a"), Expr::u32(24)),
                    ),
                ),
            ),
        ),
        Node::store(output, Expr::var("idx"), Expr::var("out_px")),
    ];

    crate::visual::packed_rgba_map::build_pixel_pipeline(
        OP_ID_MASK,
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::U32).with_count(count),
            BufferDecl::storage(mask, 1, BufferAccess::ReadOnly, DataType::U32).with_count(count),
            BufferDecl::output(output, 2, DataType::U32).with_count(count),
        ],
        count,
        body,
    )
}

const EXPECTED_SCISSOR_OUTPUT_BYTES: [u8; 16] = [
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0x00, 0x00, 0xFF,
];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID_SCISSOR,
        || apply_scissor_rect("in", "out", 2, 2, 1, 1, 2, 2),
        Some(|| {
            let pixels = [0xFF00_00FFu32; 4];
            vec![vec![
                vyre_primitives::wire::pack_u32_slice(&pixels),
            ]]
        }),
        Some(|| {
            vec![vec![EXPECTED_SCISSOR_OUTPUT_BYTES.to_vec()]]
        }),
    )
    .with_category("visual")
    .with_no_legal_rewrite()
}
