//! Packed-RGBA per-pixel map skeleton.
//!
//! Higher-level visual ops specialize the pixel expression, but they all
//! share the same execution shape: one invocation reads or derives one
//! packed `u32` RGBA pixel and writes one packed `u32` pixel.

use vyre_foundation::composition::{wrap_anonymous_region, wrap_child_region};
use vyre_foundation::ir::Ident;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Stable Tier 2.5 op id.
pub const OP_ID: &str = "vyre-libs::visual::packed_rgba_map";

/// Emit a generic identity packed-RGBA map node.
#[must_use]
pub fn packed_rgba_map_node(input: &str, output: &str, count: u32) -> Node {
    wrap_anonymous_region(
        OP_ID,
        vec![
            Node::let_bind("idx", Expr::gid_x()),
            Node::if_then(
                Expr::lt(Expr::var("idx"), Expr::u32(count)),
                vec![
                    Node::let_bind("pixel", Expr::load(input, Expr::var("idx"))),
                    Node::store(output, Expr::var("idx"), Expr::var("pixel")),
                ],
            ),
        ],
    )
}

/// Standalone identity packed-RGBA map Program.
#[must_use]
pub fn packed_rgba_map(input: &str, output: &str, count: u32) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::U32).with_count(count),
            BufferDecl::storage(output, 1, BufferAccess::ReadWrite, DataType::U32)
                .with_count(count),
        ],
        [256, 1, 1],
        vec![packed_rgba_map_node(input, output, count)],
    )
}

/// Build a standard packed-RGBA visual compute program with nested child region.
pub(crate) fn build_pixel_pipeline(
    op_id: &'static str,
    buffers: Vec<BufferDecl>,
    pixel_count: u32,
    body: Vec<Node>,
) -> Program {
    Program::wrapped(
        buffers,
        super::PIXEL_WORKGROUP_SIZE,
        vec![wrap_anonymous_region(
            op_id,
            vec![wrap_child_region(
                OP_ID,
                Ident::from(op_id),
                vec![
                    Node::let_bind("idx", Expr::gid_x()),
                    Node::if_then(Expr::lt(Expr::var("idx"), Expr::u32(pixel_count)), body),
                ],
            )],
        )],
    )
}

const EXPECTED_PACKED_RGBA_MAP_OUTPUT_BYTES: [u8; 16] = [
    0x00, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0x00, 0xFF, 0xFF,
];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || packed_rgba_map("in", "out", 4),
        Some(|| {
            let pixels = [0xFF00_0000u32, 0xFF00_00FF, 0xFF00_FF00, 0xFFFF_0000];
            let bytes = vyre_primitives::wire::pack_u32_slice(&pixels);
            vec![vec![bytes, vec![0; 16]]]
        }),
        Some(|| {
            vec![vec![EXPECTED_PACKED_RGBA_MAP_OUTPUT_BYTES.to_vec()]]
        }),
    )
}
