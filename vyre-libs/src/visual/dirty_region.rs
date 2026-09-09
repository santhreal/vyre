//! Retained resource sub-rectangle dirty-region patch compositions.
//!
//! Applies localized texture/framebuffer updates from dirty rectangle regions
//! into retained atlases and render targets without full-surface reallocation.
//!
//! Category A composition - pure IR over existing expressions.

use vyre_foundation::composition::{wrap_anonymous_region, wrap_child_region};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Ident, Node, Program};

const OP_ID_PATCH: &str = "vyre-libs::visual::dirty_region_patch";
const OP_ID_DIRECT: &str = "vyre-libs::visual::dirty_region_patch_direct";

/// Build a Program that copies `atlas` to `output` while patching a sub-rectangle from `patch`.
#[must_use]
pub fn dirty_region_patch_rgba(
    atlas: &str,
    patch: &str,
    atlas_w: u32,
    atlas_h: u32,
    patch_w: u32,
    patch_h: u32,
    dest_x: u32,
    dest_y: u32,
    output: &str,
) -> Program {
    let count = atlas_w * atlas_h;
    let patch_count = patch_w * patch_h;

    let body = vec![
        Node::let_bind("idx", Expr::logical_index(0)),
        Node::let_bind("ax", Expr::rem(Expr::var("idx"), Expr::u32(atlas_w))),
        Node::let_bind("ay", Expr::div(Expr::var("idx"), Expr::u32(atlas_w))),
        Node::let_bind("atlas_px", Expr::load(atlas, Expr::var("idx"))),
        Node::let_bind(
            "in_patch",
            Expr::and(
                Expr::and(
                    Expr::ge(Expr::var("ax"), Expr::u32(dest_x)),
                    Expr::lt(Expr::var("ax"), Expr::u32(dest_x + patch_w)),
                ),
                Expr::and(
                    Expr::ge(Expr::var("ay"), Expr::u32(dest_y)),
                    Expr::lt(Expr::var("ay"), Expr::u32(dest_y + patch_h)),
                ),
            ),
        ),
        Node::let_bind(
            "patch_x",
            Expr::sub(Expr::var("ax"), Expr::u32(dest_x)),
        ),
        Node::let_bind(
            "patch_y",
            Expr::sub(Expr::var("ay"), Expr::u32(dest_y)),
        ),
        Node::let_bind(
            "patch_idx",
            Expr::add(
                Expr::var("patch_x"),
                Expr::mul(Expr::var("patch_y"), Expr::u32(patch_w)),
            ),
        ),
        Node::let_bind(
            "patch_px",
            Expr::select(
                Expr::var("in_patch"),
                Expr::load(patch, Expr::var("patch_idx")),
                Expr::u32(0),
            ),
        ),
        Node::let_bind(
            "out_px",
            Expr::select(
                Expr::var("in_patch"),
                Expr::var("patch_px"),
                Expr::var("atlas_px"),
            ),
        ),
        Node::store(output, Expr::var("idx"), Expr::var("out_px")),
    ];

    Program::wrapped(
        vec![
            BufferDecl::storage(atlas, 0, BufferAccess::ReadOnly, DataType::U32).with_count(count),
            BufferDecl::storage(patch, 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(patch_count),
            BufferDecl::output(output, 2, DataType::U32).with_count(count),
        ],
        super::PIXEL_WORKGROUP_SIZE,
        vec![wrap_anonymous_region(
            OP_ID_PATCH,
            vec![wrap_child_region(
                OP_ID_PATCH,
                Ident::from(OP_ID_PATCH),
                vec![
                    Node::let_bind("idx_guard", Expr::logical_index(0)),
                    Node::if_then(Expr::lt(Expr::var("idx_guard"), Expr::u32(count)), body),
                ],
            )],
        )],
    )
}

/// Build a Program that directly blits `patch` into destination `target` at `(dest_x, dest_y)`.
#[must_use]
pub fn dirty_region_patch_direct(
    patch: &str,
    patch_w: u32,
    patch_h: u32,
    dest_x: u32,
    dest_y: u32,
    target: &str,
    target_w: u32,
    target_h: u32,
) -> Program {
    let patch_count = patch_w * patch_h;
    let target_count = target_w * target_h;

    let body = vec![
        Node::let_bind("idx", Expr::logical_index(0)),
        Node::let_bind("px", Expr::rem(Expr::var("idx"), Expr::u32(patch_w))),
        Node::let_bind("py", Expr::div(Expr::var("idx"), Expr::u32(patch_w))),
        Node::let_bind("tx", Expr::add(Expr::var("px"), Expr::u32(dest_x))),
        Node::let_bind("ty", Expr::add(Expr::var("py"), Expr::u32(dest_y))),
        Node::let_bind(
            "target_idx",
            Expr::add(Expr::var("tx"), Expr::mul(Expr::var("ty"), Expr::u32(target_w))),
        ),
        Node::let_bind("patch_px", Expr::load(patch, Expr::var("idx"))),
        Node::if_then(
            Expr::and(
                Expr::lt(Expr::var("tx"), Expr::u32(target_w)),
                Expr::lt(Expr::var("ty"), Expr::u32(target_h)),
            ),
            vec![Node::store(target, Expr::var("target_idx"), Expr::var("patch_px"))],
        ),
    ];

    Program::wrapped(
        vec![
            BufferDecl::storage(patch, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(patch_count),
            BufferDecl::storage(target, 1, BufferAccess::ReadWrite, DataType::U32)
                .with_count(target_count),
        ],
        super::PIXEL_WORKGROUP_SIZE,
        vec![wrap_anonymous_region(
            OP_ID_DIRECT,
            vec![wrap_child_region(
                OP_ID_DIRECT,
                Ident::from(OP_ID_DIRECT),
                vec![
                    Node::let_bind("idx_guard", Expr::logical_index(0)),
                    Node::if_then(Expr::lt(Expr::var("idx_guard"), Expr::u32(patch_count)), body),
                ],
            )],
        )],
    )
}

const EXPECTED_PATCH_OUTPUT_BYTES: [u8; 16] = [
    0x00, 0x00, 0x00, 0xFF, 0x00, 0x00, 0x00, 0xFF, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0xFF,
];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID_PATCH,
        || dirty_region_patch_rgba("atlas", "patch", 2, 2, 1, 1, 1, 1, "out"),
        Some(|| {
            let atlas = [0xFF00_0000u32; 4]; // black
            let patch = [0xFF00_00FFu32]; // red at (1,1)
            vec![vec![
                vyre_primitives::wire::pack_u32_slice(&atlas),
                vyre_primitives::wire::pack_u32_slice(&patch),
                vec![0; 16],
            ]]
        }),
        Some(|| {
            vec![vec![EXPECTED_PATCH_OUTPUT_BYTES.to_vec()]]
        }),
    )
    .with_category("visual")
    .with_opaque("retained resource sub-rectangle dirty region patch")
}
