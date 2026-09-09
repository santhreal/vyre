//! Layout-adjacent prefix scans and bounding-box reduction compositions.
//!
//! Computes cumulative flow layout offsets and bounds reduction for flex/grid
//! UI layouts and text wrapping.
//!
//! Category A composition - pure IR over existing expressions.

use vyre_foundation::composition::{wrap_anonymous_region, wrap_child_region};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Ident, Node, Program};

const OP_ID_SCAN: &str = "vyre-libs::visual::layout_prefix_scan";
const OP_ID_REDUCE: &str = "vyre-libs::visual::reduce_bounding_boxes";

/// Build a Program that computes an exclusive prefix scan of layout element sizes.
///
/// `offsets[i] = sum_{j=0}^{i-1} sizes[j]`, with `offsets[0] = 0`.
#[must_use]
pub fn layout_prefix_scan_u32(sizes: &str, offsets: &str, count: u32) -> Program {
    let body = vec![
        Node::let_bind("idx", Expr::logical_index(0)),
        Node::let_bind("acc", Expr::u32(0)),
        Node::loop_for(
            "j",
            Expr::u32(0),
            Expr::var("idx"),
            vec![
                Node::let_bind("s_j", Expr::load(sizes, Expr::var("j"))),
                Node::assign("acc", Expr::add(Expr::var("acc"), Expr::var("s_j"))),
            ],
        ),
        Node::store(offsets, Expr::var("idx"), Expr::var("acc")),
    ];

    Program::wrapped(
        vec![
            BufferDecl::storage(sizes, 0, BufferAccess::ReadOnly, DataType::U32).with_count(count),
            BufferDecl::output(offsets, 1, DataType::U32).with_count(count),
        ],
        [256, 1, 1],
        vec![wrap_anonymous_region(
            OP_ID_SCAN,
            vec![wrap_child_region(
                OP_ID_SCAN,
                Ident::from(OP_ID_SCAN),
                vec![
                    Node::let_bind("idx_guard", Expr::logical_index(0)),
                    Node::if_then(Expr::lt(Expr::var("idx_guard"), Expr::u32(count)), body),
                ],
            )],
        )],
    )
}

/// Build a Program that computes the minimum enclosing bounding box for a set of 2D boxes.
///
/// `boxes` is `[i32; count * 4]`.
/// `output_bbox` is `[i32; 4]` storing `[min_x0, min_y0, max_x1, max_y1]`.
#[must_use]
pub fn reduce_bounding_boxes_2d(boxes: &str, count: u32, output_bbox: &str) -> Program {
    let body = vec![
        Node::let_bind("min_x", Expr::load(boxes, Expr::u32(0))),
        Node::let_bind("min_y", Expr::load(boxes, Expr::u32(1))),
        Node::let_bind("max_x", Expr::load(boxes, Expr::u32(2))),
        Node::let_bind("max_y", Expr::load(boxes, Expr::u32(3))),
        Node::loop_for(
            "b",
            Expr::u32(1),
            Expr::u32(count),
            vec![
                Node::let_bind("base", Expr::mul(Expr::var("b"), Expr::u32(4))),
                Node::let_bind("bx0", Expr::load(boxes, Expr::var("base"))),
                Node::let_bind(
                    "by0",
                    Expr::load(boxes, Expr::add(Expr::var("base"), Expr::u32(1))),
                ),
                Node::let_bind(
                    "bx1",
                    Expr::load(boxes, Expr::add(Expr::var("base"), Expr::u32(2))),
                ),
                Node::let_bind(
                    "by1",
                    Expr::load(boxes, Expr::add(Expr::var("base"), Expr::u32(3))),
                ),
                Node::assign(
                    "min_x",
                    Expr::select(
                        Expr::lt(Expr::var("bx0"), Expr::var("min_x")),
                        Expr::var("bx0"),
                        Expr::var("min_x"),
                    ),
                ),
                Node::assign(
                    "min_y",
                    Expr::select(
                        Expr::lt(Expr::var("by0"), Expr::var("min_y")),
                        Expr::var("by0"),
                        Expr::var("min_y"),
                    ),
                ),
                Node::assign(
                    "max_x",
                    Expr::select(
                        Expr::gt(Expr::var("bx1"), Expr::var("max_x")),
                        Expr::var("bx1"),
                        Expr::var("max_x"),
                    ),
                ),
                Node::assign(
                    "max_y",
                    Expr::select(
                        Expr::gt(Expr::var("by1"), Expr::var("max_y")),
                        Expr::var("by1"),
                        Expr::var("max_y"),
                    ),
                ),
            ],
        ),
        Node::store(output_bbox, Expr::u32(0), Expr::var("min_x")),
        Node::store(output_bbox, Expr::u32(1), Expr::var("min_y")),
        Node::store(output_bbox, Expr::u32(2), Expr::var("max_x")),
        Node::store(output_bbox, Expr::u32(3), Expr::var("max_y")),
    ];

    Program::wrapped(
        vec![
            BufferDecl::storage(boxes, 0, BufferAccess::ReadOnly, DataType::I32)
                .with_count(count * 4),
            BufferDecl::output(output_bbox, 1, DataType::I32).with_count(4),
        ],
        [1, 1, 1],
        vec![wrap_anonymous_region(
            OP_ID_REDUCE,
            vec![wrap_child_region(
                OP_ID_REDUCE,
                Ident::from(OP_ID_REDUCE),
                body,
            )],
        )],
    )
}

const EXPECTED_SCAN_OUTPUT_BYTES: [u8; 16] = [
    0x00, 0x00, 0x00, 0x00, 0x0A, 0x00, 0x00, 0x00, 0x1E, 0x00, 0x00, 0x00, 0x3C, 0x00, 0x00, 0x00,
];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID_SCAN,
        || layout_prefix_scan_u32("sizes", "offsets", 4),
        Some(|| {
            let sizes = [10u32, 20, 30, 40];
            vec![vec![
                vyre_primitives::wire::pack_u32_slice(&sizes),
                vec![0; 16],
            ]]
        }),
        Some(|| {
            vec![vec![EXPECTED_SCAN_OUTPUT_BYTES.to_vec()]]
        }),
    )
    .with_category("visual")
    .with_opaque("layout prefix scan sum")
}
