//! 2D bounding-box viewport and frustum culling compositions.
//!
//! Evaluates visibility for thousands of scene-graph nodes and draw items
//! against active clip viewports in parallel.
//!
//! Category A composition - pure IR over existing expressions.

use vyre_foundation::composition::{wrap_anonymous_region, wrap_child_region};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Ident, Node, Program};

const OP_ID: &str = "vyre-libs::visual::cull_boxes_2d";

/// Build a Program that culls 2D bounding boxes against a viewport rectangle.
///
/// `boxes` is `[i32; count * 4]` in `[x0, y0, x1, y1]` format.
/// `visible_mask` is `[u32; count]` output flags (1 = visible, 0 = culled).
#[must_use]
pub fn cull_boxes_2d(
    boxes: &str,
    count: u32,
    min_x: i32,
    min_y: i32,
    max_x: i32,
    max_y: i32,
    visible_mask: &str,
) -> Program {
    let body = vec![
        Node::let_bind("idx", Expr::logical_index(0)),
        Node::let_bind("base", Expr::mul(Expr::var("idx"), Expr::u32(4))),
        Node::let_bind("x0", Expr::load(boxes, Expr::var("base"))),
        Node::let_bind(
            "y0",
            Expr::load(boxes, Expr::add(Expr::var("base"), Expr::u32(1))),
        ),
        Node::let_bind(
            "x1",
            Expr::load(boxes, Expr::add(Expr::var("base"), Expr::u32(2))),
        ),
        Node::let_bind(
            "y1",
            Expr::load(boxes, Expr::add(Expr::var("base"), Expr::u32(3))),
        ),
        Node::let_bind(
            "visible",
            Expr::and(
                Expr::and(
                    Expr::ge(Expr::var("x1"), Expr::i32(min_x)),
                    Expr::lt(Expr::var("x0"), Expr::i32(max_x)),
                ),
                Expr::and(
                    Expr::ge(Expr::var("y1"), Expr::i32(min_y)),
                    Expr::lt(Expr::var("y0"), Expr::i32(max_y)),
                ),
            ),
        ),
        Node::let_bind(
            "flag",
            Expr::select(Expr::var("visible"), Expr::u32(1), Expr::u32(0)),
        ),
        Node::store(visible_mask, Expr::var("idx"), Expr::var("flag")),
    ];

    Program::wrapped(
        vec![
            BufferDecl::storage(boxes, 0, BufferAccess::ReadOnly, DataType::I32)
                .with_count(count * 4),
            BufferDecl::output(visible_mask, 1, DataType::U32).with_count(count),
        ],
        [256, 1, 1],
        vec![wrap_anonymous_region(
            OP_ID,
            vec![wrap_child_region(
                OP_ID,
                Ident::from(OP_ID),
                vec![
                    Node::let_bind("idx_guard", Expr::logical_index(0)),
                    Node::if_then(Expr::lt(Expr::var("idx_guard"), Expr::u32(count)), body),
                ],
            )],
        )],
    )
}

const EXPECTED_CULL_OUTPUT_BYTES: [u8; 8] = [0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID,
        || cull_boxes_2d("boxes", 2, 0, 0, 100, 100, "mask"),
        Some(|| {
            let boxes: [i32; 8] = [
                10, 10, 50, 50,    // Inside [0..100, 0..100] -> visible
                200, 200, 300, 300 // Outside -> culled
            ];
            let box_bytes: Vec<u8> = boxes.iter().flat_map(|x| x.to_le_bytes()).collect();
            vec![vec![box_bytes]]
        }),
        Some(|| {
            vec![vec![EXPECTED_CULL_OUTPUT_BYTES.to_vec()]]
        }),
    )
    .with_category("visual")
    .with_no_legal_rewrite()
}
