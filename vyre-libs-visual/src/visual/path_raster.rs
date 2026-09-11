//! Vector path and polygon rasterization compositions.
//!
//! Evaluates analytical coverage and signed distance for 2D line segments,
//! strokes, and closed polygon boundaries against a discrete pixel grid.
//!
//! Category A composition - pure IR over existing expressions.

use vyre_foundation::composition::{wrap_anonymous_region, wrap_child_region};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Ident, Node, Program};

const OP_ID: &str = "vyre-libs::visual::path_raster";

/// Build a Program that rasterizes vector line segments onto a target framebuffer.
///
/// `segments` is `[u32; segment_count * 4]` in `[x0, y0, x1, y1]` format.
/// `bg` is `[u32; width * height]` background packed RGBA.
/// `output` is `[u32; width * height]` output packed RGBA.
#[must_use]
pub fn path_rasterize_segments(
    segments: &str,
    bg: &str,
    output: &str,
    width: u32,
    height: u32,
    segment_count: u32,
    stroke_radius: u32,
    stroke_color: u32,
) -> Program {
    let pixel_count = width * height;
    let rad_sq = stroke_radius.saturating_sub(1) * stroke_radius.saturating_sub(1);

    let mut body = vec![
        Node::let_bind("idx", Expr::logical_index(0)),
        Node::let_bind("px", Expr::rem(Expr::var("idx"), Expr::u32(width))),
        Node::let_bind("py", Expr::div(Expr::var("idx"), Expr::u32(width))),
        Node::let_bind("bg_px", Expr::load(bg, Expr::var("idx"))),
        Node::let_bind("hit_0", Expr::u32(0)),
    ];

    for s in 0..segment_count {
        let base = s * 4;
        let s_hit = format!("s_hit_{s}");
        let s_x0 = format!("s_x0_{s}");
        let s_y0 = format!("s_y0_{s}");
        let s_x1 = format!("s_x1_{s}");
        let s_y1 = format!("s_y1_{s}");
        let s_vx = format!("s_vx_{s}");
        let s_vy = format!("s_vy_{s}");
        let s_ux = format!("s_ux_{s}");
        let s_uy = format!("s_uy_{s}");
        let s_lensq = format!("s_lensq_{s}");
        let s_dot = format!("s_dot_{s}");
        let s_dist = format!("s_dist_{s}");

        body.extend(vec![
            Node::let_bind(&s_x0, Expr::load(segments, Expr::u32(base))),
            Node::let_bind(&s_y0, Expr::load(segments, Expr::u32(base + 1))),
            Node::let_bind(&s_x1, Expr::load(segments, Expr::u32(base + 2))),
            Node::let_bind(&s_y1, Expr::load(segments, Expr::u32(base + 3))),
            // vx = x1 - x0, vy = y1 - y0 (using i32 arithmetic)
            Node::let_bind(&s_vx, Expr::sub(Expr::var(&s_x1), Expr::var(&s_x0))),
            Node::let_bind(&s_vy, Expr::sub(Expr::var(&s_y1), Expr::var(&s_y0))),
            Node::let_bind(&s_ux, Expr::sub(Expr::var("px"), Expr::var(&s_x0))),
            Node::let_bind(&s_uy, Expr::sub(Expr::var("py"), Expr::var(&s_y0))),
            Node::let_bind(
                &s_lensq,
                Expr::add(
                    Expr::mul(Expr::var(&s_vx), Expr::var(&s_vx)),
                    Expr::mul(Expr::var(&s_vy), Expr::var(&s_vy)),
                ),
            ),
            Node::let_bind(
                &s_dot,
                Expr::add(
                    Expr::mul(Expr::var(&s_ux), Expr::var(&s_vx)),
                    Expr::mul(Expr::var(&s_uy), Expr::var(&s_vy)),
                ),
            ),
            Node::let_bind(
                &s_dist,
                Expr::select(
                    Expr::le(Expr::var(&s_lensq), Expr::u32(0)),
                    Expr::add(
                        Expr::mul(Expr::var(&s_ux), Expr::var(&s_ux)),
                        Expr::mul(Expr::var(&s_uy), Expr::var(&s_uy)),
                    ),
                    Expr::select(
                        Expr::le(Expr::var(&s_dot), Expr::u32(0)),
                        Expr::add(
                            Expr::mul(Expr::var(&s_ux), Expr::var(&s_ux)),
                            Expr::mul(Expr::var(&s_uy), Expr::var(&s_uy)),
                        ),
                        Expr::select(
                            Expr::ge(Expr::var(&s_dot), Expr::var(&s_lensq)),
                            Expr::add(
                                Expr::mul(
                                    Expr::sub(Expr::var("px"), Expr::var(&s_x1)),
                                    Expr::sub(Expr::var("px"), Expr::var(&s_x1)),
                                ),
                                Expr::mul(
                                    Expr::sub(Expr::var("py"), Expr::var(&s_y1)),
                                    Expr::sub(Expr::var("py"), Expr::var(&s_y1)),
                                ),
                            ),
                            Expr::sub(
                                Expr::add(
                                    Expr::mul(Expr::var(&s_ux), Expr::var(&s_ux)),
                                    Expr::mul(Expr::var(&s_uy), Expr::var(&s_uy)),
                                ),
                                Expr::div(
                                    Expr::mul(Expr::var(&s_dot), Expr::var(&s_dot)),
                                    Expr::var(&s_lensq),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            Node::let_bind(
                &s_hit,
                Expr::select(
                    Expr::le(Expr::var(&s_dist), Expr::u32(rad_sq)),
                    Expr::u32(1),
                    Expr::var(format!("hit_{s}")),
                ),
            ),
            Node::let_bind(format!("hit_{}", s + 1), Expr::var(&s_hit)),
        ]);
    }

    body.push(Node::let_bind(
        "final_color",
        Expr::select(
            Expr::gt(Expr::var(format!("hit_{segment_count}")), Expr::u32(0)),
            Expr::u32(stroke_color),
            Expr::var("bg_px"),
        ),
    ));
    body.push(Node::store(
        output,
        Expr::var("idx"),
        Expr::var("final_color"),
    ));

    Program::wrapped(
        vec![
            BufferDecl::storage(segments, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(segment_count * 4),
            BufferDecl::storage(bg, 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(pixel_count),
            BufferDecl::output(output, 2, DataType::U32).with_count(pixel_count),
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

const EXPECTED_PATH_OUTPUT_BYTES: [u8; 16] = [
    0xFF, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00, 0x00, 0xFF, 0x00, 0x00, 0x00, 0xFF,
];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID,
        || path_rasterize_segments("segments", "bg", "out", 2, 2, 1, 1, 0xFF00_00FF),
        Some(|| {
            let segs = [0u32, 0, 1, 0];
            let bg_pixels = [0xFF00_0000u32; 4];
            vec![vec![
                vyre_primitives::wire::pack_u32_slice(&segs),
                vyre_primitives::wire::pack_u32_slice(&bg_pixels),
            ]]
        }),
        Some(|| {
            vec![vec![EXPECTED_PATH_OUTPUT_BYTES.to_vec()]]
        }),
    )
    .with_category("visual")
    .with_no_legal_rewrite()
}
