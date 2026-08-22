//! Composable per-pixel filter chain.
//!
//! Applies brightness, contrast, saturate, and invert in sequence.
//! All math is integer fixed-point 16.16. Category A  -  pure IR.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

const OP_ID: &str = "vyre-libs::visual::filter_chain";

/// Build a Program that applies a filter chain to `pixels` in-place.
///
/// - `pixels`: `[u32; count]`  -  packed RGBA, modified in-place
/// - `brightness`, `contrast`, `saturate`: float ratios (1.0 = identity)
/// - `invert`: 0.0 = no invert, 1.0 = full invert
#[must_use]
pub fn filter_chain(
    pixels: &str,
    count: u32,
    brightness: f32,
    contrast: f32,
    saturate: f32,
    invert: f32,
) -> Program {
    let br_fp = (brightness * 65536.0).round() as u32;
    let ct_fp = (contrast * 65536.0).round() as u32;
    let sat_fp = (saturate * 65536.0).round() as u32;
    let inv_fp = (invert.clamp(0.0, 1.0) * 65536.0).round() as u32;

    // BT.709 luma coefficients in fixed-point 16.16:
    // 0.2126 * 65536 = 13933
    // 0.7152 * 65536 = 46871
    // 0.0722 * 65536 = 4732
    let luma_r: u32 = 13933;
    let luma_g: u32 = 46871;
    let luma_b: u32 = 4732;

    // Helper: clamp to [0, 255] using select
    let clamp255 = |name: &str| -> Vec<Node> {
        vec![Node::assign(
            name,
            crate::builder::stencil::clamp_u8(Expr::var(name)),
        )]
    };

    let mut body = vec![
        Node::let_bind("pixel", Expr::load(pixels, Expr::var("idx"))),
        // Unpack RGBA.
        Node::let_bind("r", crate::builder::stencil::unpack_channel("pixel", 0)),
        Node::let_bind("g", crate::builder::stencil::unpack_channel("pixel", 8)),
        Node::let_bind("b", crate::builder::stencil::unpack_channel("pixel", 16)),
        Node::let_bind("a", crate::builder::stencil::unpack_channel("pixel", 24)),
        // 1. Brightness: channel = channel * brightness >> 16
        Node::assign(
            "r",
            super::fixed_mul_16_16_unsigned_expr(Expr::var("r"), Expr::u32(br_fp)),
        ),
        Node::assign(
            "g",
            super::fixed_mul_16_16_unsigned_expr(Expr::var("g"), Expr::u32(br_fp)),
        ),
        Node::assign(
            "b",
            super::fixed_mul_16_16_unsigned_expr(Expr::var("b"), Expr::u32(br_fp)),
        ),
    ];

    body.extend(clamp255("r"));
    body.extend(clamp255("g"));
    body.extend(clamp255("b"));

    // 2. Contrast: channel = ((channel - 128) * contrast >> 16) + 128
    // Both arms of a select evaluate, so a reverse subtraction written as the
    // unused arm still wraps. Split the distance from the midpoint into two
    // non-negative parts, exactly one of which is nonzero, and no intermediate
    // leaves range: `up` scales the above-midpoint distance, `down` the below.
    let contrast_adjust = |ch: &str| -> Vec<Node> {
        let up = format!("{ch}_cup");
        let down = format!("{ch}_cdn");
        let base = format!("{ch}_cbase");
        vec![
            Node::let_bind(
                &up,
                super::fixed_mul_16_16_unsigned_expr(
                    Expr::sub(Expr::max(Expr::var(ch), Expr::u32(128)), Expr::u32(128)),
                    Expr::u32(ct_fp),
                ),
            ),
            Node::let_bind(
                &down,
                super::fixed_mul_16_16_unsigned_expr(
                    Expr::sub(Expr::u32(128), Expr::min(Expr::var(ch), Expr::u32(128))),
                    Expr::u32(ct_fp),
                ),
            ),
            Node::let_bind(
                &base,
                Expr::select(
                    Expr::ge(Expr::u32(128), Expr::var(&down)),
                    Expr::sub(Expr::u32(128), Expr::var(&down)),
                    Expr::u32(0),
                ),
            ),
            Node::assign(ch, Expr::add(Expr::var(&base), Expr::var(&up))),
        ]
    };
    body.extend(contrast_adjust("r"));
    body.extend(contrast_adjust("g"));
    body.extend(contrast_adjust("b"));
    body.extend(clamp255("r"));
    body.extend(clamp255("g"));
    body.extend(clamp255("b"));

    // 3. Saturate: luma + (channel - luma) * saturate
    body.push(Node::let_bind(
        "luma",
        Expr::add(
            Expr::add(
                super::fixed_mul_16_16_unsigned_expr(Expr::var("r"), Expr::u32(luma_r)),
                super::fixed_mul_16_16_unsigned_expr(Expr::var("g"), Expr::u32(luma_g)),
            ),
            super::fixed_mul_16_16_unsigned_expr(Expr::var("b"), Expr::u32(luma_b)),
        ),
    ));

    // 3. Saturate: luma + (channel - luma) * saturate, in the same two-part
    // form as the contrast stage so no intermediate underflows.
    let saturate_ch = |ch: &str| -> Vec<Node> {
        let up = format!("{ch}_sup");
        let down = format!("{ch}_sdn");
        let base = format!("{ch}_sbase");
        vec![
            Node::let_bind(
                &up,
                super::fixed_mul_16_16_unsigned_expr(
                    Expr::sub(
                        Expr::max(Expr::var(ch), Expr::var("luma")),
                        Expr::var("luma"),
                    ),
                    Expr::u32(sat_fp),
                ),
            ),
            Node::let_bind(
                &down,
                super::fixed_mul_16_16_unsigned_expr(
                    Expr::sub(
                        Expr::var("luma"),
                        Expr::min(Expr::var(ch), Expr::var("luma")),
                    ),
                    Expr::u32(sat_fp),
                ),
            ),
            Node::let_bind(
                &base,
                Expr::select(
                    Expr::ge(Expr::var("luma"), Expr::var(&down)),
                    Expr::sub(Expr::var("luma"), Expr::var(&down)),
                    Expr::u32(0),
                ),
            ),
            Node::assign(ch, Expr::add(Expr::var(&base), Expr::var(&up))),
        ]
    };
    body.extend(saturate_ch("r"));
    body.extend(saturate_ch("g"));
    body.extend(saturate_ch("b"));
    body.extend(clamp255("r"));
    body.extend(clamp255("g"));
    body.extend(clamp255("b"));

    // 4. Invert: channel = channel*(1-inv) + (255-channel)*inv
    //    = channel + (255 - 2*channel) * inv >> 16
    if inv_fp > 0 {
        let invert_ch = |ch: &str| -> Vec<Node> {
            vec![Node::assign(
                ch,
                Expr::add(
                    super::fixed_mul_16_16_unsigned_expr(
                        Expr::var(ch),
                        Expr::sub(Expr::u32(65536), Expr::u32(inv_fp)),
                    ),
                    super::fixed_mul_16_16_unsigned_expr(
                        Expr::sub(Expr::u32(255), Expr::var(ch)),
                        Expr::u32(inv_fp),
                    ),
                ),
            )]
        };
        body.extend(invert_ch("r"));
        body.extend(invert_ch("g"));
        body.extend(invert_ch("b"));
        body.extend(clamp255("r"));
        body.extend(clamp255("g"));
        body.extend(clamp255("b"));
    }

    // Pack and write.
    body.push(Node::let_bind(
        "out",
        crate::builder::stencil::pack_rgba_named("r", "g", "b", "a"),
    ));
    body.push(Node::store(pixels, Expr::var("idx"), Expr::var("out")));
    crate::visual::packed_rgba_map::build_pixel_pipeline(
        OP_ID,
        vec![
            BufferDecl::storage(pixels, 0, BufferAccess::ReadWrite, DataType::U32)
                .with_count(count),
        ],
        count,
        body,
    )
}

const EXPECTED_FILTER_CHAIN_OUTPUT_BYTES: [u8; 16] = [
    0x20, 0x40, 0x80, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0xFF,
];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || filter_chain("pixels", 4, 1.0, 1.0, 1.0, 0.0),
        Some(|| {
            // Identity transform: all params = 1.0/0.0 → output == input.
            let pixels = [0xFF_804020u32, 0xFF_FF0000, 0xFF_00FF00, 0xFF_0000FF];
            vec![vec![crate::visual::u32_word_bytes::u32_words_to_le_bytes(&pixels)]]
        }),
        Some(|| {
            vec![vec![EXPECTED_FILTER_CHAIN_OUTPUT_BYTES.to_vec()]]
        }),
    )
    .with_category("visual")
}
