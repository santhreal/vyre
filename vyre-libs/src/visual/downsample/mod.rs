//! 2× box-filter downsample for half-resolution blur.
//!
//! Averages each 2×2 block of pixels into one output pixel.
//! Category A composition  -  pure IR. No shared primitives.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

const OP_ID: &str = "vyre-libs::visual::downsample";

/// Build a Program that 2× downsamples `input` into `output`.
///
/// - `input`:  `[u32; width * height]`  -  source pixels (packed RGBA)
/// - `output`: `[u32; (width/2) * (height/2)]`  -  downsampled result
/// - Width and height must be even.
#[must_use]
pub fn downsample_2x(input: &str, output: &str, width: u32, height: u32) -> Program {
    let out_w = width / 2;
    let out_h = height / 2;
    let input_count = width.saturating_mul(height);
    let output_count = out_w.saturating_mul(out_h);

    crate::builder::stencil::Grid2DComposer::new(OP_ID, out_w, out_h)
        .with_buffers(vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(input_count),
            BufferDecl::storage(output, 1, BufferAccess::ReadWrite, DataType::U32)
                .with_count(output_count),
        ])
        .build(|_shape, _idx, py, px| {
            let [p00_idx, p10_idx, p01_idx, p11_idx] =
                crate::builder::stencil::downsample_2x_source_indices(
                    &Expr::var("oy"),
                    &Expr::var("ox"),
                    width,
                );
            vec![
                Node::let_bind("ox", px),
                Node::let_bind("oy", py),
                // Load 4 source pixels.
                Node::let_bind("p00", Expr::load(input, p00_idx)),
                Node::let_bind("p10", Expr::load(input, p10_idx)),
                Node::let_bind("p01", Expr::load(input, p01_idx)),
                Node::let_bind("p11", Expr::load(input, p11_idx)),
                // Average each channel: (c0+c1+c2+c3+2) >> 2
                Node::let_bind(
                    "r",
                    crate::builder::stencil::avg4_channel(
                        crate::builder::stencil::unpack_channel("p00", 0),
                        crate::builder::stencil::unpack_channel("p10", 0),
                        crate::builder::stencil::unpack_channel("p01", 0),
                        crate::builder::stencil::unpack_channel("p11", 0),
                    ),
                ),
                Node::let_bind(
                    "g",
                    crate::builder::stencil::avg4_channel(
                        crate::builder::stencil::unpack_channel("p00", 8),
                        crate::builder::stencil::unpack_channel("p10", 8),
                        crate::builder::stencil::unpack_channel("p01", 8),
                        crate::builder::stencil::unpack_channel("p11", 8),
                    ),
                ),
                Node::let_bind(
                    "b",
                    crate::builder::stencil::avg4_channel(
                        crate::builder::stencil::unpack_channel("p00", 16),
                        crate::builder::stencil::unpack_channel("p10", 16),
                        crate::builder::stencil::unpack_channel("p01", 16),
                        crate::builder::stencil::unpack_channel("p11", 16),
                    ),
                ),
                Node::let_bind(
                    "a",
                    crate::builder::stencil::avg4_channel(
                        crate::builder::stencil::unpack_channel("p00", 24),
                        crate::builder::stencil::unpack_channel("p10", 24),
                        crate::builder::stencil::unpack_channel("p01", 24),
                        crate::builder::stencil::unpack_channel("p11", 24),
                    ),
                ),
                // Pack RGBA.
                Node::let_bind(
                    "packed",
                    crate::builder::stencil::pack_rgba_named("r", "g", "b", "a"),
                ),
                // Write output.
                Node::let_bind(
                    "oidx",
                    crate::builder::stencil::flat_index(Expr::var("oy"), out_w, Expr::var("ox")),
                ),
                Node::store(output, Expr::var("oidx"), Expr::var("packed")),
            ]
        })
}

const EXPECTED_DOWNSAMPLE_2X_OUTPUT_BYTES: [u8; 16] = [0xFF; 16];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID,
        || downsample_2x("input", "output", 4, 4),
        Some(|| {
            // 4×4 all-white → 2×2 all-white
            let input = vec![0xFFFF_FFFFu32; 16];
            vec![vec![
                crate::visual::u32_word_bytes::u32_words_to_le_bytes(&input),
                vec![0u8; 16],
            ]]
        }),
        Some(|| {
            vec![vec![EXPECTED_DOWNSAMPLE_2X_OUTPUT_BYTES.to_vec()]]
        }),
    )
    .with_category("visual")
}
