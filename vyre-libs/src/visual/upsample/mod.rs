//! 2× nearest-neighbor upsample for the half-resolution blur path.
//!
//! Each input pixel maps to a 2×2 block in the output. This is intentionally
//! nearest-neighbor (no bilinear) because the input is already blurred  -
//! the blur itself provides the smoothing that bilinear would add.
//!
//! Category A composition  -  pure IR. No shared primitives.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

const OP_ID: &str = "vyre-libs::visual::upsample";

/// Build a Program that 2× upsamples `input` into `output`.
///
/// - `input`:  `[u32; (width/2) * (height/2)]`  -  source pixels (packed RGBA)
/// - `output`: `[u32; width * height]`  -  upsampled result
/// - `width`, `height`: the FULL output dimensions (must be even).
#[must_use]
pub fn upsample_2x(input: &str, output: &str, width: u32, height: u32) -> Program {
    let in_w = width / 2;
    let in_h = height / 2;
    let input_count = in_w.saturating_mul(in_h);
    let output_count = width.saturating_mul(height);

    crate::builder::stencil::Grid2DComposer::new(OP_ID, width, height)
        .with_buffers(vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(input_count),
            BufferDecl::storage(output, 1, BufferAccess::ReadWrite, DataType::U32)
                .with_count(output_count),
        ])
        .build(|_shape, _idx, py, px| {
            let in_sample_idx = crate::builder::stencil::upsample_2x_source_index(
                &Expr::var("oy"),
                &Expr::var("ox"),
                in_w,
            );
            vec![
                // Output coordinates.
                Node::let_bind("ox", px),
                Node::let_bind("oy", py),
                // Load input pixel.
                Node::let_bind("pixel", Expr::load(input, in_sample_idx)),
                // Write to output.
                Node::store(output, Expr::var("idx"), Expr::var("pixel")),
            ]
        })
}

const EXPECTED_UPSAMPLE_2X_OUTPUT_BYTES: [u8; 64] = [0xFF; 64];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || upsample_2x("input", "output", 4, 4),
        Some(|| {
            // 2×2 all-white → 4×4 all-white
            let input = vec![0xFFFF_FFFFu32; 4];
            vec![vec![
                crate::visual::u32_word_bytes::u32_words_to_le_bytes(&input),
                vec![0u8; 64],
            ]]
        }),
        Some(|| {
            vec![vec![EXPECTED_UPSAMPLE_2X_OUTPUT_BYTES.to_vec()]]
        }),
    )
    .with_category("visual")
}
