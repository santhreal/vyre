//! Layerwise LN scale: `y = layer_norm(x) * scale`.
//!
//! Category A  -  element-wise mul by per-dim learnable scale.

use crate::{f32_elementwise_mul, F32MulRhs};
use vyre_foundation::ir::Program;

const OP_ID: &str = "vyre-libs::nn::layerwise_ln_scale";

/// Build a Program: `output[i] = input[i] * scale[i]` (F32).
#[must_use]
pub fn layerwise_ln_scale(input: &str, scale: &str, output: &str, n: u32) -> Program {
    f32_elementwise_mul(OP_ID, input, F32MulRhs::Buffer(scale), output, n)
}

const EXPECTED_LAYERWISE_LN_SCALE_OUTPUT_BYTES: [u8; 16] = [
    0x00, 0x00, 0x00, 0x3F, 0x00, 0x00, 0x80, 0x40, 0x00, 0x00, 0x40, 0x40, 0xCD, 0xCC, 0xCC, 0x3E,
];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || layerwise_ln_scale("input", "scale", "output", 4),
        Some(|| {
            let to_f32 = |w: &[f32]| vyre_primitives::wire::pack_f32_slice(w);
            vec![vec![
                to_f32(&[1.0, 2.0, 3.0, 4.0]),  // input (post-LN)
                to_f32(&[0.5, 2.0, 1.0, 0.1]),  // scale
            ]]
        }),
        Some(|| vec![vec![EXPECTED_LAYERWISE_LN_SCALE_OUTPUT_BYTES.to_vec()]]),
    )
    .with_category("nn")
}
