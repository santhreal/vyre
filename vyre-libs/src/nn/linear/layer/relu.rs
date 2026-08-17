//! Fused `linear_relu` constructor.

use vyre_foundation::composition::trap_program;
use vyre_foundation::ir::{DataType, Program};

use super::fused_activation::linear_fused_activation;
use crate::nn::activation::relu::relu_f32_expr;

const OP_ID: &str = "vyre-libs::nn::linear_relu";

/// Build a Program that computes `out[i] = max(0, sum_k x[k] * w[k, i] + b[i])`.
///
/// Fused variant of `linear` followed by ReLU.
///
/// # Errors
/// Returns `Err` when `in_dim == 0`.
pub fn linear_relu(
    x: &str,
    w: &str,
    b: &str,
    out: &str,
    in_dim: u32,
    out_dim: u32,
) -> Result<Program, String> {
    linear_fused_activation(
        "linear_relu",
        OP_ID,
        x,
        w,
        b,
        out,
        in_dim,
        out_dim,
        relu_f32_expr,
    )
}

const EXPECTED_LINEAR_RELU_OUTPUT_BYTES: [u8; 16] = [
    0x00, 0x00, 0x60, 0x42, 0x00, 0x00, 0x78, 0x42, 0x00, 0x00, 0x88, 0x42, 0x00, 0x00, 0x94, 0x42,
];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || {
            linear_relu("x", "w", "b", "out", 4, 4).unwrap_or_else(|error| {
                trap_program(
                    OP_ID,
                    Some(("out", DataType::F32)),
                    error,
                )
            })
        },
        Some(|| {
            let f32_bytes = vyre_primitives::wire::pack_f32_slice;
            let x = f32_bytes(&(0..4).map(|i| i as f32).collect::<Vec<_>>());
            let w = f32_bytes(&(0..16).map(|i| i as f32).collect::<Vec<_>>());
            let bias = f32_bytes(&[0.0, 0.0, 0.0, 0.0]);
            vec![vec![x, w, bias]]
        }),
        Some(|| {
            vec![vec![EXPECTED_LINEAR_RELU_OUTPUT_BYTES.to_vec()]]
        }),
    )
    .with_category("nn")
}
