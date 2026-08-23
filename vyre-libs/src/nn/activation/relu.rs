//! ReLU: `y = max(0, x)`.
//!
//! Category A composition  -  one primitive per invocation. Element-wise
//! so the optimizer can trivially fuse into any upstream operation.

use crate::builder::elementwise::ElementwiseComposer;
use vyre_foundation::ir::{Expr, Program};

/// Shared unsigned ReLU expression used by the standalone activation builder.
#[must_use]
pub(crate) fn relu_u32_expr(x: Expr) -> Expr {
    Expr::max(Expr::u32(0), x)
}

/// Shared floating-point ReLU expression used by fused activation builders.
#[must_use]
pub(crate) fn relu_f32_expr(x: Expr) -> Expr {
    Expr::max(Expr::f32(0.0), x)
}

/// Build a Program that applies ReLU element-wise from `input` into
/// `output`. `n` is the element count of both buffers. u32 semantics:
/// values are unsigned so "max(0, x)" is the identity; this module
/// provides the structural Category-A shape and a future i32/f32
/// overload replaces the primitive.
#[must_use]
pub fn relu(input: &str, output: &str, n: u32) -> Program {
    ElementwiseComposer::u32_unary("vyre-libs::nn::relu", input, output, n, relu_u32_expr)
}

const EXPECTED_RELU_OUTPUT_BYTES: [u8; 16] = [
    0x00, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x0A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        "vyre-libs::nn::relu",
        || relu("input", "output", 4),
        Some(|| vec![vec![
            vyre_primitives::wire::pack_u32_slice(&[0u32, 5, 10, 0]),
        ]]),
        Some(|| vec![vec![
            EXPECTED_RELU_OUTPUT_BYTES.to_vec(),
        ]]),
    )
    .with_category("nn")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture_bytes::eval_bytes;
    use crate::fixture_bytes::eval_f32;
    use crate::fixture_bytes::u32_bytes;

    #[test]
    fn relu_empty_tensor_produces_no_panic() {
        let program = relu("input", "output", 0);
        let out = eval_f32("relu", &program, &[&[] as &[f32]], 0);
        assert!(out.is_empty());
    }

    #[test]
    fn relu_single_element_identity() {
        let input = [42u32];
        let program = relu("input", "output", 1);
        let outputs = eval_bytes("relu", &program, vec![u32_bytes(&input), vec![0u8; 4]]);
        let out: Vec<u32> = vyre_primitives::wire::decode_u32_le_bytes_all(&outputs[0]);
        assert_eq!(out, vec![42]);
    }

    #[test]
    fn relu_all_zeros_identity() {
        let input = [0u32, 0, 0, 0];
        let program = relu("input", "output", 4);
        let outputs = eval_bytes("relu", &program, vec![u32_bytes(&input), vec![0u8; 16]]);
        let out: Vec<u32> = vyre_primitives::wire::decode_u32_le_bytes_all(&outputs[0]);
        assert_eq!(out, vec![0, 0, 0, 0]);
    }

    #[test]
    fn relu_all_max_u32_identity() {
        let input = [u32::MAX; 4];
        let program = relu("input", "output", 4);
        let outputs = eval_bytes("relu", &program, vec![u32_bytes(&input), vec![0u8; 16]]);
        let out: Vec<u32> = vyre_primitives::wire::decode_u32_le_bytes_all(&outputs[0]);
        assert_eq!(out, vec![u32::MAX; 4]);
    }
}
