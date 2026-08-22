//! Element-wise square: `y = x * x`.
//!
//! Category-A composition.

use crate::builder::elementwise::{f32_elementwise_mul, F32MulRhs};
use vyre_foundation::ir::Program;

/// Build a Program that computes `output[i] = input[i] * input[i]`.
#[must_use]
pub fn square(input: &str, output: &str, n: u32) -> Program {
    f32_elementwise_mul(
        "vyre-libs::math::square",
        input,
        F32MulRhs::SameInput,
        output,
        n,
    )
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        "vyre-libs::math::square",
        || square("input", "output", 4),
        Some(|| {
            let to_bytes = vyre_primitives::wire::pack_f32_slice;
            vec![vec![
                to_bytes(&[2.0_f32, 3.0, 4.0, 5.0]), // input
            ]]
        }),
        Some(|| {
            // [4.0, 9.0, 16.0, 25.0]
            vec![vec![vec![
                0x00, 0x00, 0x80, 0x40, // 4.0
                0x00, 0x00, 0x10, 0x41, // 9.0
                0x00, 0x00, 0x80, 0x41, // 16.0
                0x00, 0x00, 0xc8, 0x41, // 25.0
            ]]]
        }),
    )
    .with_category("math")
}
