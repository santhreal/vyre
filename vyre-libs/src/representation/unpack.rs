//! Bit-unpacking primitives for compressed representations.
//!
//! Category-A compositions over `UnOp::Unpack*` primitives.

use crate::builder::elementwise::ElementwiseComposer;
use vyre_foundation::ir::{DataType, Expr, Program};

/// Unpack 4-bit values from a u32 buffer into f32.
/// Input: `n/8` u32s (each holds 8 4-bit values), Output: `n` f32s.
#[must_use]
pub fn unpack_4bit_f32(input: &str, output: &str, n: u32) -> Program {
    ElementwiseComposer::new("vyre-libs::representation::unpack_4bit_f32", n)
        .add_input(input, DataType::U32, n / 8)
        .add_output(output, DataType::F32, n)
        .build_pointwise(output, |i| {
            let u32_idx = Expr::div(i.clone(), Expr::u32(8));
            let shift = Expr::mul(Expr::rem(i, Expr::u32(8)), Expr::u32(4));
            let val = Expr::bitand(Expr::shr(Expr::load(input, u32_idx), shift), Expr::u32(0xF));
            Expr::cast(DataType::F32, val)
        })
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        "vyre-libs::representation::unpack_4bit_f32",
        || unpack_4bit_f32("input", "output", 16),
        Some(|| {

            // Pack 16 4-bit values: 0..15 into 2 u32s (8 nibbles each)
            // u32[0] = 0x76543210, u32[1] = 0xFEDCBA98
            vec![vec![
                crate::fixture_bytes::u32_bytes(&[0x7654_3210, 0xFEDC_BA98]), // input: 2 packed u32s
            ]]
        }),
        Some(|| {
            // u32 → f32 as a value-preserving cast (not a bit-cast),
            // matching target-text `f32(u32_value)`. The packed input
            // [0x76543210, 0xFEDCBA98] unpacks into nibbles 0..15
            // in LSB-first order, each of which casts to its integer
            // value as f32.
            let values: Vec<f32> = (0u32..16).map(|v| v as f32).collect();
            let bytes = vyre_primitives::wire::pack_f32_slice(&values);
            vec![vec![bytes]]
        }),
    )
}
