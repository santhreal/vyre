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

const EXPECTED_UNPACK_NIBBLE_U32_F32_OUTPUT_BYTES: [u8; 64] = [
    0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64, 0, 0, 160, 64, 0, 0, 192,
    64, 0, 0, 224, 64, 0, 0, 0, 65, 0, 0, 16, 65, 0, 0, 32, 65, 0, 0, 48, 65, 0, 0, 64, 65, 0, 0,
    80, 65, 0, 0, 96, 65, 0, 0, 112, 65,
];

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
            vec![vec![EXPECTED_UNPACK_NIBBLE_U32_F32_OUTPUT_BYTES.to_vec()]]
        }),
    )
}
