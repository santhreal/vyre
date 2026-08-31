//! Int8 per-row quantization for token embeddings.
//!
//! Unpack: `x = packed * scale[row]` (F32 output).
//! Pack: mask to 8 bits (U32→U32).

use crate::builder::elementwise::{u32_elementwise_unary, ElementwiseComposer};
use vyre_foundation::ir::{BinOp, BufferAccess, DataType, Expr, Program};

const PACK_OP_ID: &str = "vyre-libs::quant::int8_pack";
const UNPACK_OP_ID: &str = "vyre-libs::quant::int8_unpack";

/// Unpack int8: `output[i] = packed[i] * scale[row]` (F32).
#[must_use]
pub fn int8_unpack(packed: &str, scales: &str, output: &str, n: u32, cols: u32) -> Program {
    let rows = n.div_ceil(cols);
    ElementwiseComposer::new(UNPACK_OP_ID, n)
        .add_input(packed, DataType::U32, n)
        .add_input_storage(scales, BufferAccess::ReadOnly, DataType::F32, rows)
        .add_output(output, DataType::F32, n)
        .build_pointwise(output, |i| {
            let row_idx = Expr::BinOp {
                op: BinOp::Div,
                left: Box::new(i.clone()),
                right: Box::new(Expr::u32(cols)),
            };
            Expr::mul(
                Expr::cast(DataType::F32, Expr::load(packed, i)),
                Expr::load(scales, row_idx),
            )
        })
}

/// Pack to int8: mask to 8 bits.
#[must_use]
pub fn int8_pack(input: &str, output: &str, n: u32) -> Program {
    u32_elementwise_unary(PACK_OP_ID, input, output, n, |value| {
        Expr::bitand(value, Expr::u32(0xFF))
    })
}

const EXPECTED_INT8_UNPACK_OUTPUT_BYTES: [u8; 16] = [
    0x00, 0x00, 0xA0, 0x40, 0x00, 0x00, 0x20, 0x41, 0x00, 0x00, 0x70, 0x42, 0x00, 0x00, 0xA0, 0x42,
];
const EXPECTED_INT8_PACK_OUTPUT_BYTES: [u8; 16] = [
    0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        UNPACK_OP_ID,
        || int8_unpack("packed", "scales", "output", 4, 2),
        Some(|| {
            let to_u32 = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            let to_f32 = |w: &[f32]| vyre_primitives::wire::pack_f32_slice(w);
            vec![vec![
                to_u32(&[10, 20, 30, 40]),
                to_f32(&[0.5, 2.0]),  // 2 rows
            ]]
        }),
        Some(|| vec![vec![EXPECTED_INT8_UNPACK_OUTPUT_BYTES.to_vec()]]),
    )
    .with_category("nn")
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        PACK_OP_ID,
        || int8_pack("input", "output", 4),
        Some(|| {
            let to_u32 = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![vec![
                to_u32(&[255, 256, 1, 0]),
            ]]
        }),
        Some(|| vec![vec![EXPECTED_INT8_PACK_OUTPUT_BYTES.to_vec()]]),
    )
    .with_category("nn")
}
