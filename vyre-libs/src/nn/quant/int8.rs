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

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
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
        Some(|| {
            // row0: [10*0.5, 20*0.5]=[5,10], row1: [30*2, 40*2]=[60,80]
            let out = [5.0_f32, 10.0, 60.0, 80.0];
            let bytes = vyre_primitives::wire::pack_f32_slice(&out);
            vec![vec![bytes]]
        }),
    )
    .with_category("nn")
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        PACK_OP_ID,
        || int8_pack("input", "output", 4),
        Some(|| {
            let to_u32 = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![vec![
                to_u32(&[255, 256, 1, 0]),
            ]]
        }),
        Some(|| {
            let to_u32 = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![vec![to_u32(&[255, 0, 1, 0])]]
        }),
    )
    .with_category("nn")
}
