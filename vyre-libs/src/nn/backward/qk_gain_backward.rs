//! Backward for `qk_gain`: `grad_q = grad_out * gain[h]`.

use crate::builder::elementwise::ElementwiseComposer;
use vyre_foundation::ir::{BinOp, BufferAccess, DataType, Expr, Program};

const OP_ID: &str = "vyre-libs::nn::qk_gain_backward";

/// Backward for qk_gain (F32). Produces grad_q.
#[must_use]
pub fn qk_gain_backward(
    gain: &str,
    grad_out: &str,
    grad_q: &str,
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
) -> Program {
    let total = num_heads * seq_len * head_dim;
    let per_head = seq_len * head_dim;

    ElementwiseComposer::new(OP_ID, total)
        .add_input_storage(gain, BufferAccess::ReadOnly, DataType::F32, num_heads)
        .add_input(grad_out, DataType::F32, total)
        .add_output(grad_q, DataType::F32, total)
        .build_pointwise(grad_q, |i| {
            let head_idx = Expr::BinOp {
                op: BinOp::Div,
                left: Box::new(i.clone()),
                right: Box::new(Expr::u32(per_head)),
            };
            Expr::mul(Expr::load(grad_out, i), Expr::load(gain, head_idx))
        })
}

const EXPECTED_QK_GAIN_BACKWARD_OUTPUT_BYTES: [u8; 16] = [
    0x00, 0x00, 0xA8, 0x40, 0x00, 0x00, 0xA8, 0x40, 0x00, 0x00, 0x40, 0x40, 0x00, 0x00, 0x40, 0x40,
];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || qk_gain_backward("gain", "grad_out", "grad_q", 2, 1, 2),
        Some(|| {
            let to_f32 = |w: &[f32]| vyre_primitives::wire::pack_f32_slice(w);
            vec![vec![
                to_f32(&[5.25, 3.0]),
                to_f32(&[1.0, 1.0, 1.0, 1.0]),
            ]]
        }),
        Some(|| {
            vec![vec![EXPECTED_QK_GAIN_BACKWARD_OUTPUT_BYTES.to_vec()]]
        }),
    )
    .with_category("nn")
}
