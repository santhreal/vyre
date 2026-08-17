//! Backward for `parallel_residual_block`:
//!
//! Forward: `out = x + attn_out + mlp_out`
//! Backward: `grad_x = grad_attn = grad_mlp = grad_out` (addition broadcast).

use crate::builder::elementwise::ElementwiseComposer;
use vyre_foundation::ir::{BufferAccess, DataType, Expr, Program};

const OP_ID: &str = "vyre-libs::nn::residual_block_backward";

/// Backward for parallel_residual_block (F32).
///
/// Since forward is just addition, all three input gradients equal grad_out.
/// This op copies grad_out → grad_x, grad_attn, grad_mlp.
#[must_use]
pub fn residual_block_backward(
    grad_out: &str,
    grad_x: &str,
    grad_attn: &str,
    grad_mlp: &str,
    n: u32,
) -> Program {
    ElementwiseComposer::new(OP_ID, n)
        .add_input(grad_out, DataType::F32, n)
        .add_output(grad_x, DataType::F32, n)
        .add_output_storage(grad_attn, BufferAccess::ReadWrite, DataType::F32, n)
        .add_output_storage(grad_mlp, BufferAccess::ReadWrite, DataType::F32, n)
        .build_pointwise_multi(&[grad_x, grad_attn, grad_mlp], |i| {
            let dy = Expr::load(grad_out, i);
            vec![dy.clone(), dy.clone(), dy]
        })
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || residual_block_backward("grad_out", "grad_x", "grad_attn", "grad_mlp", 4),
        Some(|| {
            let to_f32 = |w: &[f32]| vyre_primitives::wire::pack_f32_slice(w);
            vec![vec![
                to_f32(&[1.0, 2.0, 3.0, 4.0]),
                vec![0u8; 4 * 4], // grad_attn
                vec![0u8; 4 * 4], // grad_mlp
            ]]
        }),
        Some(|| {
            // All three live-outs = copy of grad_out.
            let to_f32 = |w: &[f32]| vyre_primitives::wire::pack_f32_slice(w);
            let expected = to_f32(&[1.0, 2.0, 3.0, 4.0]);
            vec![vec![expected.clone(), expected.clone(), expected]]
        }),
    )
    .with_category("nn")
}

#[cfg(test)]
mod tests {
    use super::residual_block_backward;
    use vyre_reference::value::Value;

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        vyre_primitives::wire::pack_f32_slice(values)
    }

    #[test]
    fn reference_outputs_all_residual_gradient_liveouts() {
        let program = residual_block_backward("grad_out", "grad_x", "grad_attn", "grad_mlp", 4);
        let expected = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(expected.clone()),
                Value::from(vec![0_u8; 16]),
                Value::from(vec![0_u8; 16]),
            ],
        )
        .expect("Fix: residual_block_backward must satisfy the one-output plus ReadWrite live-out IR contract.");

        assert_eq!(outputs.len(), 3);
        assert_eq!(outputs[0].to_bytes(), expected);
        assert_eq!(outputs[1].to_bytes(), expected);
        assert_eq!(outputs[2].to_bytes(), expected);
    }
}
