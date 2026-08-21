//! Shared F32 unary backward kernel builder.

use crate::builder::elementwise::ElementwiseComposer;
use vyre_foundation::ir::{DataType, Expr, Program};

pub(super) fn unary_f32_backward_program<F>(
    op_id: &'static str,
    input: &str,
    grad_out: &str,
    grad_in: &str,
    n: u32,
    local_grad: F,
) -> Program
where
    F: FnOnce(Expr) -> Expr,
{
    ElementwiseComposer::new(op_id, n)
        .add_input(input, DataType::F32, n)
        .add_input(grad_out, DataType::F32, n)
        .add_output(grad_in, DataType::F32, n)
        .build_pointwise(grad_in, |i| {
            let x = Expr::load(input, i.clone());
            let dy = Expr::load(grad_out, i);
            Expr::mul(dy, local_grad(x))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_unary_backward_program_lengths_are_declared_exactly() {
        let mut cases = 0usize;
        for n in 0..=2048 {
            let program = unary_f32_backward_program(
                "vyre-libs::nn::test_unary_backward",
                "input",
                "grad_out",
                "grad_in",
                n,
                |x| x,
            );
            assert_eq!(program.buffers().len(), 3);
            let output = program
                .buffers()
                .iter()
                .find(|buffer| buffer.is_output())
                .expect("Fix: unary backward program must declare grad output.");
            assert_eq!(output.count(), n);
            cases += 1;
        }
        assert_eq!(cases, 2_049);
    }
}
#[cfg(test)]
pub(super) fn eval_unary_f32_backward(
    program: &Program,
    input: &[f32],
    grad_out: &[f32],
    error_msg: &'static str,
) -> Vec<f32> {
    let n = input.len();
    assert_eq!(n, grad_out.len());
    let outputs = crate::fixture_bytes::eval_bytes(
        "unary_f32",
        program,
        vec![
            vyre_primitives::wire::pack_f32_slice(input),
            vyre_primitives::wire::pack_f32_slice(grad_out),
            vec![0u8; n * core::mem::size_of::<f32>()],
        ],
    );
    vyre_primitives::wire::decode_f32_le_bytes_all(&outputs[0])
}
