//! Backward for `leaky_relu_sq`: derivative of `max(αx, x)²`.
//!
//! For x≥0: d/dx = 2x. For x<0: d/dx = 2·(0.5x)·0.5 = 0.5x.
//! Branchless: `grad = dy * max(0.5*x, 2*x)`.

use vyre_foundation::ir::{Expr, Program};

use super::unary_f32::unary_f32_backward_program;

const OP_ID: &str = "vyre-libs::nn::leaky_relu_sq_backward";

/// Backward for leaky_relu_sq (F32).
#[must_use]
pub fn leaky_relu_sq_backward(input: &str, grad_out: &str, grad_in: &str, n: u32) -> Program {
    unary_f32_backward_program(OP_ID, input, grad_out, grad_in, n, |x| {
        // Branchless: for x>=0 -> 2x > 0.5x, for x<0 -> 0.5x > 2x.
        Expr::max(
            Expr::mul(Expr::f32(0.5), x.clone()),
            Expr::mul(Expr::f32(2.0), x),
        )
    })
}

const EXPECTED_LEAKY_RELU_SQ_BACKWARD_OUTPUT_BYTES: [u8; 16] = [
    0x00, 0x00, 0x80, 0x40, 0x00, 0x00, 0x00, 0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40,
];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID,
        || leaky_relu_sq_backward("input", "grad_out", "grad_in", 4),
        Some(|| {
            let to_f32 = |w: &[f32]| vyre_primitives::wire::pack_f32_slice(w);
            vec![vec![
                to_f32(&[2.0, -4.0, 0.0, 1.0]),
                to_f32(&[1.0, 1.0, 1.0, 1.0]),
            ]]
        }),
        Some(|| {
            vec![vec![EXPECTED_LEAKY_RELU_SQ_BACKWARD_OUTPUT_BYTES.to_vec()]]
        }),
    )
    .with_category("nn")
}

#[cfg(test)]
mod tests {
    use super::super::unary_f32::eval_unary_f32_backward;
    use super::*;

    #[test]
    fn generated_leaky_relu_sq_backward_matches_scalar_reference() {
        let n = 512usize;
        let input = (0..n)
            .map(|i| ((i as i32 % 97) - 48) as f32 / 7.0)
            .collect::<Vec<_>>();
        let grad_out = (0..n)
            .map(|i| ((i as i32 % 31) - 15) as f32 / 5.0)
            .collect::<Vec<_>>();
        let program = leaky_relu_sq_backward("input", "grad_out", "grad_in", n as u32);
        let actual = eval_unary_f32_backward(
            &program,
            &input,
            &grad_out,
            "Fix: leaky_relu_sq_backward must execute in the reference interpreter.",
        );
        for (index, ((actual, x), dy)) in actual
            .iter()
            .copied()
            .zip(input.iter().copied())
            .zip(grad_out.iter().copied())
            .enumerate()
        {
            let expected = dy * f32::max(0.5 * x, 2.0 * x);
            assert!(
                (actual - expected).abs() <= 1.0e-5,
                "generated leaky_relu_sq_backward mismatch at {index}: {actual} != {expected}"
            );
        }
    }
}
