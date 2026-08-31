//! GELU (Gaussian Error Linear Unit): `y = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))`.
//!
//! Category A composition.

use vyre_foundation::ir::{Expr, Program, UnOp};

const GELU_SQRT_2_OVER_PI: f32 = 0.797_884_6;
const GELU_COEF: f32 = 0.044715;
const OP_ID: &str = "vyre-libs::nn::gelu";

fn gelu_expr(x: Expr) -> Expr {
    let x3 = Expr::mul(Expr::mul(x.clone(), x.clone()), x.clone());
    let inner = Expr::mul(
        Expr::f32(GELU_SQRT_2_OVER_PI),
        Expr::add(x.clone(), Expr::mul(Expr::f32(GELU_COEF), x3)),
    );
    let tanh_inner = Expr::UnOp {
        op: UnOp::Tanh,
        operand: Box::new(inner),
    };
    Expr::mul(
        Expr::f32(0.5),
        Expr::mul(x, Expr::add(Expr::f32(1.0), tanh_inner)),
    )
}

/// Build a Program that applies GELU element-wise from `input` into
/// `output`. `n` is the element count of both buffers.
#[must_use]
pub fn gelu(input: &str, output: &str, n: u32) -> Program {
    super::unary::f32_unary_activation_program(OP_ID, input, output, n, gelu_expr)
}

const EXPECTED_GELU_OUTPUT_BYTES: [u8; 16] = [
    0x00, 0x00, 0x00, 0x00, 0x5C, 0x58, 0x57, 0x3F, 0x90, 0x9E, 0x22, 0xBE, 0x42, 0x30, 0xFA, 0x3F,
];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID,
        || gelu("input", "output", 4),
        Some(|| {
            let to_bytes = vyre_primitives::wire::pack_f32_slice;
            vec![vec![
                to_bytes(&[0.0_f32, 1.0, -1.0, 2.0]), // input
            ]]
        }),
        Some(|| {
            vec![vec![EXPECTED_GELU_OUTPUT_BYTES.to_vec()]]
        }),
    )
    .with_category("nn")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture_bytes::eval_f32;

    fn gelu_ref(x: f32) -> f32 {
        let x3 = x * x * x;
        let inner = GELU_SQRT_2_OVER_PI * (x + GELU_COEF * x3);
        0.5 * x * (1.0 + inner.tanh())
    }

    #[test]
    fn gelu_all_zeros() {
        let input = [0.0f32; 4];
        let program = gelu("input", "output", 4);
        let out = eval_f32("gelu", &program, &[&input[..]], 4);
        assert_eq!(out, vec![0.0; 4]);
    }

    #[test]
    fn gelu_positive_values() {
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let program = gelu("input", "output", 4);
        let out = eval_f32("gelu", &program, &[&input[..]], 4);
        for (i, (&v, expected)) in out
            .iter()
            .zip(input.iter().copied().map(gelu_ref))
            .enumerate()
        {
            assert!(
                (v - expected).abs() <= 1.0e-5,
                "gelu mismatch at {i}: {v} != {expected}"
            );
        }
    }

    #[test]
    fn gelu_negative_values() {
        let input = [-1.0f32, -2.0, -0.5, -3.0];
        let program = gelu("input", "output", 4);
        let out = eval_f32("gelu", &program, &[&input[..]], 4);
        for (i, (&v, expected)) in out
            .iter()
            .zip(input.iter().copied().map(gelu_ref))
            .enumerate()
        {
            assert!(
                (v - expected).abs() <= 1.0e-5,
                "gelu mismatch at {i}: {v} != {expected}"
            );
        }
    }

    #[test]
    fn gelu_empty_tensor() {
        let program = gelu("input", "output", 0);
        let out = eval_f32("gelu", &program, &[&[] as &[f32]], 0);
        assert!(out.is_empty());
    }

    #[test]
    fn gelu_nan_input_propagates_nan() {
        let input = [f32::NAN];
        let program = gelu("input", "output", 1);
        let out = eval_f32("gelu", &program, &[&input[..]], 1);
        assert!(out[0].is_nan(), "gelu(NaN) must be NaN");
    }

    #[test]
    fn generated_gelu_matches_scalar_reference() {
        let input = (0..2048u32)
            .map(|i| ((i as f32) * 0.017).sin() * 6.0 - 3.0)
            .collect::<Vec<_>>();
        let program = gelu("input", "output", input.len() as u32);
        let out = eval_f32("gelu", &program, &[&input[..]], input.len());
        for (index, (actual, expected)) in out
            .iter()
            .copied()
            .zip(input.iter().copied().map(gelu_ref))
            .enumerate()
        {
            assert!(
                (actual - expected).abs() <= 1.0e-5,
                "generated gelu mismatch at {index}: {actual} != {expected}"
            );
        }
    }
}
