//! LeakyReLU²: `y = leaky_relu(x, α=0.5)² = max(α·x, x)²`.
//!
//! Category A composition  -  element-wise `leaky_relu` (alpha=0.5)
//! followed by squaring (`mul self`). Used in the Parameter Golf
//! recipe as the MLP activation: hidden = leaky_relu_sq(linear(x)).

use vyre_foundation::ir::{Expr, Program};

const OP_ID: &str = "vyre-libs::nn::leaky_relu_sq";

fn leaky_relu_sq_expr(x: Expr) -> Expr {
    let half_x = Expr::mul(Expr::f32(0.5), x.clone());
    let leaky = Expr::max(half_x, x);
    Expr::mul(leaky.clone(), leaky)
}

/// Build a Program that applies `leaky_relu(x, 0.5)²` element-wise.
///
/// `input[n]` (F32, ReadOnly) → `output[n]` (F32).
///
/// For each element `x`:
///   `leaky = max(0.5 * x, x)`
///   `out   = leaky * leaky`
#[must_use]
pub fn leaky_relu_sq(input: &str, output: &str, n: u32) -> Program {
    super::unary::f32_unary_activation_program(OP_ID, input, output, n, leaky_relu_sq_expr)
}

const EXPECTED_LEAKY_RELU_SQ_OUTPUT_BYTES: [u8; 16] = [
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x40, 0x00, 0x00, 0x80, 0x40, 0x00, 0x00, 0x80, 0x3F,
];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID,
        || leaky_relu_sq("input", "output", 4),
        Some(|| {
            let to_bytes = vyre_primitives::wire::pack_f32_slice;
            vec![vec![
                to_bytes(&[0.0_f32, 2.0, -4.0, 1.0]),
            ]]
        }),
        Some(|| {
            vec![vec![EXPECTED_LEAKY_RELU_SQ_OUTPUT_BYTES.to_vec()]]
        }),
    )
    .with_category("nn")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture_bytes::eval_f32;

    fn leaky_relu_sq_ref(x: f32) -> f32 {
        let leaky = (0.5 * x).max(x);
        leaky * leaky
    }

    #[test]
    fn leaky_relu_sq_nan_input_propagates_nan() {
        let input = [f32::NAN];
        let program = leaky_relu_sq("input", "output", 1);
        let out = eval_f32("leaky_relu_sq", &program, &[&input[..]], 1);
        assert!(out[0].is_nan(), "leaky_relu_sq(NaN) must be NaN");
    }

    #[test]
    fn leaky_relu_sq_inf_inputs() {
        let program = leaky_relu_sq("input", "output", 2);
        // +Inf: max(0.5*Inf, Inf) = Inf, Inf*Inf = Inf
        let out = eval_f32("leaky_relu_sq", &program, &[&[f32::INFINITY, 0.0][..]], 2);
        assert_eq!(out[0], f32::INFINITY, "leaky_relu_sq(+Inf) must be +Inf");

        // -Inf: max(-0.5*Inf, -Inf) = max(-Inf, -Inf) = -Inf, (-Inf)*(-Inf) = +Inf
        let out = eval_f32(
            "leaky_relu_sq",
            &program,
            &[&[f32::NEG_INFINITY, 0.0][..]],
            2,
        );
        assert_eq!(
            out[0],
            f32::INFINITY,
            "leaky_relu_sq(-Inf) must be +Inf (square of negative infinity)"
        );
    }

    #[test]
    fn leaky_relu_sq_negative_zero_vs_positive_zero() {
        let program = leaky_relu_sq("input", "output", 2);
        let out = eval_f32("leaky_relu_sq", &program, &[&[0.0f32, -0.0f32][..]], 2);
        assert_eq!(out[0].to_bits(), 0.0f32.to_bits());
        assert_eq!(
            out[1].to_bits(),
            0.0f32.to_bits(),
            "leaky_relu_sq(-0.0) must be +0.0"
        );
    }

    #[test]
    fn leaky_relu_sq_subnormal_input() {
        let sub = f32::from_bits(1);
        let program = leaky_relu_sq("input", "output", 1);
        let out = eval_f32("leaky_relu_sq", &program, &[&[sub][..]], 1);
        let expected = leaky_relu_sq_ref(sub);
        assert!(
            (out[0] - expected).abs() <= 1.0e-6,
            "leaky_relu_sq(subnormal) mismatch"
        );
    }

    #[test]
    fn generated_leaky_relu_sq_matches_scalar_reference() {
        let input = (0..2048u32)
            .map(|i| ((i as f32) * 0.031).cos() * 8.0 - 4.0)
            .collect::<Vec<_>>();
        let program = leaky_relu_sq("input", "output", input.len() as u32);
        let out = eval_f32("leaky_relu_sq", &program, &[&input[..]], input.len());
        for (index, (actual, expected)) in out
            .iter()
            .copied()
            .zip(input.iter().copied().map(leaky_relu_sq_ref))
            .enumerate()
        {
            assert!(
                (actual - expected).abs() <= 1.0e-5,
                "generated leaky_relu_sq mismatch at {index}: {actual} != {expected}"
            );
        }
    }

    #[test]
    fn leaky_relu_sq_all_zeros() {
        let input = [0.0f32; 4];
        let program = leaky_relu_sq("input", "output", 4);
        let out = eval_f32("leaky_relu_sq", &program, &[&input[..]], 4);
        assert_eq!(out, vec![0.0; 4]);
    }

    #[test]
    fn leaky_relu_sq_all_ones() {
        let input = [1.0f32; 4];
        let program = leaky_relu_sq("input", "output", 4);
        let out = eval_f32("leaky_relu_sq", &program, &[&input[..]], 4);
        assert_eq!(out, vec![1.0; 4]);
    }

    #[test]
    fn leaky_relu_sq_all_max_f32() {
        let input = [f32::MAX; 4];
        let program = leaky_relu_sq("input", "output", 4);
        let out = eval_f32("leaky_relu_sq", &program, &[&input[..]], 4);
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(
                v,
                f32::INFINITY,
                "leaky_relu_sq(f32::MAX) must overflow to +Inf at {i}: got {v}"
            );
        }
    }

    #[test]
    fn leaky_relu_sq_single_element() {
        let input = [-3.0f32];
        let program = leaky_relu_sq("input", "output", 1);
        let out = eval_f32("leaky_relu_sq", &program, &[&input[..]], 1);
        let expected = leaky_relu_sq_ref(-3.0);
        assert!(
            (out[0] - expected).abs() <= 1.0e-5,
            "leaky_relu_sq single element mismatch"
        );
    }

    #[test]
    fn leaky_relu_sq_empty_tensor() {
        let program = leaky_relu_sq("input", "output", 0);
        let out = eval_f32("leaky_relu_sq", &program, &[&[] as &[f32]], 0);
        assert!(out.is_empty());
    }

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn leaky_relu_sq_output_is_nonnegative(x in prop::num::f32::NORMAL) {
            let program = leaky_relu_sq("input", "output", 1);
            let out = eval_f32("leaky_relu_sq", &program, &[&[x][..]], 1)[0];
            if x.is_nan() {
                prop_assert!(out.is_nan());
            } else {
                prop_assert!(out >= 0.0 || out.is_nan(), "leaky_relu_sq(x) must be >= 0 or NaN, got {out}");
            }
        }
    }
}
