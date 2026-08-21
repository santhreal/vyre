//! SiLU (Sigmoid Linear Unit): `y = x * sigmoid(x) = x / (1 + exp(-x))`.
//!
//! Category A composition.

use crate::builder::elementwise::ElementwiseComposer;
use vyre_foundation::ir::{Expr, Program, UnOp};

use crate::nn::f32_stability::flush_tiny;

/// Shared SiLU expression with the same tiny-value stabilization used by
/// standalone and fused activation builders.
pub(crate) fn silu_expr(x: Expr) -> Expr {
    let sigmoid_x = Expr::div(
        Expr::f32(1.0),
        Expr::add(
            Expr::f32(1.0),
            Expr::UnOp {
                op: UnOp::Exp,
                operand: Box::new(Expr::UnOp {
                    op: UnOp::Negate,
                    operand: Box::new(x.clone()),
                }),
            },
        ),
    );
    flush_tiny(Expr::mul(x, sigmoid_x))
}

/// Build a Program that applies SiLU element-wise from `input` into
/// `output`. `n` is the element count of both buffers.
#[must_use]
pub fn silu(input: &str, output: &str, n: u32) -> Program {
    ElementwiseComposer::f32_unary("vyre-libs::nn::silu", input, output, n, silu_expr)
}

const EXPECTED_SILU_OUTPUT_BYTES: [u8; 16] = [
    0x00, 0x00, 0x00, 0x00, 0xA8, 0x26, 0x3B, 0x3F, 0xB0, 0xB2, 0x89, 0xBE, 0xEA, 0x7B, 0xE1, 0x3F,
];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        "vyre-libs::nn::silu",
        || silu("input", "output", 4),
        Some(|| {
            let to_bytes = vyre_primitives::wire::pack_f32_slice;
            vec![vec![
                to_bytes(&[0.0_f32, 1.0, -1.0, 2.0]), // input
            ]]
        }),
        Some(|| {
            vec![vec![EXPECTED_SILU_OUTPUT_BYTES.to_vec()]]
        }),
    )
    .with_category("nn")
    .with_tolerance(vyre_foundation::operation::TolerancePolicy::f32_ulp(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture_bytes::eval_f32;

    fn silu_ref(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    #[test]
    fn silu_nan_input_propagates_nan() {
        let input = [f32::NAN];
        let program = silu("input", "output", 1);
        let out = eval_f32("silu", &program, &[&input[..]], 1);
        assert!(out[0].is_nan(), "silu(NaN) must be NaN");
    }

    #[test]
    fn silu_inf_inputs() {
        let program = silu("input", "output", 2);
        // +Inf
        let out = eval_f32("silu", &program, &[&[f32::INFINITY, 0.0][..]], 2);
        assert_eq!(out[0], f32::INFINITY, "silu(+Inf) must be +Inf");

        // -Inf: sigmoid(-Inf)=0, -Inf*0 = NaN
        let out = eval_f32("silu", &program, &[&[f32::NEG_INFINITY, 0.0][..]], 2);
        assert!(
            out[0].is_nan(),
            "silu(-Inf) must be NaN (negative infinity times zero)"
        );
    }

    #[test]
    fn silu_negative_zero_vs_positive_zero() {
        let program = silu("input", "output", 2);
        let out = eval_f32("silu", &program, &[&[0.0f32, -0.0f32][..]], 2);
        assert_eq!(out[0].to_bits(), 0.0f32.to_bits());
        // silu(-0.0) = -0.0 * 0.5 = -0.0, but flush_tiny may flush it
        // The reference computes -0.0 / 2.0 = -0.0
        // Note: the reference interpreter computes -0.0 * 0.5 = -0.0, but
        // flush_tiny or later rounding may produce +0.0. We accept +0.0 as
        // long as it is not a non-zero value.
        assert!(out[1] == 0.0, "silu(-0.0) must be zero, got {}", out[1]);
    }

    #[test]
    fn silu_subnormal_input_is_flushed_to_zero() {
        let sub = f32::from_bits(1); // smallest positive subnormal
        let program = silu("input", "output", 1);
        let out = eval_f32("silu", &program, &[&[sub][..]], 1);
        assert_eq!(
            out[0].to_bits(),
            0.0f32.to_bits(),
            "silu must flush tiny subnormal to +0.0"
        );
    }

    #[test]
    fn silu_all_zeros() {
        let input = [0.0f32; 4];
        let program = silu("input", "output", 4);
        let out = eval_f32("silu", &program, &[&input[..]], 4);
        assert_eq!(out, vec![0.0; 4]);
    }

    #[test]
    fn silu_all_ones() {
        let input = [1.0f32; 4];
        let program = silu("input", "output", 4);
        let out = eval_f32("silu", &program, &[&input[..]], 4);
        let expected = silu_ref(1.0);
        for (i, &v) in out.iter().enumerate() {
            assert!(
                (v - expected).abs() <= 1.0e-6,
                "silu all-ones mismatch at {i}: {v}"
            );
        }
    }

    #[test]
    fn silu_all_max_f32() {
        let input = [f32::MAX; 4];
        let program = silu("input", "output", 4);
        let out = eval_f32("silu", &program, &[&input[..]], 4);
        for (i, &v) in out.iter().enumerate() {
            // sigmoid(MAX) ≈ 1.0, so silu(MAX) ≈ MAX (does not overflow because MAX*1.0 = MAX)
            assert_eq!(
                v,
                f32::MAX,
                "silu(f32::MAX) must be f32::MAX at {i}: got {v}"
            );
        }
    }

    #[test]
    fn silu_single_element() {
        let input = [2.5f32];
        let program = silu("input", "output", 1);
        let out = eval_f32("silu", &program, &[&input[..]], 1);
        let expected = silu_ref(2.5);
        assert!(
            (out[0] - expected).abs() <= 1.0e-6,
            "silu single element mismatch: {} != {}",
            out[0],
            expected
        );
    }

    #[test]
    fn silu_empty_tensor() {
        let program = silu("input", "output", 0);
        let out = eval_f32("silu", &program, &[&[] as &[f32]], 0);
        assert!(out.is_empty());
    }

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn silu_output_invariant_for_finite_inputs(x in -1e10f32..1e10f32) {
            let program = silu("input", "output", 1);
            let out = eval_f32("silu", &program, &[&[x][..]], 1)[0];
            if x.is_nan() {
                prop_assert!(out.is_nan());
            } else if x > 0.0 {
                // For very large x, sigmoid(x) rounds to 1.0, so out ≈ x.
                prop_assert!(out > 0.0 && out <= x, "silu(x) for x>0 must be in (0, x]");
            } else if x < 0.0 {
                // flush_tiny may turn subnormal products into 0.0, so we allow 0.0.
                prop_assert!(out >= x && out <= 0.0, "silu(x) for x<0 must be in [x, 0]");
            } else {
                prop_assert_eq!(out.to_bits(), 0.0f32.to_bits());
            }
        }
    }
}
