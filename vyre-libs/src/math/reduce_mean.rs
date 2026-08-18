//! Arithmetic mean reduction: `y = sum(x) / n`.
//!
//! Category-A composition with a workgroup-tiled sum reduction.

use crate::builder::reduction::ReductionComposer;
use vyre_foundation::ir::Program;
#[cfg(test)]
use vyre_foundation::ir::{Expr, Node};

const OP_ID: &str = "vyre-libs::math::reduce_mean";
#[cfg(test)]
const REFERENCE_OP_ID: &str = "vyre-libs::math::reduce_mean_reference";
const EMPTY_REDUCTION_FIX: &str = "Fix: reduce_mean n=0 is invalid; pass at least one input element or route empty reductions to a caller-defined identity.";

/// Build a Program that computes the mean of `input` into `output[0]`.
#[must_use]
pub fn reduce_mean(input: &str, output: &str, n: u32) -> Program {
    if n == 0 {
        return reduce_mean_invalid_program(input, output);
    }
    reduce_mean_tiled_program(input, output, n)
}

/// Fallible builder for [`reduce_mean`].
///
/// # Errors
///
/// Returns an actionable error for empty reductions.
pub fn try_reduce_mean(input: &str, output: &str, n: u32) -> Result<Program, &'static str> {
    if n == 0 {
        return Err(EMPTY_REDUCTION_FIX);
    }
    Ok(reduce_mean_tiled_program(input, output, n))
}

fn reduce_mean_invalid_program(input: &str, output: &str) -> Program {
    super::invalid_f32_reduction_program(OP_ID, input, output, EMPTY_REDUCTION_FIX)
}

fn reduce_mean_tiled_program(input: &str, output: &str, n: u32) -> Program {
    ReductionComposer::tiled_mean(OP_ID, input, output, n, 256)
}

#[cfg(test)]
fn reduce_mean_reference_program(input: &str, output: &str, n: u32) -> Program {
    let body = vec![
        Node::let_bind("sum", Expr::f32(0.0)),
        Node::loop_for(
            "i",
            Expr::u32(0),
            Expr::u32(n),
            vec![Node::assign(
                "sum",
                Expr::add(Expr::var("sum"), Expr::load(input, Expr::var("i"))),
            )],
        ),
        Node::Store {
            buffer: output.into(),
            index: Expr::u32(0),
            value: Expr::div(Expr::var("sum"), Expr::f32(n as f32)),
        },
    ];

    super::wrap_unary_f32_scalar_program(REFERENCE_OP_ID, input, output, n, body)
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        "vyre-libs::math::reduce_mean",
        || reduce_mean("input", "output", 4),
        Some(|| {
            let to_bytes = vyre_primitives::wire::pack_f32_slice;
            vec![vec![
                to_bytes(&[1.0_f32, 2.0, 3.0, 4.0]), // input
            ]]
        }),
        Some(|| {
            vec![vec![vec![
                0x00, 0x00, 0x20, 0x40, // mean of [1,2,3,4] = 2.5_f32
            ]]]
        }),
    )
    .with_category("math")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture_bytes::decode_f32_one as decode_one;
    use crate::fixture_bytes::f32_bytes;
    use vyre_reference::value::Value;

    #[test]
    fn tiled_reduce_mean_matches_scalar_reference_across_multiple_tiles() {
        let n = 777_u32;
        let input: Vec<f32> = (0..n)
            .map(|i| ((i as f32) * 0.019).sin() * 4.0 + (i % 7) as f32)
            .collect();
        let eval_mean = |program: Program| -> f32 {
            let outputs = vyre_reference::reference_eval(
                &program,
                &[
                    Value::from(f32_bytes(&input)),
                    Value::from(vec![0u8; core::mem::size_of::<f32>()]),
                ],
            )
            .expect("Fix: reduce_mean program must execute in the reference interpreter.");
            decode_one(&outputs[0].to_bytes())
        };
        let actual = eval_mean(reduce_mean("input", "output", n));
        let expected = eval_mean(reduce_mean_reference_program("input", "output", n));
        assert!(
            (actual - expected).abs() <= 1.0e-5,
            "reduce_mean mismatch: tiled={actual:?} reference={expected:?}"
        );
    }

    #[test]
    fn reduce_mean_rejects_empty_reduction_without_panicking() {
        let program = reduce_mean("input", "output", 0);
        let err = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(vec![0u8; core::mem::size_of::<f32>()]),
                Value::from(vec![0u8; core::mem::size_of::<f32>()]),
            ],
        )
        .expect_err("empty reduction must trap instead of constructing a fake mean");
        assert!(
            err.to_string().contains(EMPTY_REDUCTION_FIX),
            "wrong error: {err}"
        );
        assert_eq!(
            try_reduce_mean("input", "output", 0),
            Err(EMPTY_REDUCTION_FIX)
        );
    }

    // ------------------------------------------------------------------
    // Adversarial fixtures exposing real gaps
    // ------------------------------------------------------------------

    #[test]
    fn reduce_mean_single_element() {
        let program = reduce_mean("input", "output", 1);
        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(f32_bytes(&[42.0_f32])),
                Value::from(vec![0u8; 4]),
            ],
        )
        .expect("Fix: reduce_mean n=1 must execute");
        let actual = decode_one(&outputs[0].to_bytes());
        assert!(
            (actual - 42.0).abs() <= 1.0e-5,
            "mean of [42] = 42, got {actual}"
        );
    }

    /// NaN must pollute the entire reduction.
    #[test]
    fn reduce_mean_nan_input_propagates() {
        let program = reduce_mean("input", "output", 4);
        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(f32_bytes(&[1.0_f32, f32::NAN, 3.0, 4.0])),
                Value::from(vec![0u8; 4]),
            ],
        )
        .expect("Fix: reduce_mean with NaN must execute");
        let actual = decode_one(&outputs[0].to_bytes());
        assert!(
            actual.is_nan(),
            "mean containing NaN must be NaN, got {actual}"
        );
    }

    /// Positive infinity must dominate the mean.
    #[test]
    fn reduce_mean_inf_input_propagates() {
        let program = reduce_mean("input", "output", 4);
        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(f32_bytes(&[1.0_f32, f32::INFINITY, 3.0, 4.0])),
                Value::from(vec![0u8; 4]),
            ],
        )
        .expect("Fix: reduce_mean with Inf must execute");
        let actual = decode_one(&outputs[0].to_bytes());
        assert!(
            actual.is_infinite() && actual.is_sign_positive(),
            "mean containing Inf must be Inf, got {actual}"
        );
    }

    /// Negative infinity must dominate the mean.
    #[test]
    fn reduce_mean_negative_inf_input_propagates() {
        let program = reduce_mean("input", "output", 4);
        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(f32_bytes(&[1.0_f32, f32::NEG_INFINITY, 3.0, 4.0])),
                Value::from(vec![0u8; 4]),
            ],
        )
        .expect("Fix: reduce_mean with -Inf must execute");
        let actual = decode_one(&outputs[0].to_bytes());
        assert!(
            actual.is_infinite() && actual.is_sign_negative(),
            "mean containing -Inf must be -Inf, got {actual}"
        );
    }
}
