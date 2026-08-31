//! Variance reduction: `y = variance(x)` using Welford's parallel pair-combination algorithm.
//!
//! Category-A composition with a workgroup-tiled Welford reduction.

use crate::builder::reduction::ReductionComposer;
use vyre_foundation::ir::Program;
#[cfg(test)]
use vyre_foundation::ir::{Expr, Node};

const OP_ID: &str = "vyre-libs::math::reduce_variance";
#[cfg(test)]
const REFERENCE_OP_ID: &str = "vyre-libs::math::reduce_variance_reference";
const EMPTY_REDUCTION_FIX: &str =
    "Fix: reduce_variance n=0 is invalid; pass at least one input element or route empty reductions to a caller-defined identity.";

/// Build a Program that computes the population variance of `input` into `output[0]`.
#[must_use]
pub fn reduce_variance(input: &str, output: &str, n: u32) -> Program {
    if n == 0 {
        return reduce_variance_invalid_program(input, output);
    }
    reduce_variance_tiled_program(input, output, n, false)
}

/// Fallible builder for variance reduction.
///
/// # Errors
///
/// Returns an actionable error for empty reductions.
pub fn try_reduce_variance(
    input: &str,
    output: &str,
    n: u32,
    bessel: bool,
) -> Result<Program, &'static str> {
    if n == 0 {
        return Err(EMPTY_REDUCTION_FIX);
    }
    Ok(reduce_variance_tiled_program(input, output, n, bessel))
}

fn reduce_variance_invalid_program(input: &str, output: &str) -> Program {
    super::invalid_f32_reduction_program(OP_ID, input, output, EMPTY_REDUCTION_FIX)
}

fn reduce_variance_tiled_program(input: &str, output: &str, n: u32, bessel: bool) -> Program {
    ReductionComposer::tiled_variance(OP_ID, input, output, n, bessel, 256)
}

#[cfg(test)]
fn reduce_variance_reference_program(input: &str, output: &str, n: u32, bessel: bool) -> Program {
    let body = vec![
        Node::let_bind("sum", Expr::f32(0.0)),
        Node::let_bind("sum_sq", Expr::f32(0.0)),
        Node::loop_for(
            "i",
            Expr::u32(0),
            Expr::u32(n),
            vec![
                Node::let_bind("x", Expr::load(input, Expr::var("i"))),
                Node::assign("sum", Expr::add(Expr::var("sum"), Expr::var("x"))),
                Node::assign(
                    "sum_sq",
                    Expr::add(
                        Expr::var("sum_sq"),
                        Expr::mul(Expr::var("x"), Expr::var("x")),
                    ),
                ),
            ],
        ),
        Node::let_bind("mean", Expr::div(Expr::var("sum"), Expr::f32(n as f32))),
        Node::let_bind(
            "variance",
            Expr::div(
                Expr::sub(
                    Expr::var("sum_sq"),
                    Expr::mul(Expr::var("mean"), Expr::var("sum")),
                ),
                Expr::f32(if bessel { (n - 1) as f32 } else { n as f32 }),
            ),
        ),
        Node::Store {
            buffer: output.into(),
            index: Expr::u32(0),
            value: Expr::var("variance"),
        },
    ];

    super::wrap_unary_f32_scalar_program(REFERENCE_OP_ID, input, output, n, body)
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        "vyre-libs::math::reduce_variance",
        || reduce_variance("input", "output", 256),
        Some(|| {
            let to_bytes = vyre_primitives::wire::pack_f32_slice;
            vec![vec![
                to_bytes(&[2.0_f32; 256]), // input
            ]]
        }),
        Some(|| {
            vec![vec![vec![
                0x00, 0x00, 0x00, 0x00, // variance of constant array = 0.0_f32
            ]]]
        }),
    )
    .with_category("math")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture_bytes::decode_f32_one as decode_one;
    use crate::fixture_bytes::eval_bytes;
    use crate::fixture_bytes::f32_bytes;
    use crate::fixture_bytes::try_eval_bytes;

    fn eval_variance_reduction(program: Program, input: &[f32]) -> f32 {
        let outputs = eval_bytes(
            "reduce_variance",
            &program,
            vec![f32_bytes(input), vec![0u8; core::mem::size_of::<f32>()]],
        );
        decode_one(&outputs[0])
    }

    fn sample_input(n: u32) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32) * 0.019).sin() * 4.0 + (i % 7) as f32)
            .collect()
    }

    #[test]
    fn tiled_reduce_variance_matches_scalar_reference_across_multiple_tiles() {
        let n = 777_u32;
        let input = sample_input(n);
        let actual = eval_variance_reduction(reduce_variance("input", "output", n), &input);
        let expected = eval_variance_reduction(
            reduce_variance_reference_program("input", "output", n, false),
            &input,
        );
        assert!(
            (actual - expected).abs() <= 1.0e-4,
            "reduce_variance mismatch: tiled={actual:?} reference={expected:?}"
        );
    }

    #[test]
    fn bessel_correction_changes_result_by_expected_ratio() {
        let n = 777_u32;
        let input = sample_input(n);
        let pop = eval_variance_reduction(
            try_reduce_variance("input", "output", n, false).unwrap(),
            &input,
        );
        let sample = eval_variance_reduction(
            try_reduce_variance("input", "output", n, true).unwrap(),
            &input,
        );
        let expected_ratio = n as f32 / (n - 1) as f32;
        let actual_ratio = sample / pop;
        assert!(
            (actual_ratio - expected_ratio).abs() <= 1.0e-4,
            "Bessel correction ratio mismatch: sample={sample:?} pop={pop:?} expected_ratio={expected_ratio:?} actual_ratio={actual_ratio:?}"
        );
    }

    #[test]
    fn reduce_variance_rejects_empty_reduction_without_panicking() {
        let program = reduce_variance("input", "output", 0);
        let err = try_eval_bytes(
            &program,
            vec![
                vec![0u8; core::mem::size_of::<f32>()],
                vec![0u8; core::mem::size_of::<f32>()],
            ],
        )
        .expect_err("empty reduction must trap instead of constructing a fake variance");
        assert!(
            err.to_string().contains(EMPTY_REDUCTION_FIX),
            "wrong error: {err}"
        );
        assert_eq!(
            try_reduce_variance("input", "output", 0, false),
            Err(EMPTY_REDUCTION_FIX)
        );
    }

    #[test]
    fn try_reduce_variance_returns_err_for_zero_count() {
        assert_eq!(
            try_reduce_variance("input", "output", 0, false),
            Err(EMPTY_REDUCTION_FIX)
        );
        assert_eq!(
            try_reduce_variance("input", "output", 0, true),
            Err(EMPTY_REDUCTION_FIX)
        );
    }
}
