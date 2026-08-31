//! Autotuner gradient direction via natural_gradient.
//!
//! Closes the recursion thesis: natural_gradient ships to
//! user dialects (KFAC-trained NNs, Fisher-information-aware
//! optimizers) AND drives vyre's autotuner past the
//! `differentiable_autotune` baseline by using the Fisher-information
//! manifold's local geometry instead of plain gradient descent.
//!
//! # The self-use
//!
//! Vyre's autotuner (`differentiable_autotune` self-consumer)
//! computes a smoothed-argmax over kernel-config samples. The
//! gradient direction it follows is the plain Euclidean gradient
//! of latency w.r.t. config parameters. This converges slowly when
//! the latency surface has elongated valleys (typical: launch-cost
//! varies fast in workgroup-x, slow in y/z).
//!
//! Natural gradient preconditions the gradient by the inverse
//! Fisher information  -  the Riemannian gradient on the
//! parameter manifold. Empirically, KFAC-style block-diagonal
//! Fisher approximation gives 5-10× faster convergence than
//! plain gradient on the same configuration-tuning surfaces.
//!
//! # Algorithm
//!
//! ```text
//! 1. compute plain gradient g = ∂latency/∂config (already exists)
//! 2. compute Fisher block M = Var(∂log_latency/∂config)
//!    over recent autotune samples
//! 3. M_inv_sqrt = inverse square root of M (host-side
//!    Newton-Schulz iteration → vyre-libs::math::preconditioner)
//! 4. g_nat = natural_gradient_block_apply(M_inv_sqrt, g)
//!    → preconditioned step direction
//! 5. autotuner takes step in g_nat direction instead of g
//! ```
//!
//! This module owns the natural-gradient apply step. Callers provide
//! the Fisher block they want to use, whether estimated on the host,
//! read from telemetry, or produced by another registered primitive.

use crate::dispatch_buffers::{
    checked_square_cells, decode_u32_output_exact, ensure_input_slots, write_u32_slice_le_bytes,
    write_zero_bytes,
};
use crate::math::natural_gradient::natural_gradient_block_apply;
use vyre_megakernel::{
    execute_single_program, SemanticExecutionError, SemanticExecutionPolicy, SemanticExecutor,
};
#[cfg(test)]
use vyre_reference::composition_witness::{
    identity_matrix_witness_into, natural_gradient_autotune_step_witness_into,
    natural_gradient_block_apply_witness as reference_natural_gradient_block_apply,
    natural_gradient_block_apply_witness_into as reference_natural_gradient_block_apply_into,
};

/// Caller-owned dispatch scratch for fixed-point natural-gradient preconditioning.
#[derive(Debug, Default)]
pub struct NaturalGradientGpuScratch {
    inputs: Vec<Vec<u8>>,
}

/// Apply the inverse-Fisher preconditioner to a plain gradient,
/// yielding the natural gradient `g_nat = M_inv_sqrt · g`. The
/// autotuner then takes a step in the `g_nat` direction.
///
/// `m_inv_sqrt` is the inverse square root of the Fisher block
/// (n × n row-major); `grad` is the plain gradient (length n).
///
/// # Panics
///
/// Panics if `m_inv_sqrt.len() != n*n` or `grad.len() != n`.
#[must_use]
#[cfg(test)]
pub(crate) fn reference_precondition_autotune_gradient(
    m_inv_sqrt: &[f64],
    grad: &[f64],
    n: u32,
) -> Vec<f64> {
    use crate::telemetry::{bump, natural_gradient_autotuner_calls};
    bump(&natural_gradient_autotuner_calls);
    reference_natural_gradient_block_apply(m_inv_sqrt, grad, n)
}

/// Apply the inverse-Fisher preconditioner into caller-owned output.
#[cfg(test)]
pub(crate) fn reference_precondition_autotune_gradient_into(
    m_inv_sqrt: &[f64],
    grad: &[f64],
    n: u32,
    out: &mut Vec<f64>,
) {
    use crate::telemetry::{bump, natural_gradient_autotuner_calls};
    bump(&natural_gradient_autotuner_calls);
    reference_natural_gradient_block_apply_into(m_inv_sqrt, grad, n, out);
}

/// Primitive-native fixed-point natural-gradient preconditioning.
///
/// `m_inv_sqrt_fixed` is an `n x n` row-major 16.16 matrix and
/// `grad_fixed` is a length-`n` 16.16 vector. The dispatcher runs
/// [`natural_gradient_block_apply`] and returns `M_inv_sqrt * grad` in the
/// same fixed-point representation.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when shapes are invalid or backend readback is
/// malformed.
pub fn precondition_autotune_gradient_fixed_via(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    m_inv_sqrt_fixed: &[u32],
    grad_fixed: &[u32],
    n: u32,
) -> Result<Vec<u32>, SemanticExecutionError> {
    let mut out = Vec::new();
    precondition_autotune_gradient_fixed_via_into(
        dispatcher,
        policy,
        m_inv_sqrt_fixed,
        grad_fixed,
        n,
        &mut out,
    )?;
    Ok(out)
}

/// Primitive-native fixed-point natural-gradient preconditioning into
/// caller-owned output storage.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when validation or backend execution fails.
pub fn precondition_autotune_gradient_fixed_via_into(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    m_inv_sqrt_fixed: &[u32],
    grad_fixed: &[u32],
    n: u32,
    out: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    let mut scratch = NaturalGradientGpuScratch::default();
    precondition_autotune_gradient_fixed_via_with_scratch_into(
        dispatcher,
        policy,
        m_inv_sqrt_fixed,
        grad_fixed,
        n,
        &mut scratch,
        out,
    )
}

/// Primitive-native fixed-point natural-gradient preconditioning using caller-owned dispatch
/// scratch.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when validation or backend execution fails.
pub fn precondition_autotune_gradient_fixed_via_with_scratch_into(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    m_inv_sqrt_fixed: &[u32],
    grad_fixed: &[u32],
    n: u32,
    scratch: &mut NaturalGradientGpuScratch,
    out: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    use crate::telemetry::{bump, natural_gradient_autotuner_calls};
    bump(&natural_gradient_autotuner_calls);

    let matrix_cells = checked_square_cells(n, "precondition_autotune_gradient_fixed_via")?;
    n.checked_mul(n).ok_or_else(|| {
    SemanticExecutionError::InvalidRequest(format!(
        "Fix: precondition_autotune_gradient_fixed_via n*n exceeds the primitive u32 buffer-count limit for n={n}."
    ))
})?;
    let n_us = n as usize;
    if m_inv_sqrt_fixed.len() != matrix_cells {
        return Err(SemanticExecutionError::InvalidRequest(format!(
        "Fix: precondition_autotune_gradient_fixed_via requires m_inv_sqrt_fixed.len() == n*n, got len={}, n={n}, n*n={matrix_cells}.",
        m_inv_sqrt_fixed.len()
    )));
    }
    if grad_fixed.len() != n_us {
        return Err(SemanticExecutionError::InvalidRequest(format!(
        "Fix: precondition_autotune_gradient_fixed_via requires grad_fixed.len() == n, got len={}, n={n}.",
        grad_fixed.len()
    )));
    }

    let program = natural_gradient_block_apply("m_inv_sqrt", "grad", "grad_nat", n);
    let out_bytes = n_us
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| {
            SemanticExecutionError::InvalidRequest(format!(
                "Fix: precondition_autotune_gradient_fixed_via n={n} overflows output byte count."
            ))
        })?;
    ensure_input_slots(&mut scratch.inputs, 3);
    write_u32_slice_le_bytes(&mut scratch.inputs[0], m_inv_sqrt_fixed);
    write_u32_slice_le_bytes(&mut scratch.inputs[1], grad_fixed);
    write_zero_bytes(&mut scratch.inputs[2], out_bytes);
    let outputs = execute_single_program(
        dispatcher,
        crate::dispatch_buffers::HOST_WRAPPER_NODE,
        program,
        &scratch.inputs,
        policy,
    )
    .map(|output| output.outputs)?;
    if outputs.is_empty() {
        return Err(SemanticExecutionError::Backend(format!(
        "Fix: precondition_autotune_gradient_fixed_via expected at least one output buffer, got {}.",
        outputs.len()
    )));
    }
    decode_u32_output_exact(
        &outputs[0],
        n_us,
        "precondition_autotune_gradient_fixed_via",
        out,
    )
}

/// Compute the autotuner step from a plain gradient and learning
/// rate, with Fisher preconditioning. Returns the parameter delta:
/// `delta = -lr · g_nat`.
#[must_use]
#[cfg(test)]
pub(crate) fn autotune_step(
    m_inv_sqrt: &[f64],
    grad: &[f64],
    n: u32,
    learning_rate: f64,
) -> Vec<f64> {
    let mut out = Vec::new();
    autotune_step_into(m_inv_sqrt, grad, n, learning_rate, &mut out);
    out
}

/// Compute the autotuner step into caller-owned output.
#[cfg(test)]
pub(crate) fn autotune_step_into(
    m_inv_sqrt: &[f64],
    grad: &[f64],
    n: u32,
    learning_rate: f64,
    out: &mut Vec<f64>,
) {
    use crate::telemetry::{bump, natural_gradient_autotuner_calls};
    bump(&natural_gradient_autotuner_calls);
    natural_gradient_autotune_step_witness_into(m_inv_sqrt, grad, n, learning_rate, out);
}

/// Convenience: identity Fisher block. When the autotuner has no
/// curvature information yet (cold start), pass this so the natural
/// gradient reduces to the plain gradient.
#[must_use]
#[cfg(test)]
pub(crate) fn identity_fisher_block(n: u32) -> Vec<f64> {
    let mut out = Vec::new();
    identity_fisher_block_into(n, &mut out);
    out
}

/// Write an identity Fisher block into caller-owned storage.
#[cfg(test)]
pub(crate) fn identity_fisher_block_into(n: u32, out: &mut Vec<f64>) {
    identity_matrix_witness_into(n, out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch_buffers::u32_slice_to_le_bytes;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9 * (1.0 + a.abs() + b.abs())
    }

    struct NaturalGradientDispatcher;

    impl SemanticExecutor for NaturalGradientDispatcher {
        fn execute(
            &self,
            request: &vyre_megakernel::SemanticExecutionRequest<'_>,
        ) -> Result<vyre_megakernel::SemanticExecutionOutput, SemanticExecutionError> {
            let inputs = crate::test_parity_oracles::canonical_inputs(request)?;
            let compute_ordered = || -> Result<Vec<Vec<u8>>, SemanticExecutionError> {
                assert_eq!(inputs.len(), 3);
                let matrix = crate::dispatch_buffers::read_u32s(&inputs[0]);
                let grad = crate::dispatch_buffers::read_u32s(&inputs[1]);
                assert_eq!(inputs[2].len(), grad.len() * std::mem::size_of::<u32>());
                let n = grad.len();
                assert_eq!(matrix.len(), n * n);
                let mut out = vec![0u32; n];
                for i in 0..n {
                    let mut acc = 0u64;
                    for j in 0..n {
                        acc =
                            acc.wrapping_add(((matrix[i * n + j] as u64) * (grad[j] as u64)) >> 16);
                    }
                    out[i] = acc as u32;
                }
                Ok(vec![u32_slice_to_le_bytes(&out)])
            };
            crate::test_parity_oracles::semantic_output_padded(request, compute_ordered()?)
        }
    }

    #[test]
    fn identity_fisher_recovers_plain_gradient() {
        let id = identity_fisher_block(3);
        let grad = vec![1.0, -2.0, 0.5];
        let g_nat = reference_precondition_autotune_gradient(&id, &grad, 3);
        for (a, b) in grad.iter().zip(g_nat.iter()) {
            assert!(approx_eq(*a, *b));
        }
    }

    #[test]
    fn autotune_step_negates_gradient() {
        let id = identity_fisher_block(2);
        let grad = vec![1.0, 2.0];
        let step = autotune_step(&id, &grad, 2, 0.1);
        // step = -0.1 * grad.
        assert!(approx_eq(step[0], -0.1));
        assert!(approx_eq(step[1], -0.2));
    }

    #[test]
    fn autotune_step_zero_lr_no_motion() {
        let id = identity_fisher_block(3);
        let grad = vec![1.0, 2.0, 3.0];
        let step = autotune_step(&id, &grad, 3, 0.0);
        for v in step {
            assert!(approx_eq(v, 0.0));
        }
    }

    #[test]
    fn autotune_step_into_reuses_output() {
        let id = identity_fisher_block(2);
        let grad = vec![1.0, 2.0];
        let mut step = Vec::with_capacity(8);
        let ptr = step.as_ptr();
        autotune_step_into(&id, &grad, 2, 0.1, &mut step);
        assert!(approx_eq(step[0], -0.1));
        assert!(approx_eq(step[1], -0.2));
        assert_eq!(step.as_ptr(), ptr);
    }

    #[test]
    fn diagonal_fisher_scales_per_axis() {
        // Anisotropic: x scaled by 1.0, y scaled by 4.0.
        // M_inv_sqrt = diag(1, 0.5).
        let m_inv_sqrt = vec![1.0, 0.0, 0.0, 0.5];
        let grad = vec![10.0, 10.0];
        let g_nat = reference_precondition_autotune_gradient(&m_inv_sqrt, &grad, 2);
        // Natural gradient pulls back the steep y axis:
        //   g_nat = (10, 5).
        assert!(approx_eq(g_nat[0], 10.0));
        assert!(approx_eq(g_nat[1], 5.0));
    }

    #[test]
    fn fixed_via_dispatches_natural_gradient_primitive() {
        let one = 1 << 16;
        let half = 1 << 15;
        let matrix = vec![one, 0, 0, half];
        let grad = vec![10 * one, 10 * one];

        let out = precondition_autotune_gradient_fixed_via(
            &NaturalGradientDispatcher,
            &crate::test_parity_oracles::policy(),
            &matrix,
            &grad,
            2,
        )
        .unwrap();

        assert_eq!(out, vec![10 * one, 5 * one]);
    }

    #[test]
    fn fixed_via_rejects_invalid_shapes() {
        let err = precondition_autotune_gradient_fixed_via(
            &NaturalGradientDispatcher,
            &crate::test_parity_oracles::policy(),
            &[1],
            &[1, 2],
            2,
        )
        .unwrap_err();

        assert!(matches!(err, SemanticExecutionError::InvalidRequest(_)));
    }

    #[test]
    fn fixed_via_with_scratch_reuses_input_buffers() {
        let one = 1 << 16;
        let matrix = vec![one, 0, 0, one];
        let grad = vec![3 * one, 5 * one];
        let mut scratch = NaturalGradientGpuScratch::default();
        let mut out = Vec::new();

        precondition_autotune_gradient_fixed_via_with_scratch_into(
            &NaturalGradientDispatcher,
            &crate::test_parity_oracles::policy(),
            &matrix,
            &grad,
            2,
            &mut scratch,
            &mut out,
        )
        .unwrap();
        let input_ptrs: Vec<*const u8> = scratch.inputs.iter().map(Vec::as_ptr).collect();
        precondition_autotune_gradient_fixed_via_with_scratch_into(
            &NaturalGradientDispatcher,
            &crate::test_parity_oracles::policy(),
            &matrix,
            &grad,
            2,
            &mut scratch,
            &mut out,
        )
        .unwrap();

        crate::solvers::test_helpers::assert_input_pointers_preserved(&input_ptrs, &scratch.inputs);
    }

    #[test]
    fn identity_fisher_block_is_diagonal_of_ones() {
        let id = identity_fisher_block(4);
        for i in 0..4 {
            assert!(approx_eq(id[i * 4 + i], 1.0));
            for j in 0..4 {
                if i != j {
                    assert!(approx_eq(id[i * 4 + j], 0.0));
                }
            }
        }
    }
}
