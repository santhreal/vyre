//! End-to-end parity for the COMPOSITE
//! `math::differentiable_autotune::natural_config_gradient_magnitude_pre_exp_fixed_via`, the
//! Fisher-preconditioned fixed-point autotune gradient, through the shared faithful
//! [`common::ReferenceEvalDispatcher`].
//!
//! Closes a mock-dispatcher-coherence gap (see BACKLOG `SWEEP-self-substrate-mock-dispatcher-coherence`):
//! the two constituent kernels are each parity-covered in isolation (softmax normalization by
//! `softmax_pick_config_via_reference_parity`, the 16.16 Fisher matvec by
//! `natural_gradient_via_reference_parity`), but the COMPOSITION, where dispatch 1's normalized
//! probabilities feed as the gradient input of dispatch 2, both through the same faithful boundary 
//! was exercised ONLY by the consumer's own in-file `DifferentiableDispatcher` mock (which ignores the
//! IR and hand-computes each stage). This is the FIRST-EVER execution of the full
//! softmax→Fisher-matvec chain through a boundary that models the real backend.
//!
//! Contract (audited CLEAN): the composite runs two dispatches on one dispatcher 
//!   (1) `softmax_step`:  pre_exp RO + out RW  = 2 IC, decode outputs[0] → `probabilities`;
//!   (2) `natural_gradient_block_apply`:  M_inv_sqrt RO + grad(=probabilities) RO + out RW = 3 IC,
//!       decode outputs[0] → the natural gradient.
//! Both stages are EXACT integer arithmetic, so the oracle composes their bit-for-bit replicas → the
//! comparison is BIT-EXACT (no tolerance):
//!   `prob[j] = (pre_exp[j] << 16) / max(Σ pre_exp, 1)`   (u32 shl keeping low 32 bits, integer div)
//!   `nat[t]  = Σ_j fixed_mul_16_16(M[t*n+j], prob[j])`   (SIGNED `((a as i32 as i64 * b as i32 as
//!   i64) >> 16) as i32 as u32`, wrapping u32 add)
#![cfg(feature = "cpu-parity")]

use vyre_self_substrate::math::differentiable_autotune::natural_config_gradient_magnitude_pre_exp_fixed_via;

mod common;
use common::fixed_mul as fixed_mul_16_16;
use common::ReferenceEvalDispatcher;

const FIXED_ONE: u32 = 1 << 16;

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// Exact replica of the softmax_step IR: normalized 16.16 probabilities.
fn softmax_fixed(pre_exp: &[u32]) -> Vec<u32> {
    let sum = pre_exp.iter().fold(0u32, |a, &b| a.wrapping_add(b));
    let sum_safe = if sum == 0 { 1 } else { sum };
    pre_exp.iter().map(|&p| (p << 16) / sum_safe).collect()
}

/// Exact composite oracle: softmax normalize, then a 16.16 Fisher matvec over the probabilities.
fn natural_config_gradient(pre_exp: &[u32], m_inv_sqrt: &[u32]) -> Vec<u32> {
    let prob = softmax_fixed(pre_exp);
    let n = pre_exp.len();
    (0..n)
        .map(|t| {
            let mut acc = 0u32;
            for j in 0..n {
                acc = acc.wrapping_add(fixed_mul_16_16(m_inv_sqrt[t * n + j], prob[j]));
            }
            acc
        })
        .collect()
}

#[test]
fn natural_config_gradient_via_matches_exact_composite_oracle() {
    let dispatcher = ReferenceEvalDispatcher;
    let mut state = 0x1A_7E_00_01u32;
    let mut nonzero_gradient = 0u32;
    let mut off_diagonal_mixing = 0u32;
    for case in 0..400u32 {
        let n = 1 + (case % 12) as usize;
        // pre_exp[i] = exp(x[i]-max) in 16.16 lives in (0, 1.0]; keep it in [1, FIXED_ONE-1] so every
        // probability stays nonzero (the FIXED_ONE `p<<16`→0 wrap edge is covered by the softmax suite).
        let pre_exp: Vec<u32> = (0..n)
            .map(|_| 1 + xorshift(&mut state) % (FIXED_ONE - 1))
            .collect();
        // M_inv_sqrt is an n×n 16.16 inverse-Fisher square-root block. Values in [0, ~1.0) keep the
        // matvec well-scaled while genuinely mixing across candidates.
        let m_inv_sqrt: Vec<u32> = (0..n * n)
            .map(|_| xorshift(&mut state) % FIXED_ONE)
            .collect();

        let got =
            natural_config_gradient_magnitude_pre_exp_fixed_via(&dispatcher, &pre_exp, &m_inv_sqrt)
                .expect(
                    "natural_config_gradient_magnitude_pre_exp_fixed_via must dispatch both stages",
                );
        let want = natural_config_gradient(&pre_exp, &m_inv_sqrt);
        assert_eq!(
            got, want,
            "case {case}: composite natural gradient must match the exact oracle; n={n} pre_exp={pre_exp:?}"
        );

        if got.iter().any(|&v| v != 0) {
            nonzero_gradient += 1;
        }
        // Row t mixes column j != t → the off-diagonal Fisher coupling actually contributes.
        if n >= 2 {
            off_diagonal_mixing += 1;
        }
    }
    assert!(
        nonzero_gradient > 300,
        "composite sweep must produce nonzero natural gradients, got {nonzero_gradient}"
    );
    assert!(
        off_diagonal_mixing > 200,
        "composite sweep must exercise multi-candidate Fisher coupling, got {off_diagonal_mixing}"
    );
}

/// A signed 16.16 Fisher-block entry in roughly `[-2.0, 2.0)`: a 17-bit magnitude, optionally
/// negated. An inverse-Fisher square root is symmetric PSD, but its OFF-DIAGONAL entries are freely
/// negative (only the diagonal of a PSD matrix must be non-negative), so a faithful autotune Fisher
/// block routinely carries negative coupling (the operand class the old unsigned multiply corrupted).
fn signed_fisher(state: &mut u32) -> u32 {
    let magnitude = (xorshift(state) & 0x0001_FFFF) as i32; // [0.0, 2.0) in 16.16
    let signed = if xorshift(state) & 1 == 0 {
        magnitude
    } else {
        -magnitude
    };
    signed as u32
}

fn to_fixed(v: f64) -> u32 {
    (v * 65536.0).round() as i64 as u32
}

#[test]
fn natural_config_gradient_via_matches_signed_composite_with_negative_fisher_coupling() {
    // The softmax stage always yields NON-negative probabilities, but the Fisher block `M_inv_sqrt`
    // carries negative off-diagonal coupling, so `nat[t] = Σ_j M[t,j]·prob[j]` is genuinely SIGNED.
    // The base sweep draws M from `xorshift % FIXED_ONE` (all non-negative) and never sends a negative
    // Fisher term through the composite. This sweep draws a SIGNED M and asserts the two-stage
    // dispatch bit-exactly matches the signed oracle, locking the signed `fixed_mul_16_16_expr` fix
    // across the softmax→Fisher-matvec composition (pre-fix, a negative `M[t,j]·prob[j]` term diverged).
    let dispatcher = ReferenceEvalDispatcher;
    let mut state = 0x2545_F491u32;
    let mut neg_fisher_inputs = 0u32;
    let mut neg_outputs = 0u32;
    let mut nonzero_gradient = 0u32;
    for case in 0..400u32 {
        let n = 1 + (case % 12) as usize;
        let pre_exp: Vec<u32> = (0..n)
            .map(|_| 1 + xorshift(&mut state) % (FIXED_ONE - 1))
            .collect();
        let m_inv_sqrt: Vec<u32> = (0..n * n).map(|_| signed_fisher(&mut state)).collect();

        neg_fisher_inputs += m_inv_sqrt.iter().filter(|&&v| (v as i32) < 0).count() as u32;

        let got =
            natural_config_gradient_magnitude_pre_exp_fixed_via(&dispatcher, &pre_exp, &m_inv_sqrt)
                .expect(
                    "natural_config_gradient_magnitude_pre_exp_fixed_via must dispatch both stages",
                );
        let want = natural_config_gradient(&pre_exp, &m_inv_sqrt);
        assert_eq!(
            got, want,
            "case {case}: SIGNED composite natural gradient must match the exact signed oracle; \
             n={n} pre_exp={pre_exp:?} m_inv_sqrt={m_inv_sqrt:?}"
        );

        if got.iter().any(|&v| v != 0) {
            nonzero_gradient += 1;
        }
        neg_outputs += want.iter().filter(|&&v| (v as i32) < 0).count() as u32;
    }
    assert!(
        neg_fisher_inputs > 500,
        "sweep must feed many negative Fisher-coupling entries, got {neg_fisher_inputs}"
    );
    assert!(
        neg_outputs > 100,
        "signed Fisher coupling must produce negative natural-gradient entries, got {neg_outputs}"
    );
    assert!(
        nonzero_gradient > 300,
        "composite sweep must produce nonzero natural gradients, got {nonzero_gradient}"
    );
}

#[test]
fn natural_config_gradient_via_hand_checked_negative_fisher_coupling() {
    // A graph-Laplacian-style Fisher block M = [[1.0, -1.0], [-1.0, 1.0]] (PSD, eigenvalues 0 and 2)
    // over asymmetric probabilities prob = [0.25, 0.75] (from pre_exp = [0.25, 0.75], Σ = 1.0):
    //   nat[0] = (1.0)(0.25) + (-1.0)(0.75) = -0.5
    //   nat[1] = (-1.0)(0.25) + (1.0)(0.75) =  0.5
    let d = ReferenceEvalDispatcher;
    let pre_exp = [FIXED_ONE / 4, FIXED_ONE * 3 / 4];
    let m = vec![to_fixed(1.0), to_fixed(-1.0), to_fixed(-1.0), to_fixed(1.0)];
    let got = natural_config_gradient_magnitude_pre_exp_fixed_via(&d, &pre_exp, &m).unwrap();
    let want = natural_config_gradient(&pre_exp, &m);
    assert_eq!(
        want,
        vec![to_fixed(-0.5), to_fixed(0.5)],
        "sanity: signed Fisher-coupled natural gradient = [-0.5, 0.5]"
    );
    assert_eq!(
        got, want,
        "the composite must preserve the sign of the Fisher-coupled gradient: [-0.5, 0.5]"
    );
}

#[test]
fn natural_config_gradient_via_hand_checked_identity_fisher() {
    let d = ReferenceEvalDispatcher;
    // With an IDENTITY Fisher block (16.16 one on the diagonal, zero off-diagonal), preconditioning is
    // a no-op: nat[t] = fixed_mul(1.0, prob[t]) = prob[t]. So the composite must reproduce the plain
    // softmax probabilities exactly.
    let pre_exp = [FIXED_ONE / 4, FIXED_ONE / 4, FIXED_ONE / 2];
    let n = pre_exp.len();
    let mut m = vec![0u32; n * n];
    for i in 0..n {
        m[i * n + i] = FIXED_ONE; // 1.0 on the diagonal
    }
    let got = natural_config_gradient_magnitude_pre_exp_fixed_via(&d, &pre_exp, &m).unwrap();
    let prob = softmax_fixed(&pre_exp);
    assert_eq!(
        got, prob,
        "identity Fisher must pass the softmax probabilities through unchanged"
    );

    // A DOUBLING diagonal (2.0 on each diagonal) scales every probability by 2 in 16.16.
    let mut m2 = vec![0u32; n * n];
    for i in 0..n {
        m2[i * n + i] = 2 * FIXED_ONE; // 2.0
    }
    let got2 = natural_config_gradient_magnitude_pre_exp_fixed_via(&d, &pre_exp, &m2).unwrap();
    let want2: Vec<u32> = prob
        .iter()
        .map(|&p| fixed_mul_16_16(2 * FIXED_ONE, p))
        .collect();
    assert_eq!(
        got2, want2,
        "a 2.0 diagonal Fisher doubles each probability in fixed point"
    );
    // Sanity: doubling really is ~2× the identity result (within fixed-point rounding).
    assert!(
        got2.iter().zip(&got).all(|(&g2, &g1)| g2 >= g1),
        "doubling must not shrink any component"
    );
}
