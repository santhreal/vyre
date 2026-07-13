//! End-to-end parity for `math::sheaf_spectral_clustering::dominant_spectrum_fixed_via`.
//!
//! The consumer's own `fixed_via_dispatches_sheaf_spectrum` test uses a `SpectrumDispatcher` MOCK
//! that IGNORES the `_program` IR and hand-computes `r*v` / max, so it validated buffer plumbing
//! but NEVER ran the real `sheaf_laplacian_eigenvalue` kernel (the mock-dispatcher-coherence gap,
//! and the mock even computes a DIFFERENT function than the kernel). Now that the kernel is a
//! correct closed-form diagonal eigensolver (`lambda = max r`, `v = e_argmax`), this runs the REAL
//! IR through the shared `ReferenceEvalDispatcher` and asserts the dispatched spectrum equals that
//! exact 16.16 closed form, the first end-to-end execution of the actual kernel through the
//! consumer's dispatch path.
#![forbid(unsafe_code)]
#![cfg(feature = "cpu-parity")]

use vyre_self_substrate::math::sheaf_spectral_clustering::dominant_spectrum_fixed_via;

mod common;
use common::ReferenceEvalDispatcher;

const ONE_FP: u32 = 1 << 16;

/// Exact 16.16 closed form of the diagonal dominant eigenpair: `(max r, e_argmax)` with a 0
/// running-max floor and first-arg-max tie-break.
fn exact(restriction_diag_fixed: &[u32]) -> (u32, Vec<u32>) {
    let mut max_r = 0u32;
    let mut argmax = 0usize;
    for (i, &ri) in restriction_diag_fixed.iter().enumerate() {
        if ri > max_r {
            max_r = ri;
            argmax = i;
        }
    }
    let v = (0..restriction_diag_fixed.len())
        .map(|j| if j == argmax { ONE_FP } else { 0 })
        .collect();
    (max_r, v)
}

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

#[test]
fn dominant_spectrum_fixed_via_matches_exact_closed_form() {
    let dispatcher = ReferenceEvalDispatcher;
    let mut state = 0x5EAF_10FEu32;
    let mut distinct_max_cases = 0u32;
    for case in 0..300u32 {
        let n = 1 + xorshift(&mut state) % 8; // 1..=8 diagonal entries (d = 1)
        let restriction: Vec<u32> = (0..n).map(|_| xorshift(&mut state) & 0x000F_FFFF).collect();
        // The initial vector is irrelevant to the diagonal eigenvector; feed arbitrary 16.16 values.
        let v_init: Vec<u32> = (0..n).map(|_| xorshift(&mut state) & 0x000F_FFFF).collect();

        let spectrum = dominant_spectrum_fixed_via(&dispatcher, &restriction, &v_init, n, 1, 8)
            .expect("dominant_spectrum_fixed_via must dispatch the eigenvalue kernel");
        let (exp_lambda, exp_v) = exact(&restriction);
        let max_count = restriction.iter().filter(|&&x| x == exp_lambda).count();
        if exp_lambda > 0 && max_count == 1 {
            distinct_max_cases += 1;
        }
        assert_eq!(
            spectrum.lambda, exp_lambda,
            "case {case} (n={n}): dispatched lambda {} != max r {exp_lambda} (r={restriction:?})",
            spectrum.lambda
        );
        assert_eq!(
            spectrum.eigenvector, exp_v,
            "case {case} (n={n}): dispatched eigenvector {:?} != e_argmax {exp_v:?} (r={restriction:?})",
            spectrum.eigenvector
        );
    }
    assert!(
        distinct_max_cases > 150,
        "only {distinct_max_cases}/300 cases had a unique strict maximum, strengthen the diagonal \
         distribution so the arg-max eigenvector is exercised"
    );
}

#[test]
fn dominant_spectrum_fixed_via_picks_the_max_diagonal() {
    // r = [0.5, 2.0, 1.0]; the dispatched kernel must return (2.0, e_1), NOT the r*v the old mock
    // computed. Initial vector deliberately non-uniform to prove it does not influence the result.
    let dispatcher = ReferenceEvalDispatcher;
    let restriction = vec![ONE_FP / 2, 2 * ONE_FP, ONE_FP];
    let v_init = vec![8 * ONE_FP, 3 * ONE_FP, 5 * ONE_FP];
    let spectrum = dominant_spectrum_fixed_via(&dispatcher, &restriction, &v_init, 3, 1, 4)
        .expect("dominant_spectrum_fixed_via must dispatch");
    assert_eq!(
        spectrum.lambda,
        2 * ONE_FP,
        "lambda must be the max diagonal entry 2.0"
    );
    assert_eq!(
        spectrum.eigenvector,
        vec![0, ONE_FP, 0],
        "eigenvector must be the unit indicator at the arg-max index, independent of v_init"
    );
}
