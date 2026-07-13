//! End-to-end parity for `math::amg_pass_solver::smooth_matroid_flow_fixed_via`: one full two-level
//! algebraic-multigrid (AMG) V-cycle (pre-smooth → restrict → coarse-solve → prolong → post-smooth) 
//! through the shared faithful [`common::ReferenceEvalDispatcher`].
//!
//! Closes the LAST mock-dispatcher-coherence gap in the SWEEP drain (see BACKLOG
//! `SWEEP-self-substrate-mock-dispatcher-coherence`): the 11-buffer `amg_v_cycle` IR is not run through a
//! faithful dispatch boundary by any `vyre-primitives/tests/*` file. This is the FIRST-EVER execution of
//! the AMG V-cycle kernel through a boundary that models the real backend.
//!
//! Contract (audited CLEAN): the IR binds a RO(0) + b RO(1) + x RW(2) + r_mat RO(3) + p_mat RO(4) +
//! a_c RO(5) + omega RO(6) + scratch_fine RW(7) + scratch_coarse_b RW(8) + scratch_coarse_x RW(9) +
//! temp_coarse RW(10) = 11 input-consuming buffers (NO backend-allocated outputs), and the via feeds
//! exactly 11 inputs (the 5 RW slots zero/x-initialized) → NO over/under-feed. `reference_eval` returns
//! the writable buffers in binding order, so `outputs[0]` is the post-smoothed `x`, exactly what the via
//! decodes. This is the FULL 11-buffer contract, the widest in the substrate.
//!
//! FIXED-vs-FLOAT (calibrated tolerance, NOT bit-exact): the GPU IR is 16.16 fixed-point (fixed_mul_16_16
//! + Jacobi division) while the only oracle `reference_smooth_matroid_flow` is f64, so accumulated
//! fixed-point rounding is compared within a tolerance.
//!
//! WHY THIS CONVERSION WAS "HARD", and the FIX it surfaced: the fixed path originally MANGLED any
//! NEGATIVE 16.16 intermediate, because `fixed_mul_16_16_expr` took the UNSIGNED 64-bit product and the
//! Jacobi delta divided with UNSIGNED `Expr::div`. A negative residual `b − A·x` (which appears the moment
//! the coarse correction overshoots) was then treated as a giant ~2^32 value → garbage (got≈10813 vs
//! want≈0.39). This suite DROVE that bug out and it is now FIXED at the source (signed high-word
//! correction in `fixed_mul_16_16_expr` + new `fixed_sdiv_by_positive_expr` wired into both Jacobi
//! builders, see BACKLOG `FIXED-amg-fixed-path-unsigned-mul-negatives`). Two regimes are covered:
//!   • `_over_diagonal_dominant_systems`: a well-conditioned NON-NEGATIVE domain (diagonal `A`, LARGE
//!     coarse diagonal so the correction cannot overshoot) → TIGHT tolerance parity, isolating dispatch
//!     correctness from fixed-point drift.
//!   • `_with_negative_intermediates`: SMALL coarse diagonals that force overshoot → negative residuals →
//!     exercises (and regression-locks) the signed multiply + signed divide fix.
//! Every input value is a multiple of 0.5 or a power of two → exact in 16.16.
#![cfg(feature = "cpu-parity")]

use vyre_self_substrate::amg_pass_solver::{
    reference_smooth_matroid_flow, smooth_matroid_flow_fixed_via,
};

mod common;
use common::ReferenceEvalDispatcher;

const FIXED_ONE: f64 = 65536.0;

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// Convert a non-negative f64 (a multiple of 0.5 or a power of two in this suite) to 16.16 fixed-point.
fn to_fixed(v: f64) -> u32 {
    assert!(
        v >= 0.0,
        "this suite feeds only non-negative fixed-point inputs"
    );
    (v * FIXED_ONE).round() as u32
}

/// Decode a 16.16 fixed-point output as the SIGNED value it encodes (correct for both signs; identical
/// to the unsigned reading for any non-negative magnitude < 2^31).
fn from_fixed(v: u32) -> f64 {
    f64::from(v as i32) / FIXED_ONE
}

/// A fine-level diagonal value in {2, 4} (exact 16.16 division). Capped at 4 (not 8) so the coarse
/// correction `D·(P·xc)` provably stays below the pre-smooth residual `r = (1−ω)·b`, see the
/// no-overshoot bound in the sweep, keeping every intermediate non-negative and the unsigned fixed
/// multiply faithful.
fn fine_diag(state: &mut u32) -> f64 {
    if xorshift(state) & 1 == 0 {
        2.0
    } else {
        4.0
    }
}

/// A restriction-operator entry in {0, 0.5} (kept small so the coarse RHS `b_c = R·r` and thus the
/// prolonged correction cannot overshoot the residual).
fn restrict_entry(state: &mut u32) -> f64 {
    if xorshift(state) % 3 == 0 {
        0.5
    } else {
        0.0
    }
}

/// A prolongation-operator entry in {0, 0.5, 1.0} (non-negative).
fn prolong_entry(state: &mut u32) -> f64 {
    match xorshift(state) % 3 {
        0 => 0.0,
        1 => 0.5,
        _ => 1.0,
    }
}

/// A right-hand-side value in {1.0, 1.5, 2.0}, kept ≥ 1 so the pre-smooth residual `r = (1−ω)·b`
/// dominates the (deliberately tiny, `a_c`-damped) coarse correction, guaranteeing non-negativity.
fn rhs_entry(state: &mut u32) -> f64 {
    1.0 + 0.5 * f64::from(xorshift(state) % 3)
}

/// The coarse-level diagonal is fixed LARGE (64) so the coarse solve `a_c·xc = b_c` under-corrects:
/// `xc ≤ b_c / 64` is tiny, so the prolonged correction `D·(P·xc)` stays strictly below the pre-smooth
/// residual `r = (1−ω)·b ≥ 0.34` for every fine row (bound: `D≤4, P·xc ≤ 3·(2·0.68/64) ≈ 0.064`, so
/// `D·(P·xc) ≤ 0.255 < 0.34`). This keeps the post-smooth residual `b − A·x` non-negative, the domain
/// where the UNSIGNED fixed multiply faithfully mirrors the f64 reference, while still fully dispatching
/// and arithmetically exercising the restrict / coarse-Jacobi / prolong matvecs.
const COARSE_DIAG: f64 = 64.0;

/// ∞-norm residual ‖A·x − b‖∞ of an f64 iterate (independent correctness anchor).
fn inf_residual(a: &[f64], b: &[f64], x: &[f64], n: usize) -> f64 {
    let mut worst = 0.0_f64;
    for i in 0..n {
        let row: f64 = (0..n).map(|j| a[i * n + j] * x[j]).sum();
        worst = worst.max((row - b[i]).abs());
    }
    worst
}

#[test]
fn smooth_matroid_flow_via_matches_reference_over_diagonal_dominant_systems() {
    let d = ReferenceEvalDispatcher;
    let mut state = 0xA3_69_00_01u32;
    let mut nonzero = 0u32;
    let mut residual_reduced = 0u32;
    for case in 0..300u32 {
        // n_fine ∈ {2,3,4}; n_coarse ∈ [1, n_fine).
        let n_fine = 2 + case % 3;
        let n_coarse = 1 + xorshift(&mut state) % (n_fine - 1);
        let nf = n_fine as usize;
        let nc = n_coarse as usize;

        // DIAGONAL fine system → residual after pre-smoothing is b_i·(1−ω) ≥ 0, so with a LARGE coarse
        // diagonal (COARSE_DIAG) damping the prolonged correction below that residual, every downstream
        // quantity (residual, coarse RHS, iterates, POST-smooth residual) stays non-negative and the
        // unsigned fixed multiply is faithful. Coupling is carried by the non-negative R/P operators.
        let mut a = vec![0.0_f64; nf * nf];
        for i in 0..nf {
            a[i * nf + i] = fine_diag(&mut state);
        }
        let b: Vec<f64> = (0..nf).map(|_| rhs_entry(&mut state)).collect();
        let x0 = vec![0.0_f64; nf];
        let r_mat: Vec<f64> = (0..nc * nf).map(|_| restrict_entry(&mut state)).collect();
        let p_mat: Vec<f64> = (0..nf * nc).map(|_| prolong_entry(&mut state)).collect();
        let mut a_c = vec![0.0_f64; nc * nc];
        for i in 0..nc {
            a_c[i * nc + i] = COARSE_DIAG;
        }

        let a_fx: Vec<u32> = a.iter().map(|&v| to_fixed(v)).collect();
        let b_fx: Vec<u32> = b.iter().map(|&v| to_fixed(v)).collect();
        let x_fx: Vec<u32> = x0.iter().map(|&v| to_fixed(v)).collect();
        let r_fx: Vec<u32> = r_mat.iter().map(|&v| to_fixed(v)).collect();
        let p_fx: Vec<u32> = p_mat.iter().map(|&v| to_fixed(v)).collect();
        let ac_fx: Vec<u32> = a_c.iter().map(|&v| to_fixed(v)).collect();

        let got_fixed = smooth_matroid_flow_fixed_via(
            &d, &a_fx, &b_fx, &x_fx, &r_fx, &p_fx, &ac_fx, n_fine, n_coarse,
        )
        .expect("smooth_matroid_flow_fixed_via must dispatch the AMG V-cycle");
        let got: Vec<f64> = got_fixed.iter().map(|&v| from_fixed(v)).collect();
        let want =
            reference_smooth_matroid_flow(&a, &b, &x0, &r_mat, &p_mat, &a_c, n_fine, n_coarse);

        assert_eq!(
            got.len(),
            want.len(),
            "case {case}: output length must match"
        );
        for i in 0..nf {
            let tol = 0.01 + 0.01 * want[i].abs();
            assert!(
                (got[i] - want[i]).abs() <= tol,
                "case {case}: fixed AMG V-cycle diverged from the f64 reference at row {i}: \
                 got={} want={} tol={tol}; n_fine={n_fine} n_coarse={n_coarse} diag(A)={:?} b={b:?} \
                 R={r_mat:?} P={p_mat:?} diag(A_c)={:?}",
                got[i],
                want[i],
                (0..nf).map(|k| a[k * nf + k]).collect::<Vec<_>>(),
                (0..nc).map(|k| a_c[k * nc + k]).collect::<Vec<_>>(),
            );
        }

        if got.iter().any(|&v| v != 0.0) {
            nonzero += 1;
        }
        // Independent anchor: one V-cycle must not INCREASE the residual (‖A·x0−b‖∞ = ‖b‖∞ since x0=0).
        let init_resid = inf_residual(&a, &b, &x0, nf);
        let final_resid = inf_residual(&a, &b, &got, nf);
        assert!(
            final_resid <= init_resid + 1e-9,
            "case {case}: a V-cycle must not increase the residual; init={init_resid} final={final_resid}"
        );
        if final_resid < init_resid - 1e-6 {
            residual_reduced += 1;
        }
    }
    assert!(
        nonzero > 250,
        "sweep must produce nonzero smoothed iterates, got {nonzero}"
    );
    assert!(
        residual_reduced > 200,
        "the V-cycle must actually reduce the residual on most systems, got {residual_reduced}"
    );
}

#[test]
fn smooth_matroid_flow_via_hand_checked_two_level() {
    let d = ReferenceEvalDispatcher;
    // n_fine=2, n_coarse=1. A = diag(4,4), b=[4,4], x0=0, R=[0.5,0.5], P=[1;1], a_c=[4].
    let n_fine = 2u32;
    let n_coarse = 1u32;
    let a = [4.0, 0.0, 0.0, 4.0];
    let b = [4.0, 4.0];
    let x0 = [0.0, 0.0];
    let r_mat = [0.5, 0.5];
    let p_mat = [1.0, 1.0];
    let a_c = [4.0];

    let a_fx: Vec<u32> = a.iter().map(|&v| to_fixed(v)).collect();
    let b_fx: Vec<u32> = b.iter().map(|&v| to_fixed(v)).collect();
    let x_fx: Vec<u32> = x0.iter().map(|&v| to_fixed(v)).collect();
    let r_fx: Vec<u32> = r_mat.iter().map(|&v| to_fixed(v)).collect();
    let p_fx: Vec<u32> = p_mat.iter().map(|&v| to_fixed(v)).collect();
    let ac_fx: Vec<u32> = a_c.iter().map(|&v| to_fixed(v)).collect();

    let got_fixed = smooth_matroid_flow_fixed_via(
        &d, &a_fx, &b_fx, &x_fx, &r_fx, &p_fx, &ac_fx, n_fine, n_coarse,
    )
    .unwrap();
    let got: Vec<f64> = got_fixed.iter().map(|&v| from_fixed(v)).collect();
    let want = reference_smooth_matroid_flow(&a, &b, &x0, &r_mat, &p_mat, &a_c, n_fine, n_coarse);

    for i in 0..2 {
        let tol = 0.01 + 0.01 * want[i].abs();
        assert!(
            (got[i] - want[i]).abs() <= tol,
            "hand-checked V-cycle row {i}: got={} want={} tol={tol}",
            got[i],
            want[i]
        );
    }
    // Independent anchors: the smoothed iterate is strictly positive (driven up from x0=0 by b>0),
    // and one V-cycle reduces the initial residual ‖b‖∞ = 4.
    assert!(
        got.iter().all(|&v| v > 0.0),
        "positive RHS drives the iterate positive: got={got:?}"
    );
    let init_resid = inf_residual(&a, &b, &x0, 2);
    let final_resid = inf_residual(&a, &b, &got, 2);
    assert!(
        final_resid < init_resid,
        "the V-cycle reduces the residual: init={init_resid} final={final_resid}"
    );
}

#[test]
fn smooth_matroid_flow_via_matches_reference_with_negative_intermediates() {
    // REGRESSION for the signed-multiply fix (BACKLOG `LIMITATION-amg-fixed-path-unsigned-mul-negatives`,
    // now FIXED in fixed_mul_16_16_expr): a SMALL coarse diagonal makes the coarse correction OVERSHOOT
    // (post-prolong A·x > b), so the post-smooth residual `b − A·x` is NEGATIVE. Under the old UNSIGNED
    // fixed multiply the fixed path produced garbage (got≈10813 vs want≈0.39); the signed multiply now
    // tracks the f64 reference within fixed-point tolerance even through negative intermediates.
    let d = ReferenceEvalDispatcher;

    // The EXACT configuration that reproduced the corruption: n_fine=4, diag(A)=[2,4,4,2], b=[2,1.5,1.5,1],
    // R=[1,1,0.5,0], P=[0.5,0.5,0.5,0], a_c=[4] (small → overshoot).
    #[rustfmt::skip]
    let a = [
        2.0, 0.0, 0.0, 0.0,
        0.0, 4.0, 0.0, 0.0,
        0.0, 0.0, 4.0, 0.0,
        0.0, 0.0, 0.0, 2.0,
    ];
    let b = [2.0, 1.5, 1.5, 1.0];
    let x0 = [0.0, 0.0, 0.0, 0.0];
    let r_mat = [1.0, 1.0, 0.5, 0.0];
    let p_mat = [0.5, 0.5, 0.5, 0.0];
    let a_c = [4.0];

    let a_fx: Vec<u32> = a.iter().map(|&v| to_fixed(v)).collect();
    let b_fx: Vec<u32> = b.iter().map(|&v| to_fixed(v)).collect();
    let x_fx: Vec<u32> = x0.iter().map(|&v| to_fixed(v)).collect();
    let r_fx: Vec<u32> = r_mat.iter().map(|&v| to_fixed(v)).collect();
    let p_fx: Vec<u32> = p_mat.iter().map(|&v| to_fixed(v)).collect();
    let ac_fx: Vec<u32> = a_c.iter().map(|&v| to_fixed(v)).collect();

    let got_fixed =
        smooth_matroid_flow_fixed_via(&d, &a_fx, &b_fx, &x_fx, &r_fx, &p_fx, &ac_fx, 4, 1).unwrap();
    let got: Vec<f64> = got_fixed.iter().map(|&v| from_fixed(v)).collect();
    let want = reference_smooth_matroid_flow(&a, &b, &x0, &r_mat, &p_mat, &a_c, 4, 1);

    for i in 0..4 {
        // No blowup: the old unsigned bug produced ~10813 here; a correct result is O(1).
        assert!(
            got[i].abs() < 100.0,
            "row {i} must not be corrupted garbage (old unsigned bug gave ~10813): got={}",
            got[i]
        );
        let tol = 0.03 + 0.03 * want[i].abs();
        assert!(
            (got[i] - want[i]).abs() <= tol,
            "row {i}: signed fixed AMG must track the f64 reference through negative intermediates: \
             got={} want={} tol={tol}",
            got[i],
            want[i]
        );
    }

    // A broader overshoot sweep: small coarse diagonals (2 or 4) with heavier coupling so many systems
    // drive the correction into the negative-residual regime.
    let mut state = 0x5E_ED_00_01u32;
    let mut no_blowup = 0u32;
    for case in 0..200u32 {
        let n_fine = 3 + case % 2; // 3..4
        let n_coarse = 1 + xorshift(&mut state) % (n_fine - 1);
        let nf = n_fine as usize;
        let nc = n_coarse as usize;

        let mut am = vec![0.0_f64; nf * nf];
        for i in 0..nf {
            am[i * nf + i] = fine_diag(&mut state);
        }
        let bv: Vec<f64> = (0..nf).map(|_| rhs_entry(&mut state)).collect();
        let x0v = vec![0.0_f64; nf];
        // Heavier transfer entries (0..=1) + small a_c → overshoot.
        let rm: Vec<f64> = (0..nc * nf).map(|_| prolong_entry(&mut state)).collect();
        let pm: Vec<f64> = (0..nf * nc).map(|_| prolong_entry(&mut state)).collect();
        let mut acm = vec![0.0_f64; nc * nc];
        for i in 0..nc {
            acm[i * nc + i] = fine_diag(&mut state); // 2 or 4, small, induces overshoot
        }

        let a_fx: Vec<u32> = am.iter().map(|&v| to_fixed(v)).collect();
        let b_fx: Vec<u32> = bv.iter().map(|&v| to_fixed(v)).collect();
        let x_fx: Vec<u32> = x0v.iter().map(|&v| to_fixed(v)).collect();
        let r_fx: Vec<u32> = rm.iter().map(|&v| to_fixed(v)).collect();
        let p_fx: Vec<u32> = pm.iter().map(|&v| to_fixed(v)).collect();
        let ac_fx: Vec<u32> = acm.iter().map(|&v| to_fixed(v)).collect();

        let got_fixed = smooth_matroid_flow_fixed_via(
            &d, &a_fx, &b_fx, &x_fx, &r_fx, &p_fx, &ac_fx, n_fine, n_coarse,
        )
        .expect("dispatch must succeed");
        let got: Vec<f64> = got_fixed.iter().map(|&v| from_fixed(v)).collect();
        let want = reference_smooth_matroid_flow(&am, &bv, &x0v, &rm, &pm, &acm, n_fine, n_coarse);
        for i in 0..nf {
            assert!(
                got[i].abs() < 1000.0,
                "case {case} row {i}: no unsigned-mul blowup; got={}",
                got[i]
            );
            let tol = 0.05 + 0.05 * want[i].abs();
            assert!(
                (got[i] - want[i]).abs() <= tol,
                "case {case} row {i}: signed fixed AMG diverged: got={} want={} tol={tol}; \
                 diag(A)={:?} b={bv:?} R={rm:?} P={pm:?} diag(A_c)={:?}",
                got[i],
                want[i],
                (0..nf).map(|k| am[k * nf + k]).collect::<Vec<_>>(),
                (0..nc).map(|k| acm[k * nc + k]).collect::<Vec<_>>(),
            );
        }
        no_blowup += 1;
    }
    assert_eq!(
        no_blowup, 200,
        "every overshoot case must dispatch without corruption"
    );
}
