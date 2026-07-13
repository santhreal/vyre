//! End-to-end parity for `math::multigrid_matroid_solver::matroid_solve_step_fixed_via`
//! through the shared faithful [`common::ReferenceEvalDispatcher`].
//!
//! Closes a mock-dispatcher-coherence gap (see BACKLOG `SWEEP-self-substrate-mock-dispatcher-coherence`):
//! `jacobi_smooth_step`'s IR is run by NO `vyre-primitives/tests/*` file and the consumer's only
//! coverage is its own in-file dispatcher, so this is the FIRST-EVER execution of the weighted-Jacobi
//! matroid smoothing kernel through a dispatch boundary that models the real backend.
//!
//! `jacobi_smooth_step` binds a RO(0) + b RO(1) + x_in RO(2) + omega RO(3) + x_out plain-ReadWrite(4)
//! = 5 input-consuming (no backend-allocated output → no over/under-feed; the consumer correctly passes
//! a/b/x_in/omega plus a zero-filled `x_out` slot and decodes the sole writable buffer at outputs[0]).
//! Per row `t` the kernel computes (16.16 fixed-point, multigrid.rs):
//!   `res   = b[t] - Σ_j fixed_mul(a[t*n+j], x_in[j])`
//!   `diag_units = (a[t*n+t]==0 ? 1 : a[t*n+t]) < 1.0 ? 1 : a[t*n+t] >> 16`
//!   `x_out[t] = x_in[t] + fixed_mul(omega, res) / diag_units`
//! where `fixed_mul(a,b)` is the SIGNED 16.16 multiply (bits [16..48] of the signed 64-bit product),
//! `sub`/`add` are u32 wrapping (two's-complement-correct), and the divide is a SIGNED truncating divide
//! by an always-≥1 divisor. The signed mul + signed divide are load-bearing: a weighted-Jacobi residual
//! `b − A·x` is routinely NEGATIVE, and the earlier unsigned forms corrupted `fixed_mul(omega, res)` /
//! `res / diag` for that case (fixed in `fixed_mul_16_16_expr` + `fixed_sdiv_by_positive_expr`; see
//! BACKLOG `FIXED-amg-fixed-path-unsigned-mul-negatives`). Every operation is exactly reproducible in
//! u32, so the oracle here is BIT-EXACT (no tolerance) (the same exact-fixed-point route mz_project used).
#![cfg(feature = "cpu-parity")]

use vyre_self_substrate::math::multigrid_matroid_solver::matroid_solve_step_fixed_via;

mod common;
use common::ReferenceEvalDispatcher;
use common::{fixed_mul, fixed_sdiv_by_positive as sdiv_by_positive};

const FIXED_ONE: u32 = 1 << 16;

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// Exact u32 oracle for one weighted-Jacobi matroid solve step, mirroring the kernel bit-for-bit
/// (including its SIGNED fixed multiply + SIGNED divide).
fn jacobi_fixed(a: &[u32], b: &[u32], x_in: &[u32], omega: u32, n: usize) -> Vec<u32> {
    (0..n)
        .map(|t| {
            let row_base = t * n;
            // residual = b[t] - Σ_j A[t,j] · x_in[j]   (the loop runs over ALL j, incl. the diagonal)
            let mut res = b[t];
            for j in 0..n {
                res = res.wrapping_sub(fixed_mul(a[row_base + j], x_in[j]));
            }
            let diag = a[row_base + t];
            let diag_safe = if diag == 0 { 1 } else { diag };
            let diag_units = if diag_safe < FIXED_ONE {
                1
            } else {
                diag_safe >> 16
            };
            let delta = sdiv_by_positive(fixed_mul(omega, res), diag_units);
            x_in[t].wrapping_add(delta)
        })
        .collect()
}

#[test]
fn matroid_step_via_matches_exact_fixed_point_oracle_over_generated_systems() {
    let dispatcher = ReferenceEvalDispatcher;
    let mut state = 0x4A_C0_B1_01u32; // arbitrary nonzero seed
    let mut nontrivial = 0u32;
    let mut hit_zero_diag = 0u32;
    let mut hit_sub_unit_diag = 0u32;
    let mut hit_int_diag = 0u32;
    for case in 0..500u32 {
        let n = (1 + (case % 6)) as usize;

        // Build an n×n system. The diagonal cycles through the three `diag_units` regimes so every
        // branch of the kernel's guard is exercised across the sweep:
        //   regime 0 → diag == 0        (diag_safe = 1, diag_units = 1)
        //   regime 1 → 0 < diag < 1.0   (diag_units = 1)
        //   regime 2 → diag >= 1.0      (diag_units = diag >> 16, the real integer scale)
        let mut a = vec![0u32; n * n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    a[i * n + j] = match (case as usize + i) % 3 {
                        0 => 0,
                        1 => 1 + xorshift(&mut state) % (FIXED_ONE - 1), // [1, 1.0)
                        _ => FIXED_ONE + xorshift(&mut state) % (8 * FIXED_ONE), // [1.0, 9.0)
                    };
                } else {
                    // small off-diagonals in [0, 0.5) keep the matrix loosely diagonally sensible
                    a[i * n + j] = xorshift(&mut state) % (FIXED_ONE / 2);
                }
            }
        }
        let b: Vec<u32> = (0..n)
            .map(|_| xorshift(&mut state) % (4 * FIXED_ONE))
            .collect();
        let x_in: Vec<u32> = (0..n)
            .map(|_| xorshift(&mut state) % (4 * FIXED_ONE))
            .collect();
        let omega = xorshift(&mut state) % (FIXED_ONE + 1); // [0, 1.0]

        let got = matroid_solve_step_fixed_via(&dispatcher, &a, &b, &x_in, omega, n as u32)
            .expect("matroid_solve_step_fixed_via must dispatch the weighted-Jacobi kernel");
        let want = jacobi_fixed(&a, &b, &x_in, omega, n);
        assert_eq!(
            got, want,
            "case {case}: matroid Jacobi step must match the exact fixed-point oracle; \
             n={n} omega={omega} a={a:?} b={b:?} x_in={x_in:?}"
        );

        for i in 0..n {
            let diag = a[i * n + i];
            if diag == 0 {
                hit_zero_diag += 1;
            } else if diag < FIXED_ONE {
                hit_sub_unit_diag += 1;
            } else {
                hit_int_diag += 1;
            }
        }
        if got.iter().zip(x_in.iter()).any(|(g, x)| g != x) {
            nontrivial += 1;
        }
    }
    assert!(
        nontrivial > 200,
        "expected >200 systems where the step actually moves x, got {nontrivial}"
    );
    // Every diag_units regime must be genuinely exercised, not just the easy one.
    assert!(
        hit_zero_diag > 50 && hit_sub_unit_diag > 50 && hit_int_diag > 50,
        "all three diag regimes must be exercised: zero={hit_zero_diag} sub_unit={hit_sub_unit_diag} int={hit_int_diag}"
    );
}

#[test]
fn matroid_step_via_matches_hand_checked_cases() {
    let dispatcher = ReferenceEvalDispatcher;

    // n=1, A=[1.0], b=[2.0], x_in=[1.0], omega=1.0:
    //   res = 2.0 - fixed_mul(1.0, 1.0) = 1.0; diag_units = 1; delta = fixed_mul(1.0,1.0)/1 = 1.0;
    //   x_out = 1.0 + 1.0 = 2.0.  (Standard Jacobi: 1 + 1*(2 - 1*1)/1 = 2.)
    let got = matroid_solve_step_fixed_via(
        &dispatcher,
        &[FIXED_ONE],
        &[2 * FIXED_ONE],
        &[FIXED_ONE],
        FIXED_ONE,
        1,
    )
    .unwrap();
    assert_eq!(
        got,
        vec![2 * FIXED_ONE],
        "unit system relaxes x from 1.0 to 2.0"
    );

    // omega = 0 → delta = 0 → x unchanged regardless of residual.
    let got = matroid_solve_step_fixed_via(
        &dispatcher,
        &[3 * FIXED_ONE, FIXED_ONE / 4, FIXED_ONE / 4, 3 * FIXED_ONE],
        &[5 * FIXED_ONE, 7 * FIXED_ONE],
        &[FIXED_ONE, 2 * FIXED_ONE],
        0,
        2,
    )
    .unwrap();
    assert_eq!(
        got,
        vec![FIXED_ONE, 2 * FIXED_ONE],
        "zero relaxation weight leaves x untouched"
    );

    // diag == 0 guard: A=[0], b=[1.0], x_in=[0], omega=1.0:
    //   res = 1.0 - fixed_mul(0,0) = 1.0; diag_safe = 1; diag_units = 1; delta = fixed_mul(1.0,1.0)/1 = 1.0;
    //   x_out = 0 + 1.0 = 1.0.  Documents that a zero diagonal is treated as a unit scale, not a divide-by-zero.
    let got =
        matroid_solve_step_fixed_via(&dispatcher, &[0], &[FIXED_ONE], &[0], FIXED_ONE, 1).unwrap();
    assert_eq!(
        got,
        vec![FIXED_ONE],
        "zero diagonal is guarded to a unit scale (no divide-by-zero)"
    );

    // Sub-unit diagonal: A=[0.5], b=[1.0], x_in=[0], omega=1.0:
    //   res = 1.0 - fixed_mul(0.5, 0) = 1.0; diag_safe = 0.5 < 1.0 → diag_units = 1;
    //   delta = fixed_mul(1.0, 1.0) / 1 = 1.0; x_out = 0 + 1.0 = 1.0.
    let got = matroid_solve_step_fixed_via(
        &dispatcher,
        &[FIXED_ONE / 2],
        &[FIXED_ONE],
        &[0],
        FIXED_ONE,
        1,
    )
    .unwrap();
    assert_eq!(
        got,
        vec![FIXED_ONE],
        "a sub-unit diagonal rounds its integer scale up to 1"
    );
}
