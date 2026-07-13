//! End-to-end parity for `math::differentiable_autotune::pick_config_pre_exp_fixed_via` (the
//! fixed-point softmax normalization step) through the shared faithful [`common::ReferenceEvalDispatcher`].
//!
//! Closes a mock-dispatcher-coherence gap (see BACKLOG `SWEEP-self-substrate-mock-dispatcher-coherence`):
//! `softmax_step`'s IR is not run through a faithful dispatch boundary by any `vyre-primitives/tests/*`
//! file. This is the FIRST-EVER execution of the softmax-normalization kernel through a boundary that
//! models the real backend.
//!
//! Contract (audited CLEAN): `softmax_step` binds pre_exp RO(0) + out RW(1) = 2 IC, decode
//! outputs[0]. The GPU kernel embeds NO exp evaluator, the caller supplies `pre_exp[i]` (already
//! `exp(x[i]-max)` in 16.16 fixed-point), and the kernel only computes a normalization, which is EXACT
//! integer arithmetic (a serial lane-0 body):
//!   `sum = Σ pre_exp[i]` (u32 wrapping); `sum_safe = max(sum, 1)`; `out[j] = (pre_exp[j] << 16) / sum_safe`
//! (a u32 left-shift that keeps the low 32 bits, then integer division). The oracle here replicates
//! that bit-for-bit → BIT-EXACT (no tolerance).
#![cfg(feature = "cpu-parity")]

use vyre_self_substrate::math::differentiable_autotune::pick_config_pre_exp_fixed_via;

mod common;
use common::ReferenceEvalDispatcher;

const FIXED_ONE: u32 = 1 << 16;

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// Exact u32 oracle mirroring `softmax_step` bit-for-bit.
fn softmax_fixed(pre_exp: &[u32]) -> Vec<u32> {
    let sum = pre_exp.iter().fold(0u32, |a, &b| a.wrapping_add(b));
    let sum_safe = if sum == 0 { 1 } else { sum };
    // `p << 16` is a u32 shift keeping the low 32 bits (matches the IR's Expr::shl), then integer div.
    pre_exp.iter().map(|&p| (p << 16) / sum_safe).collect()
}

#[test]
fn softmax_pick_config_via_matches_exact_fixed_point_oracle() {
    let dispatcher = ReferenceEvalDispatcher;
    let mut state = 0x50_F7_00_01u32;
    let mut normalized_wide = 0u32;
    for case in 0..400u32 {
        let n = 1 + (case % 24) as usize;
        // pre_exp[i] = exp(x[i]-max) in 16.16 is naturally in (0, 1.0]; generate in [1, FIXED_ONE].
        // The value FIXED_ONE (==1.0) makes `p << 16` wrap to 0, a real kernel edge the oracle
        // replicates exactly, so it is included rather than avoided.
        let pre_exp: Vec<u32> = (0..n)
            .map(|_| 1 + xorshift(&mut state) % FIXED_ONE)
            .collect();

        let got = pick_config_pre_exp_fixed_via(&dispatcher, &pre_exp)
            .expect("pick_config_pre_exp_fixed_via must dispatch the softmax step");
        let want = softmax_fixed(&pre_exp);
        assert_eq!(
            got, want,
            "case {case}: fixed-point softmax must match the exact oracle; pre_exp={pre_exp:?}"
        );

        // For a multi-candidate distribution with no single dominant term, the normalized weights
        // should spread across candidates (at least two nonzero), exercises real division, not a
        // degenerate one-hot.
        if n >= 2 && got.iter().filter(|&&w| w != 0).count() >= 2 {
            normalized_wide += 1;
        }
    }
    assert!(
        normalized_wide > 200,
        "expected >200 cases with a spread (>=2 nonzero) normalized distribution, got {normalized_wide}"
    );
}

#[test]
fn softmax_pick_config_via_hand_checked_cases() {
    let d = ReferenceEvalDispatcher;

    // Two equal pre-exp weights → each normalizes to 0.5 (== FIXED_ONE/2).
    let got = pick_config_pre_exp_fixed_via(&d, &[FIXED_ONE / 2, FIXED_ONE / 2]).unwrap();
    // sum = FIXED_ONE; out[j] = ((FIXED_ONE/2) << 16) / FIXED_ONE = (FIXED_ONE/2). Both halves 0.5.
    assert_eq!(
        got,
        vec![FIXED_ONE / 2, FIXED_ONE / 2],
        "two equal weights each normalize to 0.5"
    );

    // A 3-way split 1:1:2 → normalized 0.25, 0.25, 0.5 of FIXED_ONE.
    let q = FIXED_ONE / 4;
    let got = pick_config_pre_exp_fixed_via(&d, &[q, q, 2 * q]).unwrap();
    assert_eq!(
        got,
        softmax_fixed(&[q, q, 2 * q]),
        "1:1:2 split matches the exact oracle"
    );
    // Sanity on the ratio: third weight is twice each of the first two.
    assert!(
        got[2] > got[0] && got[0] == got[1],
        "the 2x candidate gets the largest share"
    );

    // Single candidate → normalizes to 1.0 exactly ((p<<16)/p == 1<<16), for p < FIXED_ONE.
    let got = pick_config_pre_exp_fixed_via(&d, &[12345]).unwrap();
    assert_eq!(
        got,
        vec![FIXED_ONE],
        "a lone candidate takes all the mass (1.0)"
    );
}
