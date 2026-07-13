//! End-to-end parity for `math::sheaf_heterophilic_dispatch::diffuse_dispatch_stalks_fixed_via`
//! through the shared faithful [`common::ReferenceEvalDispatcher`].
//!
//! Closes a mock-dispatcher-coherence gap (see BACKLOG `SWEEP-self-substrate-mock-dispatcher-coherence`):
//! `sheaf_diffusion_step`'s IR is run by NO `vyre-primitives/tests/*` file and the consumer's only
//! coverage is its own in-file dispatcher, so this is the FIRST-EVER execution of the diagonal
//! sheaf-Laplacian diffusion kernel through a dispatch boundary that models the real backend.
//!
//! `sheaf_diffusion_step` binds stalks RO(0) + restriction_diag RO(1) + damping RO(2) + stalks_next
//! plain-ReadWrite(3) = 4 input-consuming (no backend-allocated output → no over-feed). Per cell `t`
//! it computes (16.16 fixed-point, sheaf.rs):
//!   `stalks_next[t] = stalks[t] - fixed_mul(fixed_mul(damping, restriction_diag[t]), stalks[t])`
//! where `fixed_mul(a,b)` is the SIGNED 16.16 multiply (bits [16..48] of the i64 product, matching the
//! corrected `fixed_mul_16_16_expr`; this generated corpus stays non-negative so the value is unchanged,
//! but the oracle mirrors the signed kernel exactly (see BACKLOG FIXED-amg-fixed-path-unsigned-mul)).
//! That is exactly reproducible in u32, so the oracle here is BIT-EXACT (no tolerance).
#![cfg(feature = "cpu-parity")]

use vyre_self_substrate::math::sheaf_heterophilic_dispatch::diffuse_dispatch_stalks_fixed_via;

mod common;
use common::fixed_mul;
use common::ReferenceEvalDispatcher;

const FIXED_ONE: u32 = 1 << 16;

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// Exact u32 oracle for one diagonal sheaf-diffusion step, mirroring the kernel bit-for-bit.
fn sheaf_step_fixed(stalks: &[u32], restriction: &[u32], damping: u32) -> Vec<u32> {
    stalks
        .iter()
        .zip(restriction.iter())
        .map(|(&s, &r)| {
            let damped_r = fixed_mul(damping, r);
            let delta = fixed_mul(damped_r, s);
            // The IR's `Expr::sub` is u32 wrapping; generated inputs keep damping*r <= 1 so delta<=s
            // (no underflow), but wrapping_sub matches the kernel either way.
            s.wrapping_sub(delta)
        })
        .collect()
}

#[test]
fn diffuse_step_via_matches_exact_fixed_point_oracle_over_generated_systems() {
    let dispatcher = ReferenceEvalDispatcher;
    let mut state = 0x5EAF_0001u32; // arbitrary nonzero seed
    let mut nontrivial = 0u32;
    for case in 0..400u32 {
        let n_nodes = 1 + (case % 5);
        let d = 1 + (case % 4);
        let cells = (n_nodes * d) as usize;

        // stalks in [0.5, 4.0), restriction & damping in [0, 1.0] → damping*restriction <= 1 so the
        // diffusion is contractive (delta <= stalk, no u32 underflow).
        let stalks: Vec<u32> = (0..cells)
            .map(|_| FIXED_ONE / 2 + xorshift(&mut state) % (FIXED_ONE * 4))
            .collect();
        let restriction: Vec<u32> = (0..cells)
            .map(|_| xorshift(&mut state) % (FIXED_ONE + 1))
            .collect();
        let damping = xorshift(&mut state) % (FIXED_ONE + 1);

        let got = diffuse_dispatch_stalks_fixed_via(
            &dispatcher,
            &stalks,
            &restriction,
            damping,
            n_nodes,
            d,
        )
        .expect("diffuse_dispatch_stalks_fixed_via must dispatch the sheaf-diffusion kernel");
        let want = sheaf_step_fixed(&stalks, &restriction, damping);
        assert_eq!(
            got, want,
            "case {case}: sheaf-diffusion step must match the exact fixed-point oracle; \
             n_nodes={n_nodes} d={d} damping={damping}"
        );
        // A case where some cell actually diffuses (delta > 0) exercises the multiply-shift path.
        if want.iter().zip(stalks.iter()).any(|(w, s)| w != s) {
            nontrivial += 1;
        }
    }
    assert!(
        nontrivial > 200,
        "expected >200 systems with real diffusion, got {nontrivial}"
    );
}

#[test]
fn diffuse_step_via_matches_hand_checked_cases() {
    let dispatcher = ReferenceEvalDispatcher;

    // damping = 0 → no diffusion, stalks unchanged.
    let stalks = vec![FIXED_ONE, 2 * FIXED_ONE, 3 * FIXED_ONE];
    let restriction = vec![FIXED_ONE, FIXED_ONE, FIXED_ONE];
    let got =
        diffuse_dispatch_stalks_fixed_via(&dispatcher, &stalks, &restriction, 0, 3, 1).unwrap();
    assert_eq!(got, stalks, "zero damping leaves stalks unchanged");

    // damping = 1.0, restriction = 1.0 → delta = stalk, stalks_next = 0.
    let got =
        diffuse_dispatch_stalks_fixed_via(&dispatcher, &stalks, &restriction, FIXED_ONE, 3, 1)
            .unwrap();
    assert_eq!(
        got,
        vec![0, 0, 0],
        "full damping with unit restriction zeroes stalks"
    );

    // damping = 0.5, restriction = 1.0, stalk = 2.0 → delta = 0.5*1*2 = 1.0 → next = 1.0.
    let got = diffuse_dispatch_stalks_fixed_via(
        &dispatcher,
        &[2 * FIXED_ONE],
        &[FIXED_ONE],
        FIXED_ONE / 2,
        1,
        1,
    )
    .unwrap();
    assert_eq!(
        got,
        vec![FIXED_ONE],
        "half damping halves off a unit-restriction stalk"
    );
}
