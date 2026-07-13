//! End-to-end parity for `math::tensor_train_chain_fusion::fusion_pressure_via` through the shared
//! faithful [`common::ReferenceEvalDispatcher`].
//!
//! Closes a mock-dispatcher-coherence gap (see BACKLOG `SWEEP-self-substrate-mock-dispatcher-coherence`
//! / `SWEEP-via-consumer-input-output-contract-audit`): `tt_contract_step`'s IR is run by NO
//! `vyre-primitives/tests/*` file and the consumer's only coverage is its own in-file dispatcher, so
//! this is the FIRST-EVER execution of the tensor-train contraction kernel through a dispatch
//! boundary that models the real backend.
//!
//! `fusion_pressure_via` chains `tt_contract_step` (a 16.16 multiply-then-shift contraction, buffers
//! acc_in RO(0) + core_slice RO(1) + acc_out plain-ReadWrite(2) = 3 input-consuming, no
//! backend-allocated output → no over-feed) with unit cores over the shared-buffer ranks. With every
//! core = fixed-point `1.0`, the contraction `acc_out[b] = Σ_a acc_in[a]·1` accumulates the running
//! rank product exactly (integer·2^16 >> 16 = integer, no rounding), so the final scalar is the exact
//! product of the nonzero ranks (an independent mathematical oracle the IR must reproduce).
#![cfg(feature = "cpu-parity")]

use vyre_self_substrate::math::tensor_train_chain_fusion::{
    fusion_pressure_via, should_fuse_chain_via,
};

mod common;
use common::ReferenceEvalDispatcher;

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// The exact fusion pressure: the product of the nonzero ranks (zero ranks carry no dataflow and are
/// skipped by the chain).
fn expected_pressure(ranks: &[u32]) -> f64 {
    ranks
        .iter()
        .copied()
        .filter(|&r| r != 0)
        .map(f64::from)
        .product()
}

#[test]
fn fusion_pressure_via_matches_rank_product_over_generated_chains() {
    let dispatcher = ReferenceEvalDispatcher;
    let mut state = 0x77_3401u32;
    let mut nontrivial = 0u32;
    for case in 0..400u32 {
        // 1..4 links, each rank 1..6 (occasionally 0) → product stays well under u32::MAX.
        let links = 1 + (case as usize % 4);
        let ranks: Vec<u32> = (0..links).map(|_| xorshift(&mut state) % 7).collect(); // 0..6

        let pressure = fusion_pressure_via(&dispatcher, &ranks)
            .expect("fusion_pressure_via must dispatch the tt_contract_step chain");
        let want = expected_pressure(&ranks);
        assert_eq!(
            pressure, want,
            "case {case}: fusion pressure must equal the exact nonzero-rank product; ranks={ranks:?}"
        );
        if ranks.iter().filter(|&&r| r != 0).count() >= 2 {
            nontrivial += 1;
        }
    }
    assert!(
        nontrivial > 200,
        "expected >200 multi-link chains, got {nontrivial}"
    );
}

#[test]
fn fusion_pressure_via_matches_known_chains() {
    let dispatcher = ReferenceEvalDispatcher;
    // [3,4] → 3·4 = 12; [2,3,5] → 30; a zero link is skipped: [4,0,3] → 12.
    assert_eq!(fusion_pressure_via(&dispatcher, &[3, 4]).unwrap(), 12.0);
    assert_eq!(fusion_pressure_via(&dispatcher, &[2, 3, 5]).unwrap(), 30.0);
    assert_eq!(fusion_pressure_via(&dispatcher, &[4, 0, 3]).unwrap(), 12.0);
    // Empty chain has no pressure.
    assert_eq!(fusion_pressure_via(&dispatcher, &[]).unwrap(), 0.0);
}

#[test]
fn should_fuse_chain_via_thresholds_on_geometric_mean_pressure() {
    let dispatcher = ReferenceEvalDispatcher;
    // pressure([4,4]) = 16, geometric-mean-per-link = 16^(1/2) = 4. Use non-boundary thresholds to
    // avoid f64 ln-boundary fragility.
    assert!(
        should_fuse_chain_via(&dispatcher, &[4, 4], 5.0).unwrap(),
        "per-link pressure 4 is below threshold 5 → fuse"
    );
    assert!(
        !should_fuse_chain_via(&dispatcher, &[4, 4], 3.0).unwrap(),
        "per-link pressure 4 exceeds threshold 3 → do not fuse"
    );
}
