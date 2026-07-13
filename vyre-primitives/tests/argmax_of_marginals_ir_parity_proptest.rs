//! GPU-IR vs CPU-ref parity for `math::submodular_greedy::argmax_of_marginals`,
//! the greedy submodular-maximization selection step.
//!
//! Lane 0 walks all candidates and picks the maximum `gains[c]` among the
//! UNPICKED (`picked_mask[c] == 0`), writing `(winner_idx, winner_gain)`; the
//! update uses a STRICT `>` so ties break to the LOWEST index, and an all-picked
//! set yields `(NO_WINNER, 0)`. It is a serial lane-0 kernel, so the result is
//! deterministic and `reference_eval` (which fires n_candidates lanes but only
//! lane 0 writes) models it exactly. Every shipped test drives
//! `argmax_of_marginals_cpu` or checks Program shape; the actual argmax IR (the
//! picked-mask exclusion, the `best_idx == NO_WINNER || g > best_gain` update,
//! the lowest-index tie rule, the all-picked sentinel) was never executed. A `>=`
//! vs `>` tie error, a missing exclusion, or a wrong sentinel all diverge here.
#![forbid(unsafe_code)]
#![cfg(all(feature = "math", feature = "cpu-parity"))]

use proptest::prelude::*;
use vyre_primitives::math::submodular_greedy::{
    argmax_of_marginals, argmax_of_marginals_cpu, NO_WINNER,
};
use vyre_primitives::wire::{decode_u32_le_bytes_all as unpack, pack_u32_slice as pack};
use vyre_reference::value::Value;

/// Drive the real IR and return `(winner_idx, winner_gain)`. Buffer binding
/// order: gains(0), picked_mask(1), winner_idx(2, RW), winner_gain(3, RW).
fn gpu_argmax(gains: &[u32], picked_mask: &[u32]) -> (u32, u32) {
    let program = argmax_of_marginals(
        "gains",
        "picked_mask",
        "winner_idx",
        "winner_gain",
        gains.len() as u32,
    );
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack(gains)),
            Value::from(pack(picked_mask)),
            Value::from(pack(&[0u32])),
            Value::from(pack(&[0u32])),
        ],
    )
    .expect("argmax_of_marginals reference evaluation must succeed");
    (
        unpack(&outputs[0].to_bytes())[0],
        unpack(&outputs[1].to_bytes())[0],
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn ir_matches_cpu_ref_over_random_candidates(
        // Small gain range so ties are frequent (exercises the lowest-index rule).
        pairs in proptest::collection::vec((0u32..8, 0u32..2), 1..400)
    ) {
        let gains: Vec<u32> = pairs.iter().map(|&(g, _)| g).collect();
        let picked_mask: Vec<u32> = pairs.iter().map(|&(_, p)| p).collect();
        let expected = argmax_of_marginals_cpu(&gains, &picked_mask);
        let got = gpu_argmax(&gains, &picked_mask);
        prop_assert_eq!(got, expected, "argmax IR diverged: gains={:?}, picked={:?}", gains, picked_mask);
    }
}

/// Deterministic anchors: the lowest-index tie rule, the picked-mask exclusion,
/// the all-picked sentinel, and a zero-gain first candidate (must still win over
/// no candidate).
#[test]
fn ir_matches_cpu_ref_on_boundary_candidates() {
    // Tie at gain 5 between index 1 and index 3 -> lowest index (1) wins.
    let gains = vec![2u32, 5, 4, 5, 1];
    let unpicked = vec![0u32; 5];
    let expected = argmax_of_marginals_cpu(&gains, &unpicked);
    assert_eq!(expected, (1, 5), "cpu_ref: lowest index wins the tie");
    assert_eq!(
        gpu_argmax(&gains, &unpicked),
        expected,
        "IR tie rule must match"
    );

    // Exclusion: the true max (index 1, gain 9) is picked -> next best unpicked
    // (index 4, gain 6) wins.
    let gains = vec![2u32, 9, 4, 5, 6];
    let mask = vec![0u32, 1, 0, 0, 0];
    let expected = argmax_of_marginals_cpu(&gains, &mask);
    assert_eq!(expected, (4, 6), "cpu_ref: picked max excluded");
    assert_eq!(
        gpu_argmax(&gains, &mask),
        expected,
        "IR exclusion must match"
    );

    // All picked -> NO_WINNER sentinel, gain 0.
    let all_picked = vec![1u32; 5];
    let expected = argmax_of_marginals_cpu(&gains, &all_picked);
    assert_eq!(expected, (NO_WINNER, 0), "cpu_ref: all picked -> sentinel");
    assert_eq!(
        gpu_argmax(&gains, &all_picked),
        expected,
        "IR sentinel must match"
    );

    // First unpicked candidate has gain 0: it must still be selected (best_idx
    // starts as NO_WINNER, so the `== NO_WINNER` clause fires before any `>`).
    let gains = vec![0u32, 0, 0];
    let unpicked = vec![0u32; 3];
    let expected = argmax_of_marginals_cpu(&gains, &unpicked);
    assert_eq!(
        expected,
        (0, 0),
        "cpu_ref: zero-gain first candidate still wins"
    );
    assert_eq!(
        gpu_argmax(&gains, &unpicked),
        expected,
        "IR zero-gain selection must match"
    );
}
