//! End-to-end parity for `submodular_cache_eviction::select_retention_set_via`: the greedy
//! submodular cache-retention selector (through the shared faithful [`common::ReferenceEvalDispatcher`]).
//!
//! Closes a mock-dispatcher-coherence gap (see BACKLOG `SWEEP-self-substrate-mock-dispatcher-coherence`):
//! `argmax_of_marginals`'s IR is not run through a faithful dispatch boundary by any
//! `vyre-primitives/tests/*` file, the in-file test dispatcher hand-computes the argmax and ignores
//! `_program`. This is the FIRST-EVER execution of the argmax-of-marginals kernel through a boundary that
//! models the real backend, across the full host-driven greedy loop.
//!
//! Contract (audited CLEAN): `select_retention_set_via` runs a host greedy loop that dispatches
//! `argmax_of_marginals` up to `k` times; each dispatch binds gains RO(0) + picked RO(1) + winner_idx
//! RW(2) + winner_gain RW(3) = 4 IC (the two one-word outputs zero-filled), decodes outputs[0]=winner_idx
//! and outputs[1]=winner_gain, then the host marks `picked[winner]=1` and zeroes `gains[winner]`. Both the
//! GPU path and `select_retention_set` (the reference) run the IDENTICAL greedy loop, so the produced
//! `picked` 0/1 vector must match BIT-FOR-BIT (no tolerance), this pins that the IR argmax (including its
//! tie-break) agrees with `argmax_of_marginals_cpu`.
#![cfg(feature = "cpu-parity")]

use vyre_self_substrate::submodular_cache_eviction::{
    select_retention_set, select_retention_set_via,
};

mod common;
use common::ReferenceEvalDispatcher;

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

#[test]
fn select_retention_set_via_matches_reference_greedy_over_random_gains() {
    let d = ReferenceEvalDispatcher;
    let mut rng = 0x5B_00_00_01u32;
    let mut real_pick = 0u32; // cases that actually retained >= 1 item
    let mut ties_present = 0u32;
    for case in 0..400u32 {
        let n = 1 + (case % 16); // 1..16
        let k = xorshift(&mut rng) % (n + 1); // 0..=n
                                              // Gains include deliberate collisions (small range) so tie-break parity is exercised.
        let gains: Vec<u32> = (0..n).map(|_| xorshift(&mut rng) % 8).collect();

        let mut gains_ref = gains.clone();
        let want = select_retention_set(&mut gains_ref, n, k);
        let mut gains_gpu = gains.clone();
        let got = select_retention_set_via(&d, &mut gains_gpu, n, k)
            .expect("select_retention_set_via must dispatch the greedy argmax loop");

        assert_eq!(
            got, want,
            "case {case}: GPU greedy retention set must match the reference; n={n} k={k} gains={gains:?}"
        );
        // The GPU path zeroes each winner's gain exactly as the reference does.
        assert_eq!(
            gains_gpu, gains_ref,
            "case {case}: the post-selection gains buffer must match the reference's mutation"
        );

        let picks = got.iter().filter(|&&p| p == 1).count() as u32;
        assert!(
            picks <= k,
            "case {case}: never retain more than k items; picked {picks} for k={k}"
        );
        if picks >= 1 {
            real_pick += 1;
        }
        // A duplicate gain value (with >=2 candidates) exercises the argmax tie-break.
        let mut sorted = gains.clone();
        sorted.sort_unstable();
        if sorted.windows(2).any(|w| w[0] == w[1]) {
            ties_present += 1;
        }
    }
    assert!(
        real_pick > 250,
        "sweep must exercise cases that actually retain items, got {real_pick}"
    );
    assert!(
        ties_present > 250,
        "sweep must exercise argmax tie-break (duplicate gains), got {ties_present}"
    );
}

#[test]
fn select_retention_set_via_hand_checked_greedy_order_and_early_stop() {
    let d = ReferenceEvalDispatcher;

    // Distinct gains [5, 9, 1, 7]: greedy keeps the top-3 by gain = indices 1 (9), 3 (7), 0 (5).
    let mut gains = vec![5u32, 9, 1, 7];
    let got = select_retention_set_via(&d, &mut gains, 4, 3).unwrap();
    assert_eq!(
        got,
        vec![1, 1, 0, 1],
        "greedy retains the three highest-gain items"
    );
    assert_eq!(
        gains,
        vec![0, 0, 1, 0],
        "the three kept gains are zeroed, the dropped one remains"
    );

    // k=1 retains exactly the single highest-gain item (the argmax), even amid zero gains.
    let mut gains = vec![0u32, 3, 0];
    let got = select_retention_set_via(&d, &mut gains, 3, 1).unwrap();
    assert_eq!(
        got,
        vec![0, 1, 0],
        "a single retention takes the max-gain item"
    );
    let want = select_retention_set(&mut vec![0u32, 3, 0], 3, 1);
    assert_eq!(got, want, "single-pick argmax matches the reference");

    // The greedy fills exactly k slots even when remaining gains are zero (argmax does not treat a
    // zero gain as NO_WINNER; the loop stops only once all n items are picked). k=n retains everything.
    let mut gains = vec![0u32, 3, 0];
    let got = select_retention_set_via(&d, &mut gains, 3, 3).unwrap();
    let want = select_retention_set(&mut vec![0u32, 3, 0], 3, 3);
    assert_eq!(got, want, "k=n retains all items and matches the reference");
    assert_eq!(
        got,
        vec![1, 1, 1],
        "with k=n every item is retained regardless of gain"
    );

    // k == 0 retains nothing.
    let mut gains = vec![4u32, 2];
    let got = select_retention_set_via(&d, &mut gains, 2, 0).unwrap();
    assert_eq!(got, vec![0, 0], "k=0 retains nothing");
    assert_eq!(gains, vec![4, 2], "k=0 leaves gains untouched");
}
