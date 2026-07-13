//! End-to-end parity for `scheduling::planar_rewrite_pass_scheduler::schedule_disjoint_rewrites_via`
//! through the shared faithful [`common::ReferenceEvalDispatcher`].
//!
//! Closes a mock-dispatcher-coherence gap (see BACKLOG `SWEEP-self-substrate-mock-dispatcher-coherence`):
//! `planar_rewrite_schedule`'s IR is run by NO `vyre-primitives/tests/*` file and the consumer's only
//! coverage is its own in-file dispatcher, so this is the FIRST-EVER execution of the planar
//! disjoint-rewrite scheduler kernel through a dispatch boundary that models the real backend.
//!
//! `planar_rewrite_schedule` binds candidates RO(0) + chosen plain-ReadWrite(1) = 2 input-consuming
//! (no backend-allocated output → no over/under-feed; consumer correctly passes candidates + a
//! zero-filled `chosen` slot). The kernel is a greedy raster-order disjoint selection: a candidate
//! cell is chosen iff no already-chosen cell lies in its k×k above-left footprint, a pure integer
//! computation, so the oracle here is EXACT (0/1 chosen mask, no tolerance).
#![cfg(feature = "cpu-parity")]

use vyre_primitives::parsing::planar_rewrite::reference_planar_rewrite_schedule;
use vyre_self_substrate::scheduling::planar_rewrite_pass_scheduler::schedule_disjoint_rewrites_via;

mod common;
use common::ReferenceEvalDispatcher;

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

#[test]
fn schedule_via_matches_cpu_greedy_disjoint_selection_over_generated_grids() {
    let dispatcher = ReferenceEvalDispatcher;
    let mut state = 0x9A_11_0001u32;
    let mut nontrivial = 0u32;
    for case in 0..400u32 {
        let h = 1 + (case % 6);
        let w = 1 + ((case / 6) % 6);
        let k = 1 + ((case / 36) % 4);
        let cells = (h * w) as usize;

        // candidates as 0/1/2 → exercises the "any nonzero is a candidate" semantics (output is 0/1).
        let candidates: Vec<u32> = (0..cells).map(|_| xorshift(&mut state) % 3).collect();

        let got = schedule_disjoint_rewrites_via(&dispatcher, &candidates, h, w, k)
            .expect("schedule_disjoint_rewrites_via must dispatch the planar-rewrite scheduler");
        let want = reference_planar_rewrite_schedule(&candidates, h, w, k);
        assert_eq!(
            got, want,
            "case {case}: planar rewrite schedule must match the CPU greedy disjoint selection; \
             h={h} w={w} k={k} candidates={candidates:?}"
        );
        // A case where some candidate is rejected by the footprint conflict exercises the real
        // scheduling logic (not just a pass-through of all candidates).
        let candidate_count = candidates.iter().filter(|&&c| c != 0).count();
        let chosen_count = want.iter().filter(|&&c| c != 0).count();
        if chosen_count > 0 && chosen_count < candidate_count {
            nontrivial += 1;
        }
    }
    assert!(
        nontrivial > 100,
        "expected >100 grids with a real footprint conflict, got {nontrivial}"
    );
}

#[test]
fn schedule_via_matches_hand_checked_cases() {
    let dispatcher = ReferenceEvalDispatcher;

    // All candidates, footprint 1 (each cell conflicts only with itself) → every candidate chosen.
    let all = vec![1, 1, 1, 1];
    assert_eq!(
        schedule_disjoint_rewrites_via(&dispatcher, &all, 2, 2, 1).unwrap(),
        vec![1, 1, 1, 1],
        "footprint 1 chooses every candidate"
    );

    // A 1×4 row of candidates with footprint 2: raster order picks cell 0, then cell 1 conflicts
    // (0 is within its 2-wide left footprint), cell 2 picked, cell 3 conflicts → [1,0,1,0].
    let row = vec![1, 1, 1, 1];
    assert_eq!(
        schedule_disjoint_rewrites_via(&dispatcher, &row, 1, 4, 2).unwrap(),
        vec![1, 0, 1, 0],
        "footprint 2 selects every other candidate in a row"
    );

    // No candidates → empty chosen mask.
    assert_eq!(
        schedule_disjoint_rewrites_via(&dispatcher, &[0, 0, 0], 1, 3, 2).unwrap(),
        vec![0, 0, 0],
        "no candidates → nothing chosen"
    );
}
