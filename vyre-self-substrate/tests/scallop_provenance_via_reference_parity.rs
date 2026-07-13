//! End-to-end parity for `data::scallop_provenance::provenance_closure_via` (the transitive lineage
//! closure) through the shared faithful [`common::ReferenceEvalDispatcher`].
//!
//! Closes a mock-dispatcher-coherence gap (see BACKLOG `SWEEP-self-substrate-mock-dispatcher-coherence`):
//! `scallop_join`'s IR is run by NO `vyre-primitives/tests/*` file through a faithful dispatch
//! boundary; the consumer's only coverage is its own in-file dispatcher. This is the FIRST-EVER
//! execution of the provenance-closure kernel through a boundary that models the real backend.
//!
//! Contract (audited CLEAN): `scallop_join` binds state RW(0) + next RW(1) + changed RW(2) +
//! join_rules RO(3) = 4 IC; the consumer seeds state, zero-fills next/changed, supplies join_rules,
//! and decodes outputs[0]=state (the converged closure). Unlike bellman's data-dependent
//! `next[dst[t]]` scatter (which stalls at one round through reference_eval, see
//! `BUG-reference-eval-indirect-scatter-fixpoint-1round`), scallop_join writes each lane's OWN cell
//! `next[i*n+j]` (STATIC index) under grid-sync, the static-scatter form that converges correctly
//! across iterations through reference_eval (so the full multi-iteration closure is validated here).
//! Values are exact bitset unions → BIT-EXACT (no tolerance) vs `reference_provenance_closure`.
#![cfg(feature = "cpu-parity")]

use vyre_self_substrate::data::scallop_provenance::{
    provenance_closure_via, reference_provenance_closure,
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
fn provenance_closure_via_matches_cpu_ref_over_random_lineage_graphs() {
    let dispatcher = ReferenceEvalDispatcher;
    let mut rng = 0x5CA1_10_01u32;
    let mut grew_beyond_seed = 0u32;
    for case in 0..300u32 {
        let n = 2 + (case % 4); // 2..5
        let cells = (n * n) as usize;
        // Clause bitsets are small (low 4 bits) so unions are easy to eyeball and the closure has a
        // real chance to grow transitively without saturating instantly.
        let state: Vec<u32> = (0..cells).map(|_| xorshift(&mut rng) & 0xF).collect();
        let join_rules: Vec<u32> = (0..cells).map(|_| xorshift(&mut rng) & 0xF).collect();
        let max_iterations = n + 2; // >= n guarantees the transitive closure converges

        let got = provenance_closure_via(&dispatcher, &state, &join_rules, n, max_iterations)
            .expect("provenance_closure_via must dispatch the scallop-join closure");
        let want = reference_provenance_closure(&state, &join_rules, n, max_iterations);
        assert_eq!(
            got, want,
            "case {case}: provenance closure must match cpu_ref; n={n} state={state:?} join_rules={join_rules:?}"
        );
        if got.iter().zip(&state).any(|(g, s)| g != s) {
            grew_beyond_seed += 1;
        }
    }
    assert!(
        grew_beyond_seed > 100,
        "closure sweep must exercise graphs where the lineage actually grows past the seed, got {grew_beyond_seed}"
    );
}

#[test]
fn provenance_closure_via_hand_checked_transitive_chain() {
    let d = ReferenceEvalDispatcher;
    // 3-node chain of clause bit 0b1: state seeds direct edges 0⇝1 and 1⇝2; the closure must derive
    // 0⇝2 transitively. Layout is row-major state[i*n + j]. Compared against the authoritative ref.
    let n = 3u32;
    let mut state = vec![0u32; 9];
    state[0 * 3 + 1] = 0b1; // 0 ⇝ 1
    state[1 * 3 + 2] = 0b1; // 1 ⇝ 2
    let mut join_rules = vec![0u32; 9];
    join_rules[0 * 3 + 1] = 0b1;
    join_rules[1 * 3 + 2] = 0b1;

    let got = provenance_closure_via(&d, &state, &join_rules, n, 5).unwrap();
    let want = reference_provenance_closure(&state, &join_rules, n, 5);
    assert_eq!(got, want, "transitive chain closure must match cpu_ref");
    // The closure must be at least as large as the seed everywhere (monotone growth).
    assert!(
        got.iter().zip(&state).all(|(g, s)| g & s == *s),
        "closure must be a superset of the seed at every cell"
    );
}
