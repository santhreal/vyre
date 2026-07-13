//! End-to-end parity for the do-calculus change-impact COMPOSITES
//! `logic::do_calculus_change_impact::{predict_impact_via, predict_impact_observation_form_via}`, through
//! the shared faithful [`common::ReferenceEvalDispatcher`].
//!
//! Closes a mock-dispatcher-coherence gap (see BACKLOG `SWEEP-self-substrate-mock-dispatcher-coherence`):
//! `do_calculus_surgery_via_reference_parity` covers the surgery PRIMITIVES (intervention_delete_incoming,
//! rule2, rule3) in isolation, but NOT these higher-level COMPOSITES that chain graph surgery → GPU
//! transitive `reachability_closure` → host impact-mask projection across multiple dispatches. This is the
//! FIRST-EVER execution of the full change-impact pipelines through a boundary that models the real backend.
//!
//! Contract (audited CLEAN): each `_via` runs the same stages as its importable reference 
//!   predict_impact_via = do_intervention_delete_incoming (surgery) → reachability_closure → impact mask;
//!   predict_impact_observation_form_via = the Rule-2 observation-exchange surgery → closure → mask.
//! All stages are u32 bitset / adjacency operations (no floats), so the produced n-word impact mask must
//! match the reference BIT-FOR-BIT (no tolerance). The `reachability_closure` here is the transitive
//! closure of an adjacency matrix (static-index accumulation), which converges correctly through the
//! faithful boundary (unlike the data-dependent indirect-scatter fixpoints, see
//! `BUG-reference-eval-indirect-scatter-fixpoint-1round`).
#![cfg(feature = "cpu-parity")]

use vyre_self_substrate::do_calculus_change_impact::{
    predict_impact, predict_impact_observation_form, predict_impact_observation_form_via,
    predict_impact_via,
};

mod common;
use common::ReferenceEvalDispatcher;

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// A random DAG-ish 0/1 adjacency: only i->j with i<j edges, so the transitive closure is acyclic and
/// terminates (matches how change-impact graphs are built).
fn random_dag_adj(state: &mut u32, n: usize, density_pct: u32) -> Vec<u32> {
    let mut adj = vec![0u32; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            if xorshift(state) % 100 < density_pct {
                adj[i * n + j] = 1;
            }
        }
    }
    adj
}

fn random_mask(state: &mut u32, n: usize) -> Vec<u32> {
    (0..n).map(|_| xorshift(state) & 1).collect()
}

#[test]
fn predict_impact_via_matches_reference_over_random_dags() {
    let d = ReferenceEvalDispatcher;
    let mut state = 0xD0_CA_00_01u32;
    let mut real_propagation = 0u32; // cases where the impact reached beyond the intervened nodes
    for case in 0..300u32 {
        let n = 2 + (case % 6) as usize; // 2..7
        let adj = random_dag_adj(&mut state, n, 40);
        let mask = random_mask(&mut state, n);
        let n_u = n as u32;

        let got = predict_impact_via(&d, &adj, &mask, n_u)
            .expect("predict_impact_via must dispatch surgery + closure + projection");
        let want = predict_impact(&adj, &mask, n_u);
        assert_eq!(
            got, want,
            "case {case}: GPU change-impact must match the reference; n={n} adj={adj:?} mask={mask:?}"
        );

        let intervened = mask.iter().filter(|&&m| m != 0).count();
        let impacted = got.iter().filter(|&&v| v != 0).count();
        if impacted > intervened {
            real_propagation += 1;
        }
    }
    assert!(
        real_propagation > 80,
        "sweep must exercise impact propagation beyond the intervened set, got {real_propagation}"
    );
}

#[test]
fn predict_impact_observation_form_via_matches_reference_over_random_dags() {
    let d = ReferenceEvalDispatcher;
    let mut state = 0x0B_5E_00_01u32;
    let mut nonempty = 0u32;
    for case in 0..300u32 {
        let n = 2 + (case % 6) as usize; // 2..7
        let adj = random_dag_adj(&mut state, n, 40);
        let observation_mask = random_mask(&mut state, n);
        let n_u = n as u32;

        let got = predict_impact_observation_form_via(&d, &adj, &observation_mask, n_u).expect(
            "predict_impact_observation_form_via must dispatch the observation-form pipeline",
        );
        let want = predict_impact_observation_form(&adj, &observation_mask, n_u);
        assert_eq!(
            got, want,
            "case {case}: GPU observation-form impact must match the reference; n={n} adj={adj:?} mask={observation_mask:?}"
        );
        if got.iter().any(|&v| v != 0) {
            nonempty += 1;
        }
    }
    assert!(
        nonempty > 150,
        "sweep must exercise non-empty observation-form impact, got {nonempty}"
    );
}

#[test]
fn predict_impact_via_hand_checked_chain() {
    let d = ReferenceEvalDispatcher;
    // Chain 0 -> 1 -> 2 -> 3; intervene on node 0 → impact reaches every downstream node.
    let n = 4u32;
    let mut adj = vec![0u32; 16];
    adj[0 * 4 + 1] = 1;
    adj[1 * 4 + 2] = 1;
    adj[2 * 4 + 3] = 1;
    let mask = [1u32, 0, 0, 0];
    let got = predict_impact_via(&d, &adj, &mask, n).unwrap();
    let want = predict_impact(&adj, &mask, n);
    assert_eq!(got, want, "chain impact matches the reference");
    assert_eq!(
        got,
        vec![1, 1, 1, 1],
        "intervening on the chain source impacts the whole chain"
    );

    // Intervene on node 2 → only nodes 2 and 3 (its transitive downstream) are impacted.
    let mask = [0u32, 0, 1, 0];
    let got = predict_impact_via(&d, &adj, &mask, n).unwrap();
    assert_eq!(got, predict_impact(&adj, &mask, n));
    assert_eq!(
        got,
        vec![0, 0, 1, 1],
        "impact from node 2 reaches only 2 and 3"
    );
}
