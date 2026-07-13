//! End-to-end parity for `math::sinkhorn_dispatch_clustering::sinkhorn_clustering_via`.
//!
//! Closes the tenth mock-dispatcher-coherence family (see BACKLOG
//! `SWEEP-self-substrate-mock-dispatcher-coherence`): the consumer's in-file `SinkhornDispatcher`
//! mock returns HAND-PICKED output bytes and the value tests call `reference_sinkhorn_clustering`
//! directly, so the actual Sinkhorn IR (`sinkhorn_clustering_program`: an entropic-OT dual-scaling
//! iteration with FLOAT division, followed by a per-region arg-min assignment) NEVER ran. This
//! dispatches the real kernel through the shared `ReferenceEvalDispatcher` (real reference-eval of
//! the IR) and asserts the assignment output matches the host `reference_sinkhorn_clustering` oracle
//! on well-separated clusters, where the arg-min is unambiguous (robust to f32 ordering).
//!
//! sinkhorn is ALSO the family that surfaced the OVER-FEED half of the dispatcher-coherence defect:
//! its `out_assignments` buffer (binding 6) is a `BufferDecl::output` (backend-ALLOCATED, consumes no
//! dispatch input), so the program has SIX input-consuming buffers (4 read-only + the u/v dual-scaling
//! scratch), the consumer previously passed SEVEN (a spurious zero slot for the allocated output),
//! which the real backend's strict `validate_input_lengths` would reject. The faithful dispatcher's
//! strict count check caught it; the consumer now passes exactly six.
#![cfg(feature = "cpu-parity")]

use vyre_self_substrate::math::sinkhorn_dispatch_clustering::{
    reference_sinkhorn_clustering, sinkhorn_clustering_via,
};

mod common;
use common::ReferenceEvalDispatcher;

const ITERS: u32 = 12;
const EPS: f32 = 0.1;

/// Run BOTH the IR (through the faithful dispatcher) and the host oracle on the same inputs and
/// assert the assignment vectors are byte-identical, the basis-free correctness contract for a
/// clustering (the IR must reproduce the host arg-min).
fn assert_via_matches_oracle(
    features: &[f32],
    centroids: &[f32],
    weights: &[f32],
    capacities: &[f32],
    m: u32,
    n: u32,
    d: u32,
) -> Vec<u32> {
    let via = sinkhorn_clustering_via(
        &ReferenceEvalDispatcher,
        features,
        centroids,
        weights,
        capacities,
        m,
        n,
        d,
        ITERS,
        EPS,
    )
    .expect("sinkhorn_clustering_via must dispatch the Sinkhorn IR");
    let oracle = reference_sinkhorn_clustering(
        features, centroids, weights, capacities, m, n, d, ITERS, EPS,
    );
    assert_eq!(via.len(), m as usize, "one assignment per region");
    assert_eq!(
        via, oracle,
        "IR assignments must match the host oracle (features={features:?})"
    );
    via
}

#[test]
fn two_distant_regions_assign_to_their_nearest_clusters() {
    // Regions at (0,0) and (10,10); clusters at (0,0) and (10,10). Region 0 → cluster 0, 1 → 1.
    let features = vec![0.0, 0.0, 10.0, 10.0];
    let centroids = vec![0.0, 0.0, 10.0, 10.0];
    let weights = vec![1.0, 1.0];
    let capacities = vec![1.0, 1.0];
    let assignments =
        assert_via_matches_oracle(&features, &centroids, &weights, &capacities, 2, 2, 2);
    assert_eq!(
        assignments,
        vec![0, 1],
        "each region takes its co-located cluster"
    );
}

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

#[test]
fn well_separated_clusters_match_oracle_over_generated_cases() {
    // Each cluster c sits at (SEP*c, 0) with SEP large; every region is placed AT one cluster's
    // location (+ tiny jitter), and capacities are generous (= m) so the entropic-OT competition does
    // not reassign away from the nearest cluster. The arg-min is therefore unambiguous and the IR must
    // reproduce the host oracle exactly.
    const SEP: f32 = 50.0;
    let mut state = 0x5142_9ABCu32;
    let mut nontrivial = 0u32;
    for case in 0..120u32 {
        let n = 2 + (case % 3); // 2..4 clusters
        let m = n + (case % 4); // at least one region per cluster, some clusters get several
        let d = 2u32;

        let mut centroids = Vec::new();
        for c in 0..n {
            centroids.push(SEP * c as f32);
            centroids.push(0.0);
        }
        let mut features = Vec::new();
        let mut expected = Vec::new();
        for i in 0..m {
            let c = i % n;
            // tiny jitter << SEP keeps the nearest cluster unchanged.
            let jitter = (xorshift(&mut state) >> 20) as f32 / (1u32 << 12) as f32 - 0.5;
            features.push(SEP * c as f32 + jitter);
            features.push(jitter);
            expected.push(c);
        }
        let weights = vec![1.0f32; m as usize];
        // Generous capacities: every cluster can hold all regions, so no reassignment pressure.
        let capacities = vec![m as f32; n as usize];

        let assignments =
            assert_via_matches_oracle(&features, &centroids, &weights, &capacities, m, n, d);
        assert_eq!(
            assignments, expected,
            "case {case}: each region assigns to its co-located cluster"
        );
        if m > n {
            nontrivial += 1;
        }
    }
    assert!(
        nontrivial > 40,
        "expected >40 cases with more regions than clusters, got {nontrivial}"
    );
}
