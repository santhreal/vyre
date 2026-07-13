//! End-to-end parity for `math::bellman_tn_order::bellman_tn_order_via` (the fused Bellman-Ford
//! shortest-path solver) through the shared faithful [`common::ReferenceEvalDispatcher`].
//!
//! Closes a mock-dispatcher-coherence gap (see BACKLOG `SWEEP-self-substrate-mock-dispatcher-coherence`):
//! `bellman_shortest_path`'s IR is run by only ONE `vyre-primitives/tests/*` file
//! (`bellman_oob_edge_parity`, which drives the PRIMITIVE builder directly), and the self-substrate
//! CONSUMER wrapper `bellman_tn_order_via`: the operator-facing entry point that owns the buffer
//! packing (dist/next_dist seeding, `changed=0`, binding-order mapping) and output decode, is
//! covered only by its own in-file dispatcher. This is the first exercise of that consumer path
//! through a dispatch boundary that models the real backend.
//!
//! Contract (audited CLEAN, batch-2): `bellman_shortest_path` binds, in BINDING order, dist RW(0),
//! next_dist RW(1), changed RW(2), src RO(3), dst RO(4), weight RO(5) = 6 input-consuming; the
//! consumer supplies all 6 in binding order and decodes outputs[0]=dist.
//!
//! SINGLE-HOP SCOPE (deliberate): `bellman_shortest_path` fuses the whole relaxation fixpoint into ONE
//! dispatch via `persistent_fixpoint`. `reference_eval` faithfully models a SINGLE relaxation round of
//! that fused loop for this kernel but NOT the multi-iteration convergence of its data-dependent
//! `atomic_min(next[dst[e]], …)` scatter (see BACKLOG `BUG-reference-eval-indirect-scatter-fixpoint-1round`
//!, an isolated reference_eval gap: a STATIC-index scatter `next[t+1]` converges over iterations, but
//! the value-identical data-dependent `next[dst[t]]` stalls at one round). So, exactly as the existing
//! `bellman_oob_edge_parity` test does, and for the same documented reason, every case here is a
//! SINGLE-HOP graph (all edges leave the source), which converges in one round and is therefore
//! insensitive to the multi-round modeling gap. The oracle is `bellman_shortest_path::cpu_ref`, the
//! authoritative CPU reference. Values are exact integers → BIT-EXACT (no tolerance).
#![cfg(feature = "cpu-parity")]

use vyre_primitives::math::bellman_shortest_path::cpu_ref;
use vyre_self_substrate::math::bellman_tn_order::bellman_tn_order_via;
use vyre_self_substrate::optimizer::dispatcher::DispatchError;

mod common;
use common::ReferenceEvalDispatcher;

const INF: u32 = u32::MAX;

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

#[test]
fn bellman_via_matches_cpu_ref_on_single_hop_star_graphs() {
    let dispatcher = ReferenceEvalDispatcher;
    let mut state = 0xB3_11_A0_01u32;
    let mut relaxed_some = 0u32;
    let mut had_unreachable = 0u32;
    let mut had_saturation = 0u32;
    for case in 0..400u32 {
        let n_nodes = 2 + (case % 7); // 2..8
                                      // reference_eval infers its lane grid from buffer shapes; keep n_edges >= n_nodes so the
                                      // inferred grid covers every per-NODE copy lane in persistent_fixpoint's convergence step
                                      // (a graph with far fewer edges than nodes under-fires the reference grid, a modeling
                                      // limitation, see BACKLOG `BUG-reference-eval-indirect-scatter-fixpoint-1round`; matched
                                      // buffer sizes sidestep it, exactly as the primitive `bellman_oob_edge_parity` test does).
        let n_edges = n_nodes as usize + (case % 5) as usize;

        // SINGLE-HOP: every edge leaves the source (node 0). Distances converge in one relaxation
        // round, so the result is independent of the fused loop's multi-round modeling (see the
        // module doc). All endpoints are in range, the consumer boundary-validates and REJECTS
        // out-of-range edges (covered separately by the negative test below), so this positive
        // sweep keeps every endpoint < n_nodes. Duplicate dsts exercise the atomic-min taking the
        // cheaper of several parallel edges.
        let src = vec![0u32; n_edges];
        let mut dst = Vec::with_capacity(n_edges);
        let mut weight = Vec::with_capacity(n_edges);
        for _ in 0..n_edges {
            dst.push(xorshift(&mut state) % n_nodes);
            // Occasionally a huge weight near u32::MAX to drive the saturating-add guard.
            let w = if xorshift(&mut state) % 8 == 0 {
                INF - (xorshift(&mut state) % 16)
            } else {
                xorshift(&mut state) % 5_000
            };
            weight.push(w);
        }

        let mut dist_init = vec![INF; n_nodes as usize];
        dist_init[0] = 0;
        // max_iterations doesn't affect a single-hop graph's converged value; use a small cap.
        let (want, _iters) = cpu_ref(&src, &dst, &weight, &dist_init, n_nodes, 4);
        let got = bellman_tn_order_via(&dispatcher, &src, &dst, &weight, &dist_init, n_nodes, 4)
            .expect("bellman_tn_order_via must dispatch the fused Bellman-Ford kernel");
        assert_eq!(
            got, want,
            "case {case}: consumer path must match cpu_ref on a single-hop graph; \
             n_nodes={n_nodes} src={src:?} dst={dst:?} weight={weight:?}"
        );

        if want.iter().skip(1).any(|&d| d != INF) {
            relaxed_some += 1;
        }
        if want.iter().any(|&d| d == INF) {
            had_unreachable += 1;
        }
        for e in 0..src.len() {
            if dst[e] < n_nodes && weight[e] > INF - 16 {
                had_saturation += 1;
            }
        }
    }
    assert!(
        relaxed_some > 200,
        "expected >200 star graphs that relax at least one node, got {relaxed_some}"
    );
    assert!(
        had_unreachable > 50 && had_saturation > 10,
        "sweep must exercise unreachable (∞) nodes and saturating-add weights: unreachable={had_unreachable} saturating={had_saturation}"
    );
}

#[test]
fn bellman_via_rejects_out_of_range_endpoints() {
    let dispatcher = ReferenceEvalDispatcher;
    // The consumer boundary-validates edge endpoints (unlike the primitive's in-kernel OOB gate):
    // any edge with u >= n_nodes or v >= n_nodes is rejected as BadInputs before dispatch.
    let dist_init = vec![0, INF, INF];
    // OOB source (7 >= 3).
    let err = bellman_tn_order_via(&dispatcher, &[0, 7], &[1, 1], &[5, 1], &dist_init, 3, 4)
        .expect_err("OOB-source edge must be rejected");
    assert!(
        matches!(err, DispatchError::BadInputs(_)),
        "OOB source must be a BadInputs rejection, got {err:?}"
    );
    // OOB dest (9 >= 3).
    let err = bellman_tn_order_via(&dispatcher, &[0, 0], &[1, 9], &[5, 2], &dist_init, 3, 4)
        .expect_err("OOB-dest edge must be rejected");
    assert!(
        matches!(err, DispatchError::BadInputs(_)),
        "OOB dest must be a BadInputs rejection, got {err:?}"
    );
}

#[test]
fn bellman_via_matches_hand_checked_single_hop_cases() {
    let dispatcher = ReferenceEvalDispatcher;

    // Star into a 3-node graph with n_edges>=n_nodes (see the grid-inference note in the sweep):
    // 0→1(3), 0→2(4), 0→1(5). node1=min(3,5)=3, node2=4 → [0, 3, 4].
    let src = vec![0, 0, 0];
    let dst = vec![1, 2, 1];
    let weight = vec![3, 4, 5];
    let dist_init = vec![0, INF, INF];
    let got = bellman_tn_order_via(&dispatcher, &src, &dst, &weight, &dist_init, 3, 4).unwrap();
    assert_eq!(
        got,
        vec![0, 3, 4],
        "each direct edge relaxes its own node (min over parallels)"
    );

    // Two edges into node 1: costs 10 and 3 → atomic-min keeps 3.
    let src = vec![0, 0];
    let dst = vec![1, 1];
    let weight = vec![10, 3];
    let dist_init = vec![0, INF];
    let got = bellman_tn_order_via(&dispatcher, &src, &dst, &weight, &dist_init, 2, 4).unwrap();
    assert_eq!(
        got,
        vec![0, 3],
        "atomic-min keeps the cheaper of parallel edges"
    );

    // A near-INF weight saturates rather than wrapping: 0→1 with w = INF-2 and dist[0]=0 gives
    // dist[1] = 0 + (INF-2) = INF-2 (no overflow); a second parallel edge 0→1 w=5 wins via min.
    let src = vec![0, 0];
    let dst = vec![1, 1];
    let weight = vec![INF - 2, 5];
    let dist_init = vec![0, INF];
    let got = bellman_tn_order_via(&dispatcher, &src, &dst, &weight, &dist_init, 2, 4).unwrap();
    assert_eq!(
        got,
        vec![0, 5],
        "min picks the finite 5 over the near-INF parallel edge"
    );

    // Disconnected node 2 stays ∞. Parallel edges into node 1 keep n_edges>=n_nodes so the
    // reference grid covers node 2's copy lane (which must preserve its ∞).
    let src = vec![0, 0, 0];
    let dst = vec![1, 1, 1];
    let weight = vec![5, 6, 7];
    let dist_init = vec![0, INF, INF];
    let got = bellman_tn_order_via(&dispatcher, &src, &dst, &weight, &dist_init, 3, 4).unwrap();
    assert_eq!(
        got,
        vec![0, 5, INF],
        "node with no incoming edge stays ∞ (node 1 = min 5)"
    );
}
