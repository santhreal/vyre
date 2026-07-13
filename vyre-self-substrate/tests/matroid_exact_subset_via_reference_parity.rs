//! End-to-end parity for `data::matroid_exact_megakernel::select_optimal_subset_via`, the full
//! Edmonds matroid-intersection megakernel (provably-optimal fusion subset), through the shared
//! faithful [`common::ReferenceEvalDispatcher`].
//!
//! Closes a mock-dispatcher-coherence gap (see BACKLOG `SWEEP-self-substrate-mock-dispatcher-coherence`):
//! `matroid_intersection_full`'s IR is not run through a faithful dispatch boundary by any
//! `vyre-primitives/tests/*` file. This is the FIRST-EVER execution of the megakernel through a boundary
//! that models the real backend.
//!
//! Contract (audited CLEAN): `matroid_intersection_full` binds 12 buffers, exchange_adj RO(0),
//! sources RO(1), sinks RO(2), set_x RW(3, seeded with seed_x), parent/frontier/next_frontier/visited
//! RW(4-7), any_change RW(8), path_out RW(9), path_len RW(10), and an INTERNAL `target_node_buf` RW(11)
//! scratch = 12 IC. The `select_optimal_subset_via` wrapper zero-fills the 8 non-seed RW slots (INCLUDING
//! the internal target_node_buf: 12 fed == 12 declared, no over/under-feed) and decodes outputs[0] =
//! set_x (the max independent set as a 0/1 vector).
//!
//! This is a SINGLE-THREADED sequential Edmonds algorithm with a NON-IDEMPOTENT toggle
//! (`set_x[node] = 1 - set_x[node]`); it would race if run by multiple lanes, but its ENTIRE body is
//! guarded to `InvocationId == 0` so exactly one invocation executes it, correct under ANY grid,
//! including the grid ReferenceEvalDispatcher infers from buffer shapes (which ignores the consumer's
//! `[1,1,1]` request and over-fires `n` lanes; the lane-0 guard makes that harmless). Result is exact
//! (deterministic, lowest-id tie-breaks) → compared BIT-FOR-BIT against `reference_select_optimal_subset`.
//!
//! Running this IR faithfully surfaced (and this suite locks) TWO real defects that the in-crate mock
//! `MatroidDispatcher` (which ignores `_program` and hand-returns a result) could never catch:
//!   1. `BUG-matroid-intersection-full-invalid-ir-V032-duplicate-let`: the IR failed validation
//!      (duplicate sibling `let found_sink`/`sink_node` per augmentation). FIXED in
//!      `matroid_intersection_full` (one `Node::Block` scope per augmentation); this suite validates it.
//!   2. `BUG-matroid-megakernel-static-graph-oscillates-multi-augmentation`: the augmenting-path BFS
//!      reads only the STATIC sources/sinks/exchange_adj (never `set_x`), so across the outer
//!      `max_augmentations` loop it re-found the SAME path and OSCILLATED (period 2), while the
//!      reference halts via seen-state detection. FIXED in `matroid_intersection_full`: because the
//!      static graph makes the path orbit a 2-cycle {s0, s0^P}, the IR now emits a SINGLE augmentation
//!      and reproduces the reference's seen-state / max-cardinality termination directly (keep the
//!      toggled state unless it strictly shrank the set, in which case revert). GPU and reference now
//!      agree at EVERY `max_augmentations`; `via_multi_augmentation_matches_reference` locks the >1 case
//!      and the main sweep exercises varying `max_augmentations` (1..4).
#![cfg(feature = "cpu-parity")]

use vyre_self_substrate::data::matroid_exact_megakernel::{
    reference_select_optimal_subset, select_optimal_subset_via,
};

mod common;
use common::ReferenceEvalDispatcher;

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// A random 0/1 vector of length `len`.
fn bits(state: &mut u32, len: usize) -> Vec<u32> {
    (0..len).map(|_| xorshift(state) & 1).collect()
}

#[test]
fn select_optimal_subset_via_matches_reference_over_random_exchange_graphs() {
    let d = ReferenceEvalDispatcher;
    let mut state = 0x3A_70_00_01u32;
    let mut nonempty_result = 0u32;
    let mut nontrivial_seed = 0u32;
    for case in 0..250u32 {
        let n = 2 + (case % 6) as usize; // 2..7
        let exchange_adj = bits(&mut state, n * n);
        let sources = bits(&mut state, n);
        let sinks = bits(&mut state, n);
        let mut seed_x = bits(&mut state, n);
        // Domain precondition: a `source` is a fresh, not-yet-included item, so it cannot also be in
        // the seed set. A contradictory `source[i]==1 && seed[i]==1` node is out-of-domain (the GPU
        // treats it as a length-0 augmenting path and toggles it OFF, which the sequential reference
        // does not) (enforce the precondition so the sweep stays inside the algorithm's valid domain).
        for i in 0..n {
            if sources[i] != 0 {
                seed_x[i] = 0;
            }
        }
        // Vary the augmentation budget across the sweep. Post-fix the IR converges to the reference's
        // seen-state / max-cardinality result at EVERY budget (the static graph makes >1 a no-op the
        // primitive now handles), so parity must hold for 1..4 (exercising the keep-or-revert path).
        let max_augmentations = 1 + (case % 4);

        let got = select_optimal_subset_via(
            &d,
            &exchange_adj,
            &sources,
            &sinks,
            &seed_x,
            n,
            max_augmentations,
        )
        .expect("select_optimal_subset_via must dispatch the matroid megakernel");
        let want = reference_select_optimal_subset(
            &exchange_adj,
            &sources,
            &sinks,
            &seed_x,
            n,
            max_augmentations,
        )
        .expect("reference must succeed on the same valid-shape inputs");
        assert_eq!(
            got, want,
            "case {case}: GPU matroid subset must match the sequential Edmonds reference; n={n} \
             adj={exchange_adj:?} sources={sources:?} sinks={sinks:?} seed={seed_x:?}"
        );
        // The output is a 0/1 membership vector.
        assert!(
            got.iter().all(|&b| b <= 1),
            "case {case}: the result must be a 0/1 membership vector, got {got:?}"
        );

        if got.iter().any(|&b| b == 1) {
            nonempty_result += 1;
        }
        if seed_x.iter().any(|&b| b == 1) {
            nontrivial_seed += 1;
        }
    }
    assert!(
        nonempty_result > 100,
        "sweep must produce non-empty independent sets, got {nonempty_result}"
    );
    assert!(
        nontrivial_seed > 100,
        "sweep must exercise non-empty seed sets (bootstrap path), got {nontrivial_seed}"
    );
}

#[test]
fn select_optimal_subset_via_hand_checked_empty_and_seeded() {
    let d = ReferenceEvalDispatcher;

    // No exchange edges, no sources/sinks, empty seed: nothing to augment → the empty set.
    let n = 3;
    let got = select_optimal_subset_via(
        &d,
        &vec![0; n * n],
        &[0, 0, 0],
        &[0, 0, 0],
        &[0, 0, 0],
        n,
        5,
    )
    .unwrap();
    let want =
        reference_select_optimal_subset(&vec![0; n * n], &[0, 0, 0], &[0, 0, 0], &[0, 0, 0], n, 5)
            .unwrap();
    assert_eq!(got, want, "no edges / no seed → matches reference");
    assert_eq!(
        got,
        vec![0, 0, 0],
        "with nothing to fuse the optimal set is empty"
    );

    // A pre-seeded independent set with no augmenting structure is preserved by both.
    let seed = vec![1, 0, 1];
    let got = select_optimal_subset_via(&d, &vec![0; n * n], &[0, 0, 0], &[0, 0, 0], &seed, n, 5)
        .unwrap();
    let want =
        reference_select_optimal_subset(&vec![0; n * n], &[0, 0, 0], &[0, 0, 0], &seed, n, 5)
            .unwrap();
    assert_eq!(
        got, want,
        "seeded set with no augmenting path → matches reference"
    );
}

/// PARITY LOCK on the FIX for `BUG-matroid-megakernel-static-graph-oscillates-multi-augmentation`: the
/// augmenting-path BFS reads only the STATIC sources/sinks/exchange_adj, so `set_x`'s orbit under the
/// re-found path P is the 2-cycle {s0, s0^P}. The IR now emits a single augmentation and reproduces the
/// reference's seen-state / max-cardinality termination directly, so GPU == reference at EVERY
/// `max_augmentations`: no more oscillation. This test pins that convergence across budgets 1..5 on a
/// clean length-1 augmenting path AND on a shrink case where the toggle must be reverted.
#[test]
fn via_multi_augmentation_matches_reference() {
    let d = ReferenceEvalDispatcher;
    // n=2 with a clean length-1 augmenting path: source node 0 --exchange--> sink node 1, empty seed.
    let n = 2;
    let exchange_adj = vec![0, 1, 0, 0]; // adj[0*2+1] = 1: edge 0 -> 1
    let sources = [1, 0]; // node 0 is a fresh source
    let sinks = [0, 1]; // node 1 is a sink
    let seed = [0, 0];

    let got1 = select_optimal_subset_via(&d, &exchange_adj, &sources, &sinks, &seed, n, 1).unwrap();
    let ref1 =
        reference_select_optimal_subset(&exchange_adj, &sources, &sinks, &seed, n, 1).unwrap();

    // Single augmentation: the path 0->1 is toggled once → both endpoints enter the set. GPU == ref.
    assert_eq!(
        got1,
        vec![1, 1],
        "one augmentation toggles the length-1 path 0->1 into the set"
    );
    assert_eq!(
        got1, ref1,
        "single-augmentation GPU matches the reference exactly"
    );

    // Every larger budget must yield the SAME converged answer as the reference (which halts on the
    // seen-state cycle keeping the max-cardinality endpoint s0^P=[1,1], since |[1,1]|=2 > |[0,0]|=0).
    for budget in 2..=5u32 {
        let got = select_optimal_subset_via(&d, &exchange_adj, &sources, &sinks, &seed, n, budget)
            .unwrap();
        let want =
            reference_select_optimal_subset(&exchange_adj, &sources, &sinks, &seed, n, budget)
                .unwrap();
        assert_eq!(
            want, ref1,
            "reference is stable across a redundant augmentation at budget {budget} (it halts)"
        );
        assert_eq!(
            got, want,
            "FIXED: static-graph GPU now converges (no oscillation) at budget {budget}; got {got:?} \
             want {want:?}"
        );
        assert_eq!(
            got,
            vec![1, 1],
            "the converged set is the max-cardinality 2-cycle endpoint s0^P=[1,1]"
        );
    }

    // Shrink case: the ONLY augmenting path removes MORE than it adds, so the reference keeps the seed
    // (s0 is the larger-cardinality endpoint) for any budget >= 2, and the IR must REVERT its toggle.
    // Seed [1,1,1] (all selected); the sole path is the length-1 edge 0->2 with node 0 a source and
    // node 2 a sink. Toggling {0,2} takes [1,1,1] -> [0,1,0]: gained 0, lost 2 → shrink → revert to s0.
    let n3 = 3;
    let adj3 = vec![0, 0, 1, 0, 0, 0, 0, 0, 0]; // adj[0*3+2]=1: edge 0 -> 2
    let sources3 = [1, 0, 0];
    let sinks3 = [0, 0, 1];
    let seed3 = [1, 1, 1];
    // Budget 1 toggles unconditionally (reference toggles once): [1,1,1] -> [0,1,0].
    let got_b1 = select_optimal_subset_via(&d, &adj3, &sources3, &sinks3, &seed3, n3, 1).unwrap();
    let ref_b1 = reference_select_optimal_subset(&adj3, &sources3, &sinks3, &seed3, n3, 1).unwrap();
    assert_eq!(
        got_b1, ref_b1,
        "budget-1 GPU matches the reference on the shrink case"
    );
    // Budget >= 2: reference halts on the 2-cycle keeping max cardinality = the seed [1,1,1]; the IR
    // must revert the shrinking toggle to match.
    for budget in 2..=4u32 {
        let got =
            select_optimal_subset_via(&d, &adj3, &sources3, &sinks3, &seed3, n3, budget).unwrap();
        let want =
            reference_select_optimal_subset(&adj3, &sources3, &sinks3, &seed3, n3, budget).unwrap();
        assert_eq!(
            got, want,
            "FIXED shrink case: GPU reverts the cardinality-reducing toggle to match the reference \
             at budget {budget}; got {got:?} want {want:?}"
        );
        assert_eq!(
            got,
            vec![1, 1, 1],
            "the reference keeps the larger-cardinality endpoint (the seed) on the shrink 2-cycle"
        );
    }

    // TIE case (gained == lost, |s0^P| == |s0|): the exact boundary of the IR's `gained < lost` revert
    // condition, a tie must NOT revert, keeping the toggled state s0^P (matching the reference, which
    // keeps `current`=s0^P since count(s0) > count(s0^P) is FALSE on equal cardinalities). n=2, edge
    // 0->1 with source 0 (NOT seeded, so no source∈seed out-of-domain node) and sink 1 (seeded). The
    // path {1,0} toggles sink 1: 1->0 (lost) and source 0: 0->1 (gained) → gained 1 == lost 1, Δ=0.
    let n2 = 2;
    let adj2 = vec![0, 1, 0, 0]; // edge 0 -> 1
    let sources2 = [1, 0];
    let sinks2 = [0, 1];
    let seed2 = [0, 1]; // sink seeded, source not: toggling {0,1} keeps cardinality (1 -> 1)
    for budget in 1..=4u32 {
        let got =
            select_optimal_subset_via(&d, &adj2, &sources2, &sinks2, &seed2, n2, budget).unwrap();
        let want =
            reference_select_optimal_subset(&adj2, &sources2, &sinks2, &seed2, n2, budget).unwrap();
        assert_eq!(
            got, want,
            "TIE case: equal-cardinality 2-cycle must NOT revert (keep s0^P) at budget {budget}; \
             got {got:?} want {want:?}"
        );
        assert_eq!(
            got,
            vec![1, 0],
            "the tie keeps the toggled endpoint s0^P=[1,0] (source added, sink removed, |·|=1)"
        );
    }
}
