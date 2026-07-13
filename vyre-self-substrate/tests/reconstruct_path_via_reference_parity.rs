//! End-to-end parity for `graph::path_reconstruct::reconstruct_path_via`: the parent-pointer path walk 
//! through the shared faithful [`common::ReferenceEvalDispatcher`].
//!
//! Closes a mock-dispatcher-coherence gap (see BACKLOG `SWEEP-self-substrate-mock-dispatcher-coherence`):
//! the `path_reconstruct` IR is not run through a faithful dispatch boundary by any `vyre-primitives/tests/*`
//! file. This is the FIRST-EVER execution of the path-reconstruction kernel through a boundary that models
//! the real backend.
//!
//! Contract (audited CLEAN): the via dispatches the path-reconstruct primitive, decodes the length AND the
//! `max_depth`-word (zero-padded) path buffer into the caller's `scratch`. `cpu_ref` has the IDENTICAL
//! signature and is the authoritative oracle. Values are node indices (integers) → BIT-EXACT on BOTH the
//! returned length AND the full padded path buffer (no tolerance). The walk is bounded by `max_depth`, so
//! even cyclic parent arrays terminate (both the GPU IR and `cpu_ref` bound-walk identically).
#![cfg(feature = "cpu-parity")]

use vyre_primitives::graph::path_reconstruct::cpu_ref;
use vyre_self_substrate::path_reconstruct::reconstruct_path_via;

mod common;
use common::ReferenceEvalDispatcher;

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

#[test]
fn reconstruct_path_via_matches_cpu_ref_over_random_forests() {
    let d = ReferenceEvalDispatcher;
    let mut rng = 0x9A_7C_00_01u32;
    let mut real_walk = 0u32; // cases where the path traversed >= 2 nodes
    let mut hit_root = 0u32; // cases that terminated at a root before max_depth
    for case in 0..400u32 {
        let n = 2 + (case % 12); // 2..13
                                 // Random parent pointers in [0, n): some self-loops (roots), some chains, some cycles, all
                                 // bounded by max_depth so the walk terminates.
        let parent: Vec<u32> = (0..n).map(|_| xorshift(&mut rng) % n).collect();
        let target = xorshift(&mut rng) % n;
        let max_depth = 1 + xorshift(&mut rng) % (n + 2); // 1..=n+2

        let mut got_scratch = Vec::new();
        let got_len = reconstruct_path_via(&d, &parent, target, max_depth, &mut got_scratch)
            .expect("reconstruct_path_via must dispatch the path walk");

        let mut want_scratch = Vec::new();
        let want_len = cpu_ref(&parent, target, max_depth, &mut want_scratch);

        assert_eq!(
            got_len, want_len,
            "case {case}: path length must match cpu_ref; n={n} target={target} max_depth={max_depth} parent={parent:?}"
        );
        assert_eq!(
            got_scratch, want_scratch,
            "case {case}: the full zero-padded path buffer must match cpu_ref; n={n} target={target} max_depth={max_depth} parent={parent:?}"
        );
        // The first written node is always the target itself.
        assert_eq!(
            got_scratch[0], target,
            "case {case}: the walk starts at the target node"
        );

        if want_len >= 2 {
            real_walk += 1;
        }
        if want_len < max_depth {
            hit_root += 1;
        }
    }
    assert!(
        real_walk > 200,
        "sweep must exercise multi-node walks, got {real_walk}"
    );
    assert!(
        hit_root > 100,
        "sweep must exercise walks that terminate at a root before max_depth, got {hit_root}"
    );
}

#[test]
fn reconstruct_path_via_hand_checked_chain_and_root() {
    let d = ReferenceEvalDispatcher;

    // Chain: parent = [0, 0, 1, 2] → node 0 is the root (self-loop). Walk from 3: 3 -> 2 -> 1 -> 0.
    let parent = [0u32, 0, 1, 2];
    let mut scratch = Vec::new();
    let len = reconstruct_path_via(&d, &parent, 3, 8, &mut scratch).unwrap();
    let mut want_scratch = Vec::new();
    let want_len = cpu_ref(&parent, 3, 8, &mut want_scratch);
    assert_eq!(len, want_len);
    assert_eq!(scratch, want_scratch);
    assert_eq!(len, 4, "3 -> 2 -> 1 -> 0 is a 4-node path");
    assert_eq!(&scratch[..4], &[3, 2, 1, 0], "walk order is target-to-root");
    assert!(
        scratch[4..].iter().all(|&v| v == 0),
        "the tail is zero-padded to max_depth"
    );

    // max_depth truncation: walk from 3 with max_depth 2 → only [3, 2], len 2.
    let mut scratch = Vec::new();
    let len = reconstruct_path_via(&d, &parent, 3, 2, &mut scratch).unwrap();
    assert_eq!(len, 2, "max_depth=2 truncates the walk to two nodes");
    assert_eq!(
        scratch,
        vec![3, 2],
        "truncated walk keeps the first two nodes, no padding needed"
    );

    // A lone root: parent[1]==1, walk from 1 → just [1].
    let parent = [0u32, 1, 1];
    let mut scratch = Vec::new();
    let len = reconstruct_path_via(&d, &parent, 1, 4, &mut scratch).unwrap();
    assert_eq!(len, 1, "a root node walks to itself only");
    assert_eq!(scratch, vec![1, 0, 0, 0], "single node then zero padding");
    assert_eq!(len, cpu_ref(&parent, 1, 4, &mut Vec::new()));
}
