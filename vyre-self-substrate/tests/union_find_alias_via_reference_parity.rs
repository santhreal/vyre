//! End-to-end parity for `graph::union_find_emit::union_find_alias_via`: the batched lock-free
//! concurrent union-find (alias analysis) (through the shared faithful [`common::ReferenceEvalDispatcher`]).
//!
//! Closes a mock-dispatcher-coherence gap (see BACKLOG `SWEEP-self-substrate-mock-dispatcher-coherence`):
//! `union_find_program`'s IR is not run through a faithful dispatch boundary by any
//! `vyre-primitives/tests/*` file. This is the FIRST-EVER execution of the atomic union-find kernel
//! through a boundary that models the real backend.
//!
//! Contract (audited CLEAN): `union_find_program` binds parent RW(0) (seeded with `parent_init`, mutated
//! in place) + edge_a RO(1) + edge_b RO(2) = 3 IC; the `union_find_alias_via` wrapper owns all input
//! layout/padding and decodes outputs[0] = the post-batch parent vector.
//!
//! Unlike bellman multi-hop / sum-product multi-level (which race through the single-pass boundary, see
//! `BUG-sum-product-multilevel-dag-no-topo-barrier`), union-find is a legitimately CONCURRENT-CORRECT
//! primitive: it coordinates across lanes with `atomic_min` / `atomic_compare_exchange` on `parent`, and
//! `atomic_op_count > 0` forces the reference to a full-span dispatch. The atomic union-by-min is
//! order-independent, so any round-robin interleaving converges to the SAME partition. Roots are compared
//! MODULO `canonicalize_parent_to_roots` (the GPU CAS-min contract agrees with the sequential reference
//! only up to intermediate parent links, not byte-for-byte), so the assertion is a full partition-equality
//! check (exact, no tolerance).
#![cfg(feature = "cpu-parity")]

use vyre_self_substrate::graph::union_find_emit::{
    canonicalize_parent_to_roots, reference_union_find_alias, union_find_alias_via,
};

mod common;
use common::ReferenceEvalDispatcher;

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// Number of distinct components in a root-canonicalized parent vector.
fn component_count(roots: &[u32]) -> usize {
    let mut seen = roots.to_vec();
    seen.sort_unstable();
    seen.dedup();
    seen.len()
}

#[test]
fn union_find_alias_via_matches_reference_partition_over_random_graphs() {
    let dispatcher = ReferenceEvalDispatcher;
    let mut state = 0x0F1_0001u32;
    let mut actually_merged = 0u32; // cases where the batch fused >= 2 seed singletons
    for case in 0..400u32 {
        let n = 2 + (case % 14); // 2..15 nodes
        let parent_init: Vec<u32> = (0..n).collect(); // identity = all singletons
        let n_edges = 1 + (case % 20) as usize;
        let mut edge_a = Vec::with_capacity(n_edges);
        let mut edge_b = Vec::with_capacity(n_edges);
        for _ in 0..n_edges {
            edge_a.push(xorshift(&mut state) % n);
            edge_b.push(xorshift(&mut state) % n);
        }

        let got = union_find_alias_via(&dispatcher, &parent_init, &edge_a, &edge_b)
            .expect("union_find_alias_via must dispatch the atomic union-find");
        let want = reference_union_find_alias(&parent_init, &edge_a, &edge_b);

        let got_roots = canonicalize_parent_to_roots(&got);
        let want_roots = canonicalize_parent_to_roots(&want);
        assert_eq!(
            got_roots, want_roots,
            "case {case}: GPU union-find partition must match the reference; n={n} \
             edge_a={edge_a:?} edge_b={edge_b:?}"
        );

        if component_count(&want_roots) < n as usize {
            actually_merged += 1;
        }
    }
    assert!(
        actually_merged > 250,
        "sweep must exercise graphs where the batch actually merges components, got {actually_merged}"
    );
}

#[test]
fn union_find_alias_via_hand_checked_chain_and_star() {
    let d = ReferenceEvalDispatcher;

    // A 5-node chain 0-1, 1-2, 2-3, 3-4 → one component, min root 0 everywhere.
    let parent_init: Vec<u32> = (0..5).collect();
    let got = union_find_alias_via(&d, &parent_init, &[0, 1, 2, 3], &[1, 2, 3, 4]).unwrap();
    let roots = canonicalize_parent_to_roots(&got);
    assert_eq!(
        roots,
        vec![0, 0, 0, 0, 0],
        "a connected chain collapses to the min root 0"
    );
    assert_eq!(
        roots,
        canonicalize_parent_to_roots(&reference_union_find_alias(
            &parent_init,
            &[0, 1, 2, 3],
            &[1, 2, 3, 4]
        ))
    );

    // Two disjoint pairs {0,2} and {1,3} in a 4-node graph → two components rooted at 0 and 1.
    let parent_init: Vec<u32> = (0..4).collect();
    let got = union_find_alias_via(&d, &parent_init, &[0, 1], &[2, 3]).unwrap();
    let roots = canonicalize_parent_to_roots(&got);
    assert_eq!(
        roots,
        vec![0, 1, 0, 1],
        "disjoint pairs stay in two min-rooted components"
    );
    assert_eq!(component_count(&roots), 2, "exactly two components survive");

    // Self-edges are inert: unioning a node with itself changes nothing.
    let parent_init: Vec<u32> = (0..3).collect();
    let got = union_find_alias_via(&d, &parent_init, &[0, 1, 2], &[0, 1, 2]).unwrap();
    let roots = canonicalize_parent_to_roots(&got);
    assert_eq!(
        roots,
        vec![0, 1, 2],
        "self-edges leave every node its own component"
    );
}
