//! End-to-end parity for `graph::motif::{match_motif_via, motif_matches_via,
//! motif_participation_count_via}`, the CSR subgraph-motif matcher, through the shared faithful
//! [`common::ReferenceEvalDispatcher`].
//!
//! Closes a mock-dispatcher-coherence gap (see BACKLOG `SWEEP-self-substrate-mock-dispatcher-coherence`):
//! the `match_motif` IR is not run through a faithful dispatch boundary by any `vyre-primitives/tests/*`
//! file (the in-file dispatcher hand-computes the witness and ignores `_program`). This is the FIRST-EVER
//! execution of the motif-matching kernel through a boundary that models the real backend.
//!
//! Contract (audited CLEAN): `match_motif`'s IR binds nodes RW(0) + edge_offsets RO(1) + edge_targets RO(2)
//! + edge_kind_mask RO(3) + node_tags RW(4) + motif_hits RW(5) + witness_out RW(6) = 7 IC (the RW slots
//! zero-filled by the wrapper), decoding TWO outputs (motif_hits + witness_out). The importable
//! `match_motif` / `motif_matches` / `motif_participation_count` references (cpu-parity gated) are the
//! authoritative oracles; values are integer witnesses / bool / counts → BIT-EXACT (no tolerance).
#![cfg(feature = "cpu-parity")]

use vyre_primitives::graph::motif::MotifEdge;
use vyre_self_substrate::motif::{
    match_motif, match_motif_via, motif_matches, motif_matches_via, motif_participation_count,
    motif_participation_count_via,
};

mod common;
use common::ReferenceEvalDispatcher;

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// Build a random valid CSR graph: `n` nodes, each with 0..=3 out-edges to random targets carrying a
/// small kind mask. Returns (edge_offsets[n+1], edge_targets, edge_kind_mask).
fn random_csr(state: &mut u32, n: u32) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut offsets = vec![0u32];
    let mut targets = Vec::new();
    let mut masks = Vec::new();
    for _ in 0..n {
        let deg = xorshift(state) % 4; // 0..=3 out-edges
        for _ in 0..deg {
            targets.push(xorshift(state) % n);
            masks.push(1 + xorshift(state) % 3); // kind mask in {1,2,3}
        }
        offsets.push(targets.len() as u32);
    }
    (offsets, targets, masks)
}

/// Build a random small motif pattern (1..=2 edges) over pattern-node ids in `[0, n)`.
fn random_motif(state: &mut u32, n: u32) -> Vec<MotifEdge> {
    let n_edges = 1 + xorshift(state) % 2; // 1..=2 edges
    (0..n_edges)
        .map(|_| MotifEdge {
            from: xorshift(state) % n,
            kind_mask: 1 + xorshift(state) % 3,
            to: xorshift(state) % n,
        })
        .collect()
}

#[test]
fn match_motif_via_matches_reference_over_random_graphs() {
    let d = ReferenceEvalDispatcher;
    let mut state = 0x30_71_00_01u32;
    let mut some_match = 0u32; // cases where the motif actually matched somewhere
    let mut some_miss = 0u32; // cases where it did not
    for case in 0..400u32 {
        let n = 2 + (case % 5); // 2..6
        let (offsets, targets, masks) = random_csr(&mut state, n);
        let motif = random_motif(&mut state, n);

        let got_witness = match_motif_via(&d, n, &offsets, &targets, &masks, &motif)
            .expect("match_motif_via must dispatch the motif matcher");
        let want_witness = match_motif(n, &offsets, &targets, &masks, &motif);
        assert_eq!(
            got_witness, want_witness,
            "case {case}: GPU motif witness must match the reference; n={n} offsets={offsets:?} \
             targets={targets:?} masks={masks:?} motif={motif:?}"
        );

        let got_matches = motif_matches_via(&d, n, &offsets, &targets, &masks, &motif)
            .expect("motif_matches_via must dispatch");
        let want_matches = motif_matches(n, &offsets, &targets, &masks, &motif);
        assert_eq!(
            got_matches, want_matches,
            "case {case}: GPU motif-existence predicate must match the reference"
        );

        let got_count = motif_participation_count_via(&d, n, &offsets, &targets, &masks, &motif)
            .expect("motif_participation_count_via must dispatch");
        let want_count = motif_participation_count(n, &offsets, &targets, &masks, &motif);
        assert_eq!(
            got_count, want_count,
            "case {case}: GPU motif participation count must match the reference"
        );

        // Cross-consistency: the boolean predicate agrees with the participation count being nonzero.
        assert_eq!(
            got_matches,
            got_count > 0,
            "case {case}: motif_matches must equal (participation_count > 0)"
        );

        if want_matches {
            some_match += 1;
        } else {
            some_miss += 1;
        }
    }
    assert!(
        some_match > 50,
        "sweep must exercise graphs where the motif matches, got {some_match}"
    );
    assert!(
        some_miss > 50,
        "sweep must exercise graphs where the motif does NOT match, got {some_miss}"
    );
}

#[test]
fn match_motif_via_hand_checked_chain() {
    let d = ReferenceEvalDispatcher;
    // Chain graph 0 --(1)--> 1 --(1)--> 2. CSR: offsets [0,1,2,2], targets [1,2], masks [1,1].
    let offsets = [0u32, 1, 2, 2];
    let targets = [1u32, 2];
    let masks = [1u32, 1];
    // Motif = the 2-edge chain 0->1->2 with kind 1: this pattern IS present.
    let motif = [
        MotifEdge {
            from: 0,
            kind_mask: 1,
            to: 1,
        },
        MotifEdge {
            from: 1,
            kind_mask: 1,
            to: 2,
        },
    ];
    let got = match_motif_via(&d, 3, &offsets, &targets, &masks, &motif).unwrap();
    let want = match_motif(3, &offsets, &targets, &masks, &motif);
    assert_eq!(got, want, "chain motif witness matches the reference");
    assert!(
        motif_matches_via(&d, 3, &offsets, &targets, &masks, &motif).unwrap(),
        "the chain motif is present in the chain graph"
    );

    // A motif requiring kind 2 on the first edge does NOT match (all graph edges are kind 1).
    let motif_no = [
        MotifEdge {
            from: 0,
            kind_mask: 2,
            to: 1,
        },
        MotifEdge {
            from: 1,
            kind_mask: 1,
            to: 2,
        },
    ];
    let got_no = match_motif_via(&d, 3, &offsets, &targets, &masks, &motif_no).unwrap();
    assert_eq!(
        got_no,
        match_motif(3, &offsets, &targets, &masks, &motif_no)
    );
    assert!(
        !motif_matches_via(&d, 3, &offsets, &targets, &masks, &motif_no).unwrap(),
        "a kind-2 first edge cannot match the kind-1 graph edge"
    );
}
