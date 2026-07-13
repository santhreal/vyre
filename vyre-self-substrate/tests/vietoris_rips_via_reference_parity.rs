//! End-to-end parity for `math::persistent_homology_loop_signature::region_loop_skeleton_fixed_via`.
//!
//! The dispatched kernel is `topology::vietoris_rips_edge_filter`: the Vietoris-Rips 1-skeleton
//! edge mask `edge_mask[i*n+j] = (i < j) AND (dist[i*n+j] <= epsilon) ? 1 : 0`. It had NO
//! IR-execution coverage: `rg -l vietoris_rips vyre-primitives/tests/` = zero files, and its only
//! self-substrate consumer (`region_loop_skeleton_fixed_via`) was exercised solely by a
//! `SkeletonDispatcher` mock that IGNORES the `_program` IR and hand-returns a mask, so the actual
//! edge-filter kernel never ran (the mock-dispatcher-coherence gap; see the SWEEP-self-substrate row
//! in BACKLOG.md).
//!
//! This runs the real `vietoris_rips_edge_filter` Program through the shared `ReferenceEvalDispatcher`
//! and asserts it EXACTLY (no tolerance) reproduces a u32 oracle. The mask is a pure comparison 
//! the upper-triangle predicate `i < j` and the unsigned 16.16 threshold `dist <= epsilon`, with no
//! arithmetic, so the u32 oracle mirrors the IR bit-for-bit and any divergence is a real defect.
#![forbid(unsafe_code)]

use vyre_self_substrate::math::persistent_homology_loop_signature::region_loop_skeleton_fixed_via;

mod common;
use common::ReferenceEvalDispatcher;

/// Exact u32 replica of the `vietoris_rips_edge_filter` kernel: for the flat row-major `t = i*n + j`
/// cell, emit `1` iff the cell is strictly upper-triangular (`i < j`) AND the fixed-point distance is
/// within the (inclusive) fixed-point threshold. Lower triangle and diagonal are always `0`.
fn vietoris_rips_edge_mask(dist_fixed: &[u32], epsilon_fixed: u32, n: usize) -> Vec<u32> {
    let mut mask = vec![0u32; n * n];
    for i in 0..n {
        for j in 0..n {
            let t = i * n + j;
            if i < j && dist_fixed[t] <= epsilon_fixed {
                mask[t] = 1;
            }
        }
    }
    mask
}

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

#[test]
fn region_loop_skeleton_fixed_via_matches_exact_edge_mask() {
    let dispatcher = ReferenceEvalDispatcher;
    let mut state = 0x5171_3971u32;
    let mut mixed_cases = 0u32;
    for case in 0..400u32 {
        let n = 3 + xorshift(&mut state) % 6; // 3..=8 points (>=3 upper pairs, so a mix is possible)
        let cells = (n * n) as usize;
        // 16.16 distances in [0, ~16.0); epsilon biased to the middle half [4.0, 12.0) so most cases
        // carry a genuine mix of in-threshold and out-of-threshold pairs (not all-edges/no-edges).
        let dist: Vec<u32> = (0..cells)
            .map(|_| xorshift(&mut state) & 0x000F_FFFF)
            .collect();
        let epsilon = (4u32 << 16) + (xorshift(&mut state) % (8u32 << 16));

        let via = region_loop_skeleton_fixed_via(&dispatcher, &dist, epsilon, n)
            .expect("region_loop_skeleton_fixed_via must dispatch the edge-filter kernel");
        let oracle = vietoris_rips_edge_mask(&dist, epsilon, n as usize);

        // A case is "mixed" when at least one upper-triangular pair is an edge and at least one is
        // not (that is what actually exercises both branches of the select).
        let edges = oracle.iter().filter(|&&m| m == 1).count();
        let upper_pairs = (n as usize * (n as usize - 1)) / 2;
        if edges > 0 && edges < upper_pairs {
            mixed_cases += 1;
        }

        assert_eq!(
            via, oracle,
            "case {case} (n={n}, epsilon={epsilon}): edge mask _via {via:?} != exact oracle \
             {oracle:?} (dist={dist:?})"
        );
    }
    assert!(
        mixed_cases > 250,
        "only {mixed_cases}/400 cases had a mix of in/out-of-threshold pairs, strengthen the \
         distance/epsilon distribution so both select branches are exercised"
    );
}

#[test]
fn region_loop_skeleton_fixed_via_is_upper_triangular_and_boundary_inclusive() {
    // n=3, symmetric 16.16 distances: d(0,1)=1.0, d(0,2)=3.0, d(1,2)=2.0 (diagonal 0, lower mirror).
    // epsilon = 2.0. Upper-triangle edges with dist <= 2.0: (0,1) 1.0<=2 -> 1; (0,2) 3.0<=2 -> 0;
    // (1,2) 2.0<=2 -> 1 (the `<=` boundary is INCLUSIVE). All lower-triangle + diagonal cells are 0
    // regardless of distance.
    let dispatcher = ReferenceEvalDispatcher;
    let one = 1u32 << 16;
    let dist = vec![
        0,
        one,
        3 * one, // row 0: d00, d01=1.0, d02=3.0
        one,
        0,
        2 * one, // row 1: d10=1.0, d11, d12=2.0
        3 * one,
        2 * one,
        0, // row 2: d20=3.0, d21=2.0, d22
    ];
    let epsilon = 2 * one;
    let via = region_loop_skeleton_fixed_via(&dispatcher, &dist, epsilon, 3)
        .expect("region_loop_skeleton_fixed_via must dispatch");
    let oracle = vietoris_rips_edge_mask(&dist, epsilon, 3);
    assert_eq!(
        oracle,
        vec![0, 1, 0, 0, 0, 1, 0, 0, 0],
        "sanity: only (0,1) and (1,2) are edges; (0,2) exceeds epsilon; lower triangle stays 0"
    );
    assert_eq!(
        via, oracle,
        "the dispatched edge-filter kernel must equal the exact upper-triangular threshold mask"
    );
}
