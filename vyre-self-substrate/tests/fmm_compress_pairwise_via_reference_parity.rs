//! End-to-end parity for the COMPOSITE `math::fmm_polyhedral_compress::fmm_compress_pairwise_via`, the
//! full zeroth-order Fast-Multipole compress pipeline P2M → M2L → L2P, through the shared faithful
//! [`common::ReferenceEvalDispatcher`].
//!
//! Closes a mock-dispatcher-coherence gap (see BACKLOG `SWEEP-self-substrate-mock-dispatcher-coherence`):
//! the three stages are each parity-covered in isolation (`fmm_polyhedral_via_reference_parity`), but the
//! COMPOSITION, where P2M's moments feed M2L's translate and M2L's locals feed L2P's gather, all through
//! the same faithful boundary across three chained dispatches, was not. This is the FIRST-EVER execution
//! of the full compress pipeline through a boundary that models the real backend.
//!
//! Contract (audited CLEAN): `fmm_compress_pairwise_via` runs three dispatches on one dispatcher 
//!   (1) `aggregate_to_cells_via` (P2M): scores RO + cell_assignment RO + moments RW = 3 IC → cell_moments;
//!   (2) `translate_to_targets_via` (M2L): cell_moments RO + cell_distances RO + cell_local RW = 3 IC;
//!   (3) `evaluate_at_regions_via` (L2P): cell_local RO + cell_assignment RO + region_out RW = 3 IC → out.
//! Each stage's contract is audited CLEAN in the per-stage suite; the composite oracle chains their
//! confirmed f64 semantics:
//!   moments[c] = Σ_{r: assignment[r]==c} scores[r]
//!   local[t]   = Σ_{s != t} moments[s] / max(distances[t*n_cells + s], 1e-12)
//!   out[r]     = local[assignment[r]]
//! f32 GPU (three chained stages) vs f64 oracle → small numeric TOLERANCE. Distances are kept in [1, 4)
//! so the M2L reciprocal is well-conditioned and rounding stays far below tolerance.
#![cfg(feature = "cpu-parity")]

use vyre_self_substrate::math::fmm_polyhedral_compress::fmm_compress_pairwise_via;

mod common;
use common::ReferenceEvalDispatcher;

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

fn unit_f32(state: &mut u32) -> f32 {
    (xorshift(state) >> 8) as f32 / (1u32 << 24) as f32
}

/// Chained f64 oracle for the P2M → M2L → L2P compress pipeline.
fn compress_oracle(
    scores: &[f32],
    cell_assignment: &[u32],
    cell_distances: &[f32],
    n_cells: usize,
) -> Vec<f64> {
    // P2M: zeroth moment per cell = sum of contained region scores.
    let mut moments = vec![0.0f64; n_cells];
    for (r, &c) in cell_assignment.iter().enumerate() {
        moments[c as usize] += f64::from(scores[r]);
    }
    // M2L: local[t] = Σ_{s != t} moments[s] / max(dist[t*n_cells + s], 1e-12).
    let mut local = vec![0.0f64; n_cells];
    for t in 0..n_cells {
        for s in 0..n_cells {
            if t == s {
                continue;
            }
            let d = f64::from(cell_distances[t * n_cells + s]).max(1e-12);
            local[t] += moments[s] / d;
        }
    }
    // L2P: per-region gather of the owning cell's local moment.
    cell_assignment.iter().map(|&c| local[c as usize]).collect()
}

fn approx_slice(got: &[f32], want: &[f64], ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let diff = (f64::from(g) - w).abs();
        assert!(
            diff <= 1.0e-3 + 2.0e-3 * w.abs(),
            "{ctx}[{i}]: got={g} want={w} diff={diff} exceeds tolerance"
        );
    }
}

/// Build a valid instance: `n_cells` cells, `n_regions >= n_cells` regions. The first `n_cells` regions
/// are assigned to cells `0..n_cells` (guaranteeing `cell_count == n_cells`), the rest random.
fn instance(state: &mut u32, n_cells: usize, n_regions: usize) -> (Vec<f32>, Vec<u32>, Vec<f32>) {
    let scores: Vec<f32> = (0..n_regions).map(|_| unit_f32(state)).collect();
    let mut cell_assignment = vec![0u32; n_regions];
    for c in 0..n_cells {
        cell_assignment[c] = c as u32; // cover every cell so cell_count == n_cells
    }
    for r in n_cells..n_regions {
        cell_assignment[r] = xorshift(state) % n_cells as u32;
    }
    // Distances in [1, 4): well-conditioned reciprocal, no near-zero blow-up.
    let cell_distances: Vec<f32> = (0..n_cells * n_cells)
        .map(|_| 1.0 + 3.0 * unit_f32(state))
        .collect();
    (scores, cell_assignment, cell_distances)
}

#[test]
fn fmm_compress_pairwise_via_matches_chained_f64_oracle() {
    let d = ReferenceEvalDispatcher;
    let mut state = 0xF3_C0_00_01u32;
    let mut multi_region_cell = 0u32;
    for case in 0..300u32 {
        let n_cells = 2 + (case % 4) as usize; // 2..5
        let n_regions = n_cells + (case % 7) as usize; // >= n_cells
        let (scores, cell_assignment, cell_distances) = instance(&mut state, n_cells, n_regions);

        let got = fmm_compress_pairwise_via(
            &d,
            &scores,
            &cell_assignment,
            &cell_distances,
            n_regions as u32,
        )
        .expect("fmm_compress_pairwise_via must dispatch the P2M->M2L->L2P pipeline");
        let want = compress_oracle(&scores, &cell_assignment, &cell_distances, n_cells);
        approx_slice(&got, &want, &format!("case {case} compress"));

        let mut counts = vec![0u32; n_cells];
        for &c in &cell_assignment {
            counts[c as usize] += 1;
        }
        if counts.iter().any(|&c| c >= 2) {
            multi_region_cell += 1;
        }
    }
    assert!(
        multi_region_cell > 150,
        "sweep must aggregate multiple regions into a cell (real P2M sums), got {multi_region_cell}"
    );
}

#[test]
fn fmm_compress_pairwise_via_hand_checked_two_cell() {
    let d = ReferenceEvalDispatcher;
    // 2 cells, 3 regions: region 0 -> cell 0, regions 1,2 -> cell 1.
    //   moments[0] = 0.5;  moments[1] = 0.25 + 0.25 = 0.5
    //   dist = [[_, 2.0], [4.0, _]] (diagonal ignored)
    //   local[0] = moments[1]/2.0 = 0.25 ;  local[1] = moments[0]/4.0 = 0.125
    //   out = [local[0], local[1], local[1]] = [0.25, 0.125, 0.125]
    let scores = [0.5f32, 0.25, 0.25];
    let cell_assignment = [0u32, 1, 1];
    let cell_distances = [0.0f32, 2.0, 4.0, 0.0]; // 2x2, diagonal unused
    let got = fmm_compress_pairwise_via(&d, &scores, &cell_assignment, &cell_distances, 3).unwrap();
    let want = compress_oracle(&scores, &cell_assignment, &cell_distances, 2);
    approx_slice(&got, &want, "hand-checked two-cell");
    // Explicit expected values.
    approx_slice(
        &got,
        &[0.25, 0.125, 0.125],
        "hand-checked two-cell explicit",
    );
}
