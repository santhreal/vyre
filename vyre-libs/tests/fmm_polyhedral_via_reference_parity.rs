//! End-to-end parity for `math::fmm_polyhedral_compress::{aggregate_to_cells_via,
//! translate_to_targets_via, evaluate_at_regions_via}`, the three zeroth-order Fast-Multipole stages
//! (P2M aggregate, M2L translate, L2P evaluate), through the shared faithful
//! [`vyre_driver_reference::ReferenceEvalDispatcher`].
//!
//! Closes a mock-dispatcher-coherence gap (see BACKLOG `SWEEP-self-substrate-mock-dispatcher-coherence`):
//! the FMM f32 IRs are not run through a faithful dispatch boundary by any `vyre-primitives/tests/*`
//! file. This is the FIRST-EVER execution of the P2M/M2L/L2P kernels through a boundary that models the
//! real backend.
//!
//! Contracts: each stage binds two read-only inputs and one read-write output.
//! Independent f64 expectations come from `vyre-reference`'s canonical sequential witnesses:
//! P2M aggregates scores by cell, M2L translates every non-self cell pair, and L2P gathers
//! each assigned cell local into its region output.
//! f32 GPU vs f64 oracle → comparison uses a small numeric TOLERANCE (as the kfac/natural_gradient/
//! sinkhorn f32 suites do). Inputs are bounded (and M2L distances kept >= 1 so the reciprocal is
//! well-conditioned) so rounding stays far below tolerance while a wrong kernel fails by orders.

use vyre_libs::solvers::fmm_polyhedral_compress::{
    aggregate_to_cells_via, evaluate_at_regions_via, translate_to_targets_via,
};
use vyre_reference::composition_witness::{
    l2p_zeroth_all_witness, m2l_zeroth_all_witness, p2m_zeroth_moment_witness,
};

use vyre_driver_reference::ReferenceEvalDispatcher;
use vyre_test_support::fixed_point::xorshift32 as xorshift;

/// A bounded f32 in [0, 1).
fn unit_f32(state: &mut u32) -> f32 {
    (xorshift(state) >> 8) as f32 / (1u32 << 24) as f32
}

fn approx_slice(got: &[f32], want: &[f64], ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let diff = (f64::from(g) - w).abs();
        assert!(
            diff <= 1.0e-3 + 1.0e-3 * w.abs(),
            "{ctx}[{i}]: got={g} want={w} diff={diff} exceeds tolerance"
        );
    }
}

/// Reference-owned f64 P2M oracle.
fn p2m_oracle(scores: &[f32], cell_assignment: &[u32]) -> Vec<f64> {
    let charges_f64: Vec<f64> = scores.iter().map(|&s| f64::from(s)).collect();
    p2m_zeroth_moment_witness(&charges_f64, cell_assignment)
}

/// Reference-owned f64 M2L oracle.
fn m2l_oracle(moments: &[f32], distances: &[f32]) -> Vec<f64> {
    let moments_f64: Vec<f64> = moments.iter().map(|&value| f64::from(value)).collect();
    let distances_f64: Vec<f64> = distances.iter().map(|&value| f64::from(value)).collect();
    m2l_zeroth_all_witness(&moments_f64, &distances_f64)
}

/// Reference-owned f64 L2P oracle.
fn l2p_oracle(cell_local: &[f32], cell_assignment: &[u32]) -> Vec<f64> {
    let local_f64: Vec<f64> = cell_local.iter().map(|&value| f64::from(value)).collect();
    l2p_zeroth_all_witness(&local_f64, cell_assignment, cell_assignment.len() as u32)
}

#[test]
fn p2m_aggregate_via_matches_inline_f64_oracle() {
    let dispatcher = ReferenceEvalDispatcher;
    let mut state = 0xF3_31_00_01u32;
    let mut multi_region_cell = 0u32;
    for case in 0..300u32 {
        let n_cells = 2 + (case % 4); // 2..5
        let n_regions = (n_cells + (case % 8)) as usize;
        let scores: Vec<f32> = (0..n_regions).map(|_| unit_f32(&mut state)).collect();
        let cell_assignment: Vec<u32> = (0..n_regions)
            .map(|_| xorshift(&mut state) % n_cells)
            .collect();

        let got = aggregate_to_cells_via(&dispatcher, &scores, &cell_assignment)
            .expect("aggregate_to_cells_via must dispatch");
        approx_slice(
            &got,
            &p2m_oracle(&scores, &cell_assignment),
            &format!("case {case} P2M"),
        );

        let mut counts = vec![0u32; n_cells as usize];
        for &c in &cell_assignment {
            counts[c as usize] += 1;
        }
        if counts.iter().any(|&c| c >= 2) {
            multi_region_cell += 1;
        }
    }
    assert!(
        multi_region_cell > 150,
        "P2M sweep must aggregate multiple regions into a cell, got {multi_region_cell}"
    );
}

#[test]
fn m2l_translate_via_matches_inline_f64_oracle() {
    let dispatcher = ReferenceEvalDispatcher;
    let mut state = 0x0312_0001_u32;
    for case in 0..300u32 {
        let n_cells = 2 + (case % 4) as usize; // 2..5
        let moments: Vec<f32> = (0..n_cells).map(|_| unit_f32(&mut state)).collect();
        // Distances in [1, 4) keep the reciprocal well-conditioned (no near-zero blow-up).
        let distances: Vec<f32> = (0..n_cells * n_cells)
            .map(|_| 1.0 + 3.0 * unit_f32(&mut state))
            .collect();

        let got = translate_to_targets_via(&dispatcher, &moments, &distances)
            .expect("translate_to_targets_via must dispatch");
        approx_slice(
            &got,
            &m2l_oracle(&moments, &distances),
            &format!("case {case} M2L"),
        );
    }
}

#[test]
fn l2p_evaluate_via_matches_inline_f64_oracle() {
    let dispatcher = ReferenceEvalDispatcher;
    let mut state = 0x51_2A_00_01u32;
    for case in 0..300u32 {
        let n_cells = 2 + (case % 4); // 2..5
        let n = n_cells + (case % 6); // regions >= cells
        let cell_local: Vec<f32> = (0..n_cells).map(|_| unit_f32(&mut state)).collect();
        let cell_assignment: Vec<u32> = (0..n as usize)
            .map(|_| xorshift(&mut state) % n_cells)
            .collect();

        let got = evaluate_at_regions_via(&dispatcher, &cell_local, &cell_assignment, n)
            .expect("evaluate_at_regions_via must dispatch");
        approx_slice(
            &got,
            &l2p_oracle(&cell_local, &cell_assignment),
            &format!("case {case} L2P"),
        );
    }
}
