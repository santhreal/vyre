//! End-to-end parity for `math::fmm_polyhedral_compress::{aggregate_to_cells_via,
//! translate_to_targets_via, evaluate_at_regions_via}`, the three zeroth-order Fast-Multipole stages
//! (P2M aggregate, M2L translate, L2P evaluate), through the shared faithful
//! [`common::ReferenceEvalDispatcher`].
//!
//! Closes a mock-dispatcher-coherence gap (see BACKLOG `SWEEP-self-substrate-mock-dispatcher-coherence`):
//! the FMM f32 IRs are not run through a faithful dispatch boundary by any `vyre-primitives/tests/*`
//! file. This is the FIRST-EVER execution of the P2M/M2L/L2P kernels through a boundary that models the
//! real backend.
//!
//! Contracts (audited CLEAN): each stage binds two RO inputs + one RW output = 3 IC, decode
//! outputs[0]. These are f32 kernels; their in-crate f64 reference oracles are `#[cfg(test)]`-only
//! (not reachable from an integration test), so the oracle is reimplemented INLINE here in f64 from the
//! documented zeroth-order semantics (verified against the module's `#[cfg(test)]` refs):
//!   P2M: `moments[cell] = Σ_{r: assignment[r]==cell} scores[r]`
//!   M2L: `local[t] = Σ_{s != t} moments[s] / max(dist[t*n+s], 1e-12)`  (skips the self-cell)
//!   L2P: `region_out[r] = cell_local[assignment[r]]`  (zeroth-order = pass-through gather)
//! f32 GPU vs f64 oracle → comparison uses a small numeric TOLERANCE (as the kfac/natural_gradient/
//! sinkhorn f32 suites do). Inputs are bounded (and M2L distances kept >= 1 so the reciprocal is
//! well-conditioned) so rounding stays far below tolerance while a wrong kernel fails by orders.
#![cfg(feature = "cpu-parity")]

use vyre_self_substrate::math::fmm_polyhedral_compress::{
    aggregate_to_cells_via, evaluate_at_regions_via, translate_to_targets_via,
};

mod common;
use common::ReferenceEvalDispatcher;

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

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

/// Inline f64 P2M oracle: zeroth moment = sum of contained region scores.
fn p2m_oracle(scores: &[f32], cell_assignment: &[u32]) -> Vec<f64> {
    let n_cells = cell_assignment.iter().copied().max().unwrap_or(0) as usize + 1;
    let mut moments = vec![0.0f64; n_cells];
    for (i, &cell) in cell_assignment.iter().enumerate() {
        moments[cell as usize] += f64::from(scores[i]);
    }
    moments
}

/// Inline f64 M2L oracle: local[t] = Σ_{s != t} moments[s] / max(dist[t*n+s], 1e-12).
fn m2l_oracle(moments: &[f32], distances: &[f32]) -> Vec<f64> {
    let n = moments.len();
    let mut local = vec![0.0f64; n];
    for t in 0..n {
        for s in 0..n {
            if t == s {
                continue;
            }
            let d = f64::from(distances[t * n + s]).max(1e-12);
            local[t] += f64::from(moments[s]) / d;
        }
    }
    local
}

/// Inline f64 L2P oracle: pass-through gather.
fn l2p_oracle(cell_local: &[f32], cell_assignment: &[u32]) -> Vec<f64> {
    cell_assignment
        .iter()
        .map(|&c| f64::from(cell_local[c as usize]))
        .collect()
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
    let mut state = 0x312_00_01u32;
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
        let n = (n_cells + (case % 6)) as u32; // regions >= cells
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
