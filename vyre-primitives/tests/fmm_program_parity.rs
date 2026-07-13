//! GPU-IR parity for the FMM (Fast Multipole Method) zeroth-moment f32 kernels.
//!
//! `math/fmm.rs` ships three `*_zeroth_f32_step` Program builders. P2M
//! (particle→multipole), M2L (multipole→local), L2P (local→particle), but
//! shipped with NO `tests/` file and its inline `#[cfg(test)]` block only
//! exercises the CPU reference helpers (`p2m_zeroth_moment_cpu`, …), never the
//! GPU IR. So the GPU program bodies had ZERO parity coverage (found by the
//! registry-coverage audit, BACKLOG.md WIRING-registry-coverage). This pins each
//! builder's semantics against a hand-computed reference via `reference_eval`,
//! asserting concrete output values (Testing-Contract: never `!is_empty`).
//!
//! Contracts locked here (read directly from the GPU IR in `math/fmm.rs`):
//! - P2M: lane `cell` sums `scores[r]` over every region `r` with
//!   `cell_assignment[r] == cell`; an assignment `>= n_cells` matches no lane
//!   and is silently dropped (only existing cells accumulate).
//! - M2L: lane `target` sums `cell_moments[s] / max(dist[target·n+s], 1e-12)`
//!   over every source `s != target` (self-cell skipped; near-field owns it).
//! - L2P: lane `region` writes `cell_local[cell_assignment[region]]`, GATED by
//!   `cell < n_cells`: an out-of-range assignment SKIPS the write (the lane
//!   keeps its zero-init), which the gate exists to guarantee (no OOB load).
#![cfg(feature = "math")]

use vyre_primitives::math::fmm::{l2p_zeroth_f32_step, m2l_zeroth_f32_step, p2m_zeroth_f32_step};
use vyre_primitives::wire::{
    decode_f32_le_bytes_all as unpack_f32, pack_f32_slice as pack_f32, pack_u32_slice as pack_u32,
};
use vyre_reference::value::Value;

#[track_caller]
fn assert_f32_close(got: &[f32], exp: &[f32], ctx: &str) {
    assert_eq!(
        got.len(),
        exp.len(),
        "{ctx}: length mismatch got={} exp={}",
        got.len(),
        exp.len()
    );
    for (i, (g, e)) in got.iter().zip(exp.iter()).enumerate() {
        assert!(
            (g - e).abs() <= 1e-4,
            "{ctx}: lane {i} got={g} exp={e} (full got={got:?} exp={exp:?})"
        );
    }
}

#[test]
fn p2m_scatters_scores_into_owning_cells() {
    let n_regions = 5u32;
    let n_cells = 3u32;
    let scores = [1.0f32, 2.0, 4.0, 8.0, 16.0];
    let cell_assignment = [0u32, 1, 0, 2, 1];
    // cell 0: scores[0]+scores[2] = 5; cell 1: scores[1]+scores[4] = 18; cell 2: scores[3] = 8.
    let expected = [5.0f32, 18.0, 8.0];

    let program = p2m_zeroth_f32_step("scores", "cells", "moments", n_regions, n_cells);
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack_f32(&scores)),
            Value::from(pack_u32(&cell_assignment)),
            Value::from(pack_f32(&vec![0.0f32; n_cells as usize])),
        ],
    )
    .expect("p2m reference evaluation must succeed");
    assert_f32_close(&unpack_f32(&outputs[0].to_bytes()), &expected, "p2m");
}

#[test]
fn p2m_drops_out_of_range_cell_assignment() {
    // region 4 is assigned to cell 5 (>= n_cells == 3): it matches no lane, so it
    // must be silently dropped (cell 1 gets only scores[1], not scores[1]+scores[4]).
    let n_regions = 5u32;
    let n_cells = 3u32;
    let scores = [1.0f32, 2.0, 4.0, 8.0, 16.0];
    let cell_assignment = [0u32, 1, 0, 2, 5];
    let expected = [5.0f32, 2.0, 8.0];

    let program = p2m_zeroth_f32_step("scores", "cells", "moments", n_regions, n_cells);
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack_f32(&scores)),
            Value::from(pack_u32(&cell_assignment)),
            Value::from(pack_f32(&vec![0.0f32; n_cells as usize])),
        ],
    )
    .expect("p2m reference evaluation must succeed");
    assert_f32_close(
        &unpack_f32(&outputs[0].to_bytes()),
        &expected,
        "p2m out-of-range assignment dropped",
    );
}

#[test]
fn m2l_translates_moments_by_inverse_distance() {
    let n_cells = 3u32;
    let cell_moments = [2.0f32, 4.0, 8.0];
    // row-major target x source; diagonal (self) is skipped so its value is irrelevant.
    let cell_distances = [
        1.0f32, 2.0, 4.0, // target 0: cm[1]/2 + cm[2]/4 = 2 + 2 = 4
        1.0, 1.0, 8.0, // target 1: cm[0]/1 + cm[2]/8 = 2 + 1 = 3
        2.0, 4.0, 1.0, // target 2: cm[0]/2 + cm[1]/4 = 1 + 1 = 2
    ];
    let expected = [4.0f32, 3.0, 2.0];

    let program = m2l_zeroth_f32_step("moments", "dist", "local", n_cells);
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack_f32(&cell_moments)),
            Value::from(pack_f32(&cell_distances)),
            Value::from(pack_f32(&vec![0.0f32; n_cells as usize])),
        ],
    )
    .expect("m2l reference evaluation must succeed");
    assert_f32_close(&unpack_f32(&outputs[0].to_bytes()), &expected, "m2l");
}

#[test]
fn l2p_broadcasts_cell_local_to_assigned_regions() {
    let n_regions = 4u32;
    let n_cells = 3u32;
    let cell_local = [10.0f32, 20.0, 30.0];
    let cell_assignment = [2u32, 0, 1, 0];
    // region_out[r] = cell_local[cell_assignment[r]].
    let expected = [30.0f32, 10.0, 20.0, 10.0];

    let program = l2p_zeroth_f32_step("local", "cells", "out", n_regions, n_cells);
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack_f32(&cell_local)),
            Value::from(pack_u32(&cell_assignment)),
            Value::from(pack_f32(&vec![0.0f32; n_regions as usize])),
        ],
    )
    .expect("l2p reference evaluation must succeed");
    assert_f32_close(&unpack_f32(&outputs[0].to_bytes()), &expected, "l2p");
}

#[test]
fn l2p_skips_out_of_range_cell_assignment() {
    // region 2 is assigned to cell 5 (>= n_cells == 3): the `cell < n_cells` gate
    // SKIPS the write (no OOB load of cell_local), leaving region_out[2] zero-init.
    let n_regions = 4u32;
    let n_cells = 3u32;
    let cell_local = [10.0f32, 20.0, 30.0];
    let cell_assignment = [2u32, 0, 5, 0];
    let expected = [30.0f32, 10.0, 0.0, 10.0];

    let program = l2p_zeroth_f32_step("local", "cells", "out", n_regions, n_cells);
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack_f32(&cell_local)),
            Value::from(pack_u32(&cell_assignment)),
            Value::from(pack_f32(&vec![0.0f32; n_regions as usize])),
        ],
    )
    .expect("l2p reference evaluation must succeed");
    assert_f32_close(
        &unpack_f32(&outputs[0].to_bytes()),
        &expected,
        "l2p out-of-range assignment skipped",
    );
}
