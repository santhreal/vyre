//! `mlp_4x_leaky_sq` walks its hidden and output dimensions in fixed 256-wide
//! strides off the global logical point, with no gate confining that walk to one
//! tile, so above `model_dim = 256` it writes activations computed from
//! workgroup scratch it never filled.
//!
//! Mechanism. The builder binds `lane` to `Expr::LogicalIndex { axis: 0 }`,
//! the schedule-free global logical index. Both projection bodies then index
//! `chunk * MLP_WORKGROUP + lane`, so in workgroup `g` the effective index is
//! `(chunk + g) * 256 + local`. That window is SHIFTED UP by `g * 256`:
//!
//! 1. The hidden projection therefore never writes `HIDDEN_SCRATCH[0 .. g*256)`
//!    in workgroup `g`. `HIDDEN_SCRATCH` is declared `BufferDecl::workgroup`, so
//!    every workgroup owns a PRIVATE copy and the reference interpreter zeroes it
//!    (`vyre-reference` `workgroup.rs` allocates `vec![0; len]`). Group `g > 0`
//!    thus holds zeros where the activations belong.
//! 2. The output projection still reads the FULL `[0, hidden_dim)` range of that
//!    scratch, and for `model_dim > 256` group `g > 0` also passes its own
//!    `i < model_dim` gate and STORES to `output[i]`. Group 0 wrote the correct
//!    value to those same slots first, so the last writer wins and the correct
//!    value is overwritten with one computed from zeros.
//!
//! The grid is not the caller's choice. A `BufferDecl::workgroup` binding makes
//! the program shared, and `vyre-driver` `dispatch_element_count_for_program`
//! (mirrored by the reference interpreter's `force_full_span`) then sizes the
//! dispatch to the WIDEST NON-SHARED binding. Here that is `w1` at
//! `model_dim * hidden_dim`, never the `model_dim` the body actually walks, so a
//! realistic weight matrix alone produces a many-workgroup grid.
//!
//! Threshold. Two conditions must BOTH hold, and the tests below separate them
//! rather than conflating them: the grid must exceed one workgroup, AND
//! `model_dim` must exceed 256 so that group `g > 0` clears its own output gate.
//! At `model_dim <= 256` the grid is still wide but every lane of group `g > 0`
//! fails `i < model_dim` and retires, which is why the 256 case is CORRECT and
//! is kept here as a control.
//!
//! This is NOT the shared-cleared-flag class from santhreal/vyre#2: there is no
//! shared flag, no collective early exit, and no barrier involved. The defect is
//! purely an ungated global lane index.

#![cfg(feature = "nn-activation")]

use vyre_foundation::ir::Program;
use vyre_libs::nn::activation::mlp_4x_leaky_sq;
use vyre_primitives::wire::{decode_f32_le_bytes_all, pack_f32_slice};
use vyre_reference::value::Value;

/// Declared workgroup width of the builder, mirrored here so a change to the
/// constant shows up as a failure in these tests rather than silently moving the
/// threshold they pin.
const MLP_WORKGROUP: u32 = 256;

struct Fixture {
    x: Vec<f32>,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}

/// Build inputs whose every intermediate is exactly representable in `f32`, so
/// the assertions below can compare bit-exact values instead of tolerating an
/// epsilon that could hide a real numeric divergence.
///
/// `x` selects row 0 of `w1` only, `w1[j] = j + 1`, `b1 = 0`, so `h[j] = j + 1`
/// and every `h` is positive, making the leaky term the identity and
/// `act[j] = (j + 1)^2`. `w2` is all ones and `b2 = 0`, so the correct output is
/// the constant `sum over j of (j + 1)^2` in EVERY element.
fn fixture(model_dim: u32, hidden_dim: u32) -> Fixture {
    let md = model_dim as usize;
    let hd = hidden_dim as usize;
    let mut x = vec![0.0_f32; md];
    x[0] = 1.0;
    let mut w1 = vec![0.0_f32; md * hd];
    for j in 0..hd {
        w1[j] = (j + 1) as f32;
    }
    Fixture {
        x,
        w1,
        b1: vec![0.0_f32; hd],
        w2: vec![1.0_f32; hd * md],
        b2: vec![0.0_f32; md],
    }
}

/// The intended math, independent of any lane or workgroup structure.
fn oracle(f: &Fixture, model_dim: u32, hidden_dim: u32) -> Vec<f32> {
    let md = model_dim as usize;
    let hd = hidden_dim as usize;
    let act: Vec<f32> = (0..hd)
        .map(|j| {
            let h = f.b1[j] + (0..md).map(|k| f.x[k] * f.w1[k * hd + j]).sum::<f32>();
            let lk = h.max(0.5 * h);
            lk * lk
        })
        .collect();
    (0..md)
        .map(|i| f.b2[i] + (0..hd).map(|j| act[j] * f.w2[j * md + i]).sum::<f32>())
        .collect()
}

fn build(model_dim: u32, hidden_dim: u32) -> Program {
    mlp_4x_leaky_sq("x", "w1", "b1", "w2", "b2", "out", model_dim, hidden_dim)
        .expect("Fix: mlp_4x_leaky_sq must build for non-zero dimensions")
}

/// Run the builder under the CPU reference oracle and return `(observed, expected)`.
fn run(model_dim: u32, hidden_dim: u32) -> (Vec<f32>, Vec<f32>) {
    let f = fixture(model_dim, hidden_dim);
    let program = build(model_dim, hidden_dim);
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack_f32_slice(&f.x)),
            Value::from(pack_f32_slice(&f.w1)),
            Value::from(pack_f32_slice(&f.b1)),
            Value::from(pack_f32_slice(&f.w2)),
            Value::from(pack_f32_slice(&f.b2)),
        ],
    )
    .expect("Fix: mlp_4x_leaky_sq must reference-evaluate");
    let index = vyre_reference::output_index(&program, "out")
        .expect("Fix: mlp_4x_leaky_sq must declare output `out`");
    let observed = decode_f32_le_bytes_all(&outputs[index].to_bytes());
    (observed, oracle(&f, model_dim, hidden_dim))
}

/// Report the indices where observed and expected differ, so a failure names the
/// exact elements rather than only reporting that two buffers are unequal.
fn divergent_indices(observed: &[f32], expected: &[f32]) -> Vec<usize> {
    observed
        .iter()
        .zip(expected.iter())
        .enumerate()
        .filter(|(_, (got, want))| got != want)
        .map(|(index, _)| index)
        .collect()
}

/// The builder's declared width is the constant these tests pin their threshold
/// to. If it ever changes, the 256 and 257 cases below stop bracketing the
/// boundary and would pass vacuously, so assert it directly.
#[test]
fn builder_declares_the_fixed_two_hundred_fifty_six_lane_workgroup() {
    let program = build(768, 4);
    assert_eq!(
        program.workgroup_size(),
        [MLP_WORKGROUP, 1, 1],
        "Fix: mlp_4x_leaky_sq must declare a [256,1,1] workgroup; these tests pin the 256 threshold against it"
    );
}

/// Control, and the case that separates the two conditions. `model_dim = 256`
/// still produces a MANY-workgroup grid, because the dispatch is sized by `w1` at
/// `model_dim * hidden_dim = 1024`, not by `model_dim`. Every lane of group
/// `g > 0` nonetheless fails the `i < model_dim` gate and retires, so the output
/// is correct. This proves the defect is NOT merely "the grid is wide": a wide
/// grid alone is harmless, and it must stay harmless after the fix.
#[test]
fn model_dim_at_exactly_the_workgroup_width_is_correct_despite_a_multi_workgroup_grid() {
    let (observed, expected) = run(MLP_WORKGROUP, 4);
    assert_eq!(
        observed, expected,
        "Fix: at model_dim == 256 no lane above the workgroup width may store, so the output must match the oracle exactly"
    );
    assert!(
        observed.iter().all(|value| *value == 30.0),
        "Fix: fixture is constructed so every correct element is exactly 30.0, got {observed:?}"
    );
}

/// The threshold, pinned to a single element. At `model_dim = 257` exactly ONE
/// index, 256, is reachable by group 1 (`(0 + 1) * 256 + 0`), and group 1's
/// `HIDDEN_SCRATCH` is entirely zero because the hidden gate `j < hidden_dim`
/// admits only global lanes below `hidden_dim`. So `output[256]` is overwritten
/// with `b2[256] + 0`, which is `0.0`, while the correct value is `30.0`.
///
/// Locks out: the ungated global lane index in `mlp_4x_leaky_sq`. If this
/// regresses, every transformer width above 256 silently returns activations
/// computed from zeroed workgroup scratch, with no crash and no diagnostic.
#[test]
fn model_dim_one_above_the_workgroup_width_must_not_corrupt_the_first_element_above_it() {
    let (observed, expected) = run(MLP_WORKGROUP + 1, 4);
    let divergent = divergent_indices(&observed, &expected);
    assert!(
        divergent.is_empty(),
        "Fix: model_dim=257 must match the oracle, but indices {divergent:?} diverge; \
         observed[256]={:?} expected[256]={:?}. Group 1 stores to output[256] from a \
         HIDDEN_SCRATCH it never wrote.",
        observed.get(256),
        expected.get(256)
    );
    assert_eq!(
        observed[256], 30.0,
        "Fix: output[256] must be the oracle value 30.0, not the zero-scratch value"
    );
}

/// A realistic width. At `model_dim = 768` every index in `[256, 768)` is
/// reachable by at least one group above 0, so two thirds of the output is
/// overwritten with the zero-scratch value while `[0, 256)` stays correct. This
/// asymmetry, correct below the width and wrong above it, is the signature of the
/// bug and is asserted element-wise rather than as a whole-buffer inequality.
///
/// Locks out: the same ungated global lane index at a width people actually run.
/// If this regresses, `mlp_4x_leaky_sq` is wrong for 768, 1024, 4096 and every
/// other real model dimension.
#[test]
fn realistic_model_dim_matches_the_cpu_oracle_for_every_element() {
    let model_dim = 768_u32;
    let (observed, expected) = run(model_dim, 4);
    assert_eq!(
        observed.len(),
        model_dim as usize,
        "Fix: output buffer must hold model_dim elements"
    );
    let divergent = divergent_indices(&observed, &expected);
    assert!(
        divergent.is_empty(),
        "Fix: model_dim=768 must match the oracle for every element, but {} of {} \
         indices diverge, first at {:?} and last at {:?}. observed[256]={:?} \
         expected[256]={:?}, observed[0]={:?} expected[0]={:?}.",
        divergent.len(),
        model_dim,
        divergent.first(),
        divergent.last(),
        observed.get(256),
        expected.get(256),
        observed.first(),
        expected.first()
    );
}

/// Coverage must hold across the whole boundary region, not only at the two
/// hand-picked widths above. Sweeps widths that bracket 256 and 512, including
/// non-multiples, since a stride-256 walk whose grid grows is exactly where
/// coverage starts to shift.
#[test]
fn every_width_across_the_workgroup_boundaries_matches_the_oracle() {
    for model_dim in [1_u32, 2, 255, 256, 257, 258, 300, 511, 512, 513, 700] {
        let (observed, expected) = run(model_dim, 4);
        let divergent = divergent_indices(&observed, &expected);
        assert!(
            divergent.is_empty(),
            "Fix: model_dim={model_dim} must match the oracle, but indices {divergent:?} diverge"
        );
    }
}

/// The hidden dimension is walked by the same strided index, so it needs the same
/// coverage guarantee. Sweeps `hidden_dim` across the width boundary at a fixed
/// realistic `model_dim`, which also exercises a `HIDDEN_SCRATCH` larger than one
/// stride.
#[test]
fn every_hidden_dim_across_the_workgroup_boundary_matches_the_oracle() {
    for hidden_dim in [1_u32, 2, 8, 255, 256, 257, 300] {
        let (observed, expected) = run(300, hidden_dim);
        let divergent = divergent_indices(&observed, &expected);
        assert!(
            divergent.is_empty(),
            "Fix: hidden_dim={hidden_dim} must match the oracle, but indices {divergent:?} diverge"
        );
    }
}

/// Degenerate extents must stay correct rather than being skipped by the gate.
/// `model_dim = 1` with `hidden_dim = 1` is the smallest program the builder
/// accepts and exercises the `div_ceil` trip count at its lower bound.
#[test]
fn smallest_accepted_dimensions_match_the_oracle() {
    let (observed, expected) = run(1, 1);
    assert_eq!(
        observed, expected,
        "Fix: the 1x1 program must still compute the oracle value"
    );
    assert_eq!(
        observed.len(),
        1,
        "Fix: output must hold exactly one element"
    );
    assert_eq!(
        observed[0], 1.0,
        "Fix: with hidden_dim=1 the only activation is (0+1)^2 = 1.0"
    );
}

/// Zero dimensions are rejected at build time, which is the contract that keeps
/// the `div_ceil` trip counts from being zero and silently covering nothing.
#[test]
fn zero_dimensions_are_rejected_rather_than_silently_covering_nothing() {
    assert!(
        mlp_4x_leaky_sq("x", "w1", "b1", "w2", "b2", "out", 0, 4).is_err(),
        "Fix: model_dim=0 must be rejected"
    );
    assert!(
        mlp_4x_leaky_sq("x", "w1", "b1", "w2", "b2", "out", 4, 0).is_err(),
        "Fix: hidden_dim=0 must be rejected"
    );
}
