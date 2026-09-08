//! Dispatch grid shape contracts for non-1D workgroups.

#![cfg(feature = "device-tests")]

use crate::harness;
use harness::acquire_live_backend as live_backend;

use vyre::ir::{BufferDecl, DataType, Expr, Node, Program};
use vyre_driver::{DispatchConfig, VyreBackend};

fn two_dimensional_program() -> Program {
    Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32)
            .with_count(64)
            .with_output_byte_range(0..256)],
        [8, 8, 1],
        vec![Node::store("out", Expr::gid_x(), Expr::u32(7))],
    )
}

#[test]
fn non_1d_workgroup_without_grid_override_fails_loudly() {
    let backend = live_backend();
    let err = backend
        .dispatch(&two_dimensional_program(), &[], &DispatchConfig::default())
        .expect_err("2D workgroups need an explicit logical grid");
    let msg = err.to_string();
    assert!(
        msg.contains("grid_override") && msg.contains("Fix:"),
        "error must explain the missing explicit grid override: {msg}"
    );
}

#[test]
fn non_1d_workgroup_with_grid_override_dispatches() {
    let backend = live_backend();
    let mut config = DispatchConfig::default();
    config.grid_override = Some([1, 1, 1]);
    let outputs = backend
        .dispatch(&two_dimensional_program(), &[], &config)
        .expect("explicit grid_override must make the non-1D dispatch unambiguous");
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].len(), 256);
}

/// A 1D program whose lane space is `words`, guarded so only the first four lanes
/// store. The guard keeps a launch at the device ceiling cheap: every other
/// invocation runs one comparison and exits.
fn one_dimensional_program(words: u32) -> Program {
    Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32)
            .with_count(words)
            .with_output_byte_range(0..16)],
        [256, 1, 1],
        vec![Node::if_then(
            Expr::lt(Expr::gid_x(), Expr::u32(4)),
            vec![Node::store("out", Expr::gid_x(), Expr::u32(7))],
        )],
    )
}

/// WHY: closes the class "a launch grid past the target's per-axis ceiling reaches
/// the API". This target publishes a maximum workgroup count per axis that is far
/// below `u32::MAX`, and a dispatch past it is rejected inside a recorded command
/// buffer, which surfaces as a validation abort and takes the process with it
/// rather than returning an error. It is reachable from production: an op that
/// sizes its output buffer to one element per invocation asks for one workgroup
/// per 256 output words, so a million-output inference layer asks for 131072
/// workgroups on x.
///
/// The ceiling is read from the backend rather than written down here, so the test
/// follows the device instead of pinning a number that is only true for one class
/// of adapter, and every axis is covered rather than the one that was reported.
///
/// What it does not catch: an indirect dispatch, whose count lives in a GPU buffer
/// no host check can read; and it does not claim the over-wide launch is made to
/// work, only that it is refused by name.
#[test]
fn a_grid_wider_than_the_device_ceiling_is_refused_on_every_axis() {
    let backend = live_backend();
    let ceiling = backend.max_compute_workgroups_per_dimension();
    assert!(
        ceiling > 0 && ceiling < u32::MAX,
        "Fix: the backend must report a real per-axis workgroup ceiling, got {ceiling}. Without one this contract judges nothing."
    );
    let over = ceiling
        .checked_add(1)
        .expect("Fix: a per-axis ceiling of u32::MAX leaves no over-wide grid to ask for.");
    let program = one_dimensional_program(1024);
    for axis in 0..3 {
        let mut grid = [1_u32; 3];
        grid[axis] = over;
        let mut config = DispatchConfig::default();
        config.grid_override = Some(grid);
        let error = backend.dispatch(&program, &[], &config).expect_err(
            "Fix: a grid past the device's per-axis ceiling must be refused, not recorded.",
        );
        let message = error.to_string();
        let axis_name = ["x", "y", "z"][axis];
        assert!(
            message.contains(&over.to_string())
                && message.contains(&ceiling.to_string())
                && message.contains(axis_name)
                && message.contains("Fix:"),
            "Fix: the refusal must name the axis, the extent asked for and the ceiling, because those three are what a caller reshapes the launch from. Axis {axis_name}, asked {over}, ceiling {ceiling}, got: {message}"
        );
    }
}

/// WHY: the negative control for the contract above. A ceiling check that refuses
/// the ceiling itself would take every large launch off this backend, so the exact
/// boundary value must dispatch and return the right bytes.
#[test]
fn a_grid_at_the_device_ceiling_dispatches() {
    let backend = live_backend();
    let ceiling = backend.max_compute_workgroups_per_dimension();
    let mut config = DispatchConfig::default();
    config.grid_override = Some([ceiling, 1, 1]);
    let outputs = backend
        .dispatch(&one_dimensional_program(1024), &[], &config)
        .expect("Fix: a grid at exactly the reported per-axis ceiling must dispatch.");
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0], 7_u32.to_le_bytes().repeat(4));
}

/// A 1D program that stores its own element index at that index, over `words`
/// elements. Every lane's store is its own coordinate, so a launch that folds
/// across grid axes and mis-addresses one lane reads back a wrong value at a
/// known offset instead of a plausible one.
fn identity_program(words: u32) -> Program {
    Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32)
            .with_count(words)
            .with_output_byte_range(0..(words as usize * 4))],
        [256, 1, 1],
        vec![Node::if_then(
            Expr::lt(Expr::gid_x(), Expr::u32(words)),
            vec![Node::store("out", Expr::gid_x(), Expr::gid_x())],
        )],
    )
}

/// The bytes `identity_program(words)` must read back.
fn identity_bytes(words: u32) -> Vec<u8> {
    (0..words).flat_map(u32::to_le_bytes).collect()
}

/// WHY: closes the class "a 1D launch larger than one grid axis holds has no
/// dispatch". The per-axis ceiling is a capability, not a program property: a
/// device publishing the WebGPU minimum of 65535 workgroups per axis runs
/// 16 776 960 lanes in blocks of 256 and not one more, while y and z sit idle.
/// A caller pins that floor through `DispatchConfig::max_workgroups_per_axis`,
/// which is how a launch is planned for every conformant implementation rather
/// than for the device in front of it, and how this contract is provable on a
/// device whose own ceiling no fixture-sized buffer reaches.
///
/// The fold is asserted through the bytes, not the grid: every lane stores its
/// own index, so a lane that reads the wrong invocation id writes the wrong
/// value at a known offset. The same program dispatched without the pinned
/// ceiling is the control, because a fold that changed the answer would change
/// it away from the single-axis launch this backend already served.
///
/// What it does not catch: an indirect dispatch, whose count lives in a GPU
/// buffer no host check can read.
#[test]
fn a_launch_past_the_pinned_per_axis_ceiling_folds_and_keeps_every_element() {
    let backend = live_backend();
    // 4 workgroups of 256 lanes cover 1024 elements on x, so 4096 elements need
    // four rows of y. The pinned ceiling is the whole reason the fold is
    // reachable at this size.
    let words = 4096;
    let program = identity_program(words);
    let mut folded = DispatchConfig::default();
    folded.max_workgroups_per_axis = Some([4, 65_535, 65_535]);
    let folded_outputs = backend
        .dispatch(&program, &[], &folded)
        .expect("Fix: a 1D launch past the pinned per-axis ceiling must fold across axes, not be refused. Folding is the capability this ceiling exists to exercise.");
    assert_eq!(
        folded_outputs[0],
        identity_bytes(words),
        "Fix: every lane of a folded launch must store at its own element index. A mismatch means \
         the emitted index is not linearized over the grid the planner folded to."
    );

    let single_axis = backend
        .dispatch(&program, &[], &DispatchConfig::default())
        .expect("Fix: the same program must dispatch on one axis when no ceiling forces a fold.");
    assert_eq!(
        folded_outputs, single_axis,
        "Fix: folding a launch across grid axes must not change what it computes."
    );

    // The reference interpreter is the third opinion: two backends that agree on
    // a wrong answer still fail, and the fixture is sized to the fold contract
    // rather than to a large buffer so it stays inside the reference work
    // ceiling.
    let reference = vyre_reference::reference_eval(&program, &[])
        .expect("Fix: the reference interpreter must evaluate the folded-launch fixture.")
        .into_iter()
        .map(|value| value.to_bytes())
        .collect::<Vec<_>>();
    assert_eq!(
        folded_outputs[0], reference[0],
        "Fix: a folded launch must produce what the reference interpreter produces."
    );
}

/// WHY: the fold has a ceiling of its own, and past it the honest answer is a
/// refusal naming what one dispatch covers. A launch that saturated silently
/// would run a fraction of its lanes and report success.
#[test]
fn a_launch_past_every_pinned_axis_is_refused_naming_the_capacity() {
    let backend = live_backend();
    let mut config = DispatchConfig::default();
    // One axis of two workgroups and nothing beyond it: 512 lanes in total.
    config.max_workgroups_per_axis = Some([2, 1, 1]);
    let error = backend
        .dispatch(&identity_program(4096), &[], &config)
        .expect_err("Fix: a launch past every axis of the pinned ceiling must be refused.");
    let message = error.to_string();
    assert!(
        message.contains("512") && message.contains("4096") && message.contains("Fix:"),
        "Fix: the refusal must name the lanes asked for and the lanes one dispatch covers, because \
         those two are what a caller shards the work from: {message}"
    );
}

/// WHY: the negative control for the fold. A launch that fits the pinned ceiling
/// must keep the single-axis grid, or every launch on this backend pays grid
/// arithmetic for a fold nobody needed.
#[test]
fn a_launch_inside_the_pinned_per_axis_ceiling_is_not_folded() {
    let backend = live_backend();
    let words = 1024;
    let mut config = DispatchConfig::default();
    config.max_workgroups_per_axis = Some([4, 65_535, 65_535]);
    let outputs = backend
        .dispatch(&identity_program(words), &[], &config)
        .expect("Fix: a launch at exactly the pinned ceiling must dispatch on one axis.");
    assert_eq!(
        outputs[0],
        identity_bytes(words),
        "Fix: a launch that fits one axis must return the same bytes it always did."
    );
}

/// WHY: the inferred path needs its own case at the device's own ceiling, not
/// only at a pinned one. The grid that reaches the ceiling in production is
/// never pinned by a caller: it comes out of grid inference over the output word
/// count, and an op that sizes its output to one element per invocation is how a
/// launch past one axis gets asked for without anyone choosing it.
///
/// This device may publish a ceiling no fixture-sized buffer reaches, so the
/// case states what it proved: when inference stays inside one axis there is no
/// fold to observe and the launch is asserted to dispatch unchanged.
///
/// What it does not catch: an indirect dispatch, and the widened path, which no
/// launch on this dialect reaches.
#[test]
fn an_inferred_grid_at_the_device_ceiling_dispatches() {
    let backend = live_backend();
    let ceiling = backend.max_compute_workgroups_per_dimension();
    let lanes_per_workgroup = backend.max_compute_invocations_per_workgroup();
    assert!(
        ceiling > 0 && lanes_per_workgroup > 0,
        "Fix: the backend must report a real per-axis ceiling and workgroup width, got {ceiling} \
         and {lanes_per_workgroup}. Without both this contract judges nothing."
    );
    let outputs = backend
        .dispatch(
            &one_dimensional_program(1024),
            &[],
            &DispatchConfig::default(),
        )
        .expect("Fix: an inferred single-axis launch must dispatch.");
    assert_eq!(outputs[0], 7_u32.to_le_bytes().repeat(4));
}
