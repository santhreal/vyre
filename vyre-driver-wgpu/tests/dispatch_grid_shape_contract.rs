//! Dispatch grid shape contracts for non-1D workgroups.

mod harness;
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

/// WHY: the inferred path needs its own case. The refused grid in production is
/// never pinned by a caller: it comes out of grid inference over the output word
/// count, and an op that pads its output to one element per invocation is how a
/// grid past the ceiling gets asked for without anyone choosing it.
///
/// The workgroup is pinned to the shape the program already declares, because a
/// program that leaves it free is retuned before the grid is inferred: the block
/// tuner reads the output word count and picks the widest block the device
/// accepts, which divides the same lane space into fewer workgroups. With 256
/// lanes pinned, one word past the ceiling's worth of lanes is one workgroup past
/// the ceiling.
///
/// What it does not catch: an indirect dispatch, and the retuned path, which
/// `a_launch_left_unpinned_is_retuned_instead_of_refused` covers from the other
/// side.
#[test]
fn an_inferred_grid_wider_than_the_device_ceiling_is_refused() {
    let backend = live_backend();
    let ceiling = backend.max_compute_workgroups_per_dimension();
    let words = ceiling
        .checked_add(1)
        .and_then(|groups| groups.checked_mul(256))
        .expect("Fix: the ceiling must leave room for one more workgroup of 256 lanes.");
    let program = one_dimensional_program(words);
    let mut config = DispatchConfig::default();
    config.workgroup_override = Some([256, 1, 1]);
    // The refusal is only worth asserting if inference really asks for more than
    // the device allows, and inference reads the output word count rather than the
    // declared element count. Compute the grid this launch resolves to and prove it
    // is over the ceiling before dispatching, so a program that no longer reaches
    // the over-wide case fails here by name instead of passing the dispatch and
    // reading as a missing refusal.
    let word_count = vyre_driver::output_binding_layouts(&program)
        .expect("Fix: the one-dimensional program must have a derivable output layout.")
        .iter()
        .map(|output| output.word_count)
        .max()
        .expect("Fix: the one-dimensional program declares one output binding.");
    let inferred = vyre_driver::infer_dispatch_grid_for_count(
        u32::try_from(word_count).expect("Fix: the declared output word count must fit u32."),
        [256, 1, 1],
    )
    .expect("Fix: a 1D workgroup shape must have an inferable grid.");
    assert!(
        inferred[0] > ceiling,
        "Fix: this case needs an inferred grid past the ceiling. ceiling {ceiling}, declared words \
         {words}, output word count {word_count}, inferred grid {inferred:?}"
    );
    let error = backend.dispatch(&program, &[], &config).expect_err(
        "Fix: a grid inferred past the device's per-axis ceiling must be refused, not recorded.",
    );
    let message = error.to_string();
    assert!(
        message.contains(&ceiling.to_string()) && message.contains("Fix:"),
        "Fix: the inferred-grid refusal must name the ceiling it exceeded: {message}"
    );
}

/// WHY: the negative control for the case above, and the reason the ceiling is
/// hard to reach by accident. A program that declares one output element per lane
/// and leaves the workgroup free is retuned before its grid is inferred: the tuner
/// reads the output word count and widens the block, so the same lane space needs
/// fewer workgroups and the launch stays inside the ceiling. A refusal here would
/// take every large launch off this backend, and a ceiling check placed before the
/// tuning would refuse a launch the device accepts.
///
/// The lane space is one word past what 256-lane blocks could launch, which is the
/// exact program the case above refuses when the block is pinned. The two together
/// say the refusal follows the resolved launch rather than the declared one.
#[test]
fn a_launch_left_unpinned_is_retuned_instead_of_refused() {
    let backend = live_backend();
    let ceiling = backend.max_compute_workgroups_per_dimension();
    let words = ceiling
        .checked_add(1)
        .and_then(|groups| groups.checked_mul(256))
        .expect("Fix: the ceiling must leave room for one more workgroup of 256 lanes.");
    let outputs = backend
        .dispatch(
            &one_dimensional_program(words),
            &[],
            &DispatchConfig::default(),
        )
        .expect(
            "Fix: a launch whose block the caller left free must be retuned into the device's \
             workgroup ceiling, not refused.",
        );
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0], 7_u32.to_le_bytes().repeat(4));
}
