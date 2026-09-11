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
            .with_output_byte_range(0_usize..256)],
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
            .with_output_byte_range(0_usize..16)],
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
/// boundary value must be admitted.
///
/// Admission is observed at the validation seam rather than by dispatching. The
/// ceiling is a device capability, and on a discrete adapter it is 2147483647
/// workgroups: at the 256-lane workgroup this program declares that is 5.5e11
/// invocations, which does not complete. Executing it read four output words
/// after a launch whose only bound was the test process, and one run held a
/// device for 23 hours at full host spin while every other job on that machine
/// queued behind it. The boundary this test defends is whether the check admits
/// the value, and `validate_program_for_backend` answers that in full.
///
/// What it does not catch: a refusal raised after validation, inside command
/// recording. The test below covers execution at a grid wide enough to prove a
/// launch past the WebGPU per-axis minimum runs, and does it in a bounded time.
#[test]
fn a_grid_at_the_device_ceiling_is_admitted() {
    let backend = live_backend();
    let ceiling = backend.max_compute_workgroups_per_dimension();
    let program = one_dimensional_program(1024);
    for axis in 0..3 {
        let mut grid = [1_u32; 3];
        grid[axis] = ceiling;
        let mut config = DispatchConfig::default();
        config.grid_override = Some(grid);
        vyre_driver::validation::validate_program_for_backend(&backend, &program, &config)
            .unwrap_or_else(|error| {
                let axis_name = ["x", "y", "z"][axis];
                panic!(
                    "Fix: a grid of exactly the reported per-axis ceiling on {axis_name} must be \
                     admitted, because refusing it takes every large launch off this backend. \
                     Ceiling {ceiling}, got: {error}"
                )
            });
    }
}

/// WHY: the executable half of the boundary. A per-axis ceiling check is worth
/// nothing if a launch past the WebGPU per-axis minimum of 65535 does not run, so
/// one grid wider than that minimum dispatches and its bytes are read back.
///
/// The width is the minimum plus one rather than the device ceiling: the property
/// is that the folding path is entered at all, and the smallest grid that enters
/// it proves the same thing in milliseconds.
#[test]
fn a_grid_wider_than_the_per_axis_minimum_dispatches() {
    let backend = live_backend();
    let floor = harness::WEBGPU_MAX_WORKGROUPS_PER_AXIS;
    let ceiling = backend.max_compute_workgroups_per_dimension();
    assert!(
        ceiling > floor,
        "Fix: this backend reports a per-axis ceiling of {ceiling}, at or below the WebGPU \
         minimum of {floor}, so there is no wider grid to dispatch."
    );
    let mut config = DispatchConfig::default();
    config.grid_override = Some([floor + 1, 1, 1]);
    let outputs = backend
        .dispatch(&one_dimensional_program(1024), &[], &config)
        .expect("Fix: a grid wider than the WebGPU per-axis minimum must dispatch.");
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
            .with_output_byte_range(0_usize..(words as usize * 4))],
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
    let reference = vyre_reference::ReferenceRequest::standard(&program, &[])
        .outputs()
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

/// WHY: closes the class "a 1D program whose element count exceeds the device's
/// single-axis ceiling is refused instead of folded in grid inference". The element
/// count is derived at run time from the adapter's reported per-axis ceiling and
/// the program's workgroup size, and the fixture proves that the folded lane in
/// the second row is executed, addresses with its linearized index, and produces
/// the exact same buffer on wgpu and on the reference interpreter.
#[test]
fn an_inferred_launch_past_the_device_single_axis_ceiling_folds_and_matches_reference() {
    let backend = live_backend();
    let ceiling = backend.max_compute_workgroups_per_dimension();
    let workgroup_size = 256_u32;
    let single_axis_ceiling = ceiling
        .checked_mul(workgroup_size)
        .expect("Fix: single-axis ceiling multiplication must not overflow u32.");
    let words = single_axis_ceiling
        .checked_add(1)
        .expect("Fix: adding one element past the single-axis ceiling must not overflow u32.");

    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32)
            .with_count(words)
            .with_output_byte_range(0_usize..8)],
        [workgroup_size, 1, 1],
        vec![
            Node::if_then(
                Expr::eq(Expr::gid_x(), Expr::u32(0)),
                vec![Node::store("out", Expr::u32(0), Expr::u32(7))],
            ),
            Node::if_then(
                Expr::eq(Expr::gid_x(), Expr::u32(single_axis_ceiling)),
                vec![Node::store("out", Expr::u32(1), Expr::u32(9))],
            ),
        ],
    );

    let outputs = backend
        .dispatch(&program, &[], &DispatchConfig::default())
        .expect("Fix: an inferred 1D launch past the device single-axis ceiling must fold across axes, not be refused.");

    let expected = [7_u32, 9_u32]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    assert_eq!(
        outputs[0], expected,
        "Fix: lane 0 and the first folded lane must write their marker values at their linearized indices."
    );

    let reference = vyre_reference::ReferenceRequest::standard(&program, &[])
        .outputs()
        .expect("Fix: reference interpreter must evaluate the folded launch program.")
        .into_iter()
        .map(|value| value.to_bytes())
        .collect::<Vec<_>>();
    assert_eq!(
        outputs[0], reference[0],
        "Fix: a folded launch above the single-axis ceiling must produce what the reference interpreter produces."
    );
}

/// WHY: a launch whose element count exceeds the product of every grid axis limit
/// must be refused rather than silently under-dispatched or overflowing.
#[test]
fn an_inferred_launch_past_the_product_of_every_axis_limit_is_refused_naming_the_limit() {
    let backend = live_backend();
    let mut config = DispatchConfig::default();
    config.max_workgroups_per_axis = Some([2, 2, 1]);
    // 2 * 2 * 1 * 256 = 1024 lanes max capacity. Asking for 1025 elements must fail.
    let words = 1025_u32;
    let program = identity_program(words);
    let error = backend
        .dispatch(&program, &[], &config)
        .expect_err("Fix: a launch past the product of every axis limit must be refused.");
    let message = error.to_string();
    assert!(
        message.contains("1025") && message.contains("1024") && message.contains("Fix:"),
        "Fix: the refusal must name the element count asked for and the capacity limit: {message}"
    );
}
/// WHY: closes the class "an out-of-range invocation in the folded tail writes past the buffer
/// or wraps and overwrites a valid element". When a 1D launch of 1025 elements with a 256-lane
/// workgroup is folded across a 4-workgroup per-axis ceiling, the inferred grid is [4, 2, 1],
/// which launches 2048 total invocations. Invocations 0..1025 must write their exact values,
/// while the 1023 tail invocations (1025..2048) in the second row must not write into the
/// buffer or overwrite elements 0..1022.
#[test]
fn a_launch_one_element_above_the_pinned_per_axis_ceiling_proves_tail_guard() {
    let backend = live_backend();
    let words = 1025_u32;
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32)
            .with_count(words)
            .with_output_byte_range(0_usize..(words as usize * 4))],
        [256, 1, 1],
        vec![Node::if_then(
            Expr::lt(Expr::gid_x(), Expr::u32(words)),
            vec![Node::store(
                "out",
                Expr::gid_x(),
                Expr::add(Expr::gid_x(), Expr::u32(50)),
            )],
        )],
    );

    let mut config = DispatchConfig::default();
    config.max_workgroups_per_axis = Some([4, 65_535, 65_535]);

    let outputs = backend
        .dispatch(&program, &[], &config)
        .expect("Fix: 1025 elements on a 1024 ceiling must fold to [4, 2, 1] and dispatch.");

    let expected = (0..words)
        .map(|i| i + 50)
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    assert_eq!(
        outputs[0], expected,
        "Fix: every lane 0..1025 must store its value and no tail invocation may overwrite elements."
    );

    let reference = vyre_reference::ReferenceRequest::standard(&program, &[])
        .outputs()
        .expect("Fix: reference interpreter must evaluate 1025-element folded launch.")
        .into_iter()
        .map(|value| value.to_bytes())
        .collect::<Vec<_>>();
    assert_eq!(
        outputs[0], reference[0],
        "Fix: folded launch one element above ceiling must match reference interpreter."
    );
}

/// WHY: closes the class "a launch one element below a fold boundary miscalculates grid shape
/// or fails to guard the single trailing idle invocation".
/// Boundary 1: 1023 elements (one element below the 1024 single-axis ceiling, grid [4, 1, 1]).
/// Boundary 2: 2047 elements (one element below the 2048 2-row fold boundary, grid [4, 2, 1]).
#[test]
fn a_launch_one_element_below_fold_boundary_proves_boundary_and_tail_guard() {
    let backend = live_backend();
    for words in [1023_u32, 2047_u32] {
        let program = Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32)
                .with_count(words)
                .with_output_byte_range(0_usize..(words as usize * 4))],
            [256, 1, 1],
            vec![Node::if_then(
                Expr::lt(Expr::gid_x(), Expr::u32(words)),
                vec![Node::store(
                    "out",
                    Expr::gid_x(),
                    Expr::add(Expr::gid_x(), Expr::u32(77)),
                )],
            )],
        );

        let mut config = DispatchConfig::default();
        config.max_workgroups_per_axis = Some([4, 65_535, 65_535]);

        let outputs = backend
            .dispatch(&program, &[], &config)
            .unwrap_or_else(|error| {
                panic!("Fix: {words} elements must dispatch under pinned ceiling: {error}")
            });

        let expected = (0..words)
            .map(|i| i + 77)
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        assert_eq!(
            outputs[0], expected,
            "Fix: every lane 0..{words} must store correctly without tail corruption."
        );

        let reference = vyre_reference::ReferenceRequest::standard(&program, &[])
            .outputs()
            .unwrap_or_else(|error| {
                panic!("Fix: reference interpreter must evaluate {words} elements: {error}")
            })
            .into_iter()
            .map(|value| value.to_bytes())
            .collect::<Vec<_>>();
        assert_eq!(
            outputs[0], reference[0],
            "Fix: {words} elements must match reference interpreter."
        );
    }
}

/// WHY: verifies boundary cases around the live device's single-axis ceiling:
/// 1. Exactly at the ceiling (`single_axis_ceiling`).
/// 2. One element below the ceiling (`single_axis_ceiling - 1`).
/// 3. One element above the ceiling (`single_axis_ceiling + 1`).
/// Asserts exact marker buffer contents and parity against the reference interpreter.
#[test]
fn an_inferred_launch_at_exact_device_ceiling_and_boundaries_matches_reference() {
    let backend = live_backend();
    let ceiling = backend.max_compute_workgroups_per_dimension();
    let workgroup_size = 256_u32;
    let single_axis_ceiling = ceiling
        .checked_mul(workgroup_size)
        .expect("Fix: single-axis ceiling multiplication must not overflow u32.");

    // Case 1: Exactly at ceiling
    {
        let words = single_axis_ceiling;
        let program = Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32)
                .with_count(words)
                .with_output_byte_range(0_usize..8)],
            [workgroup_size, 1, 1],
            vec![
                Node::if_then(
                    Expr::eq(Expr::gid_x(), Expr::u32(0)),
                    vec![Node::store("out", Expr::u32(0), Expr::u32(11))],
                ),
                Node::if_then(
                    Expr::eq(Expr::gid_x(), Expr::u32(single_axis_ceiling - 1)),
                    vec![Node::store("out", Expr::u32(1), Expr::u32(22))],
                ),
            ],
        );
        let outputs = backend
            .dispatch(&program, &[], &DispatchConfig::default())
            .expect("Fix: exact device ceiling launch must dispatch.");
        let expected = [11_u32, 22_u32]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        assert_eq!(outputs[0], expected);
        let reference = vyre_reference::ReferenceRequest::standard(&program, &[])
            .outputs()
            .expect("Fix: reference must evaluate exact ceiling program.")
            .into_iter()
            .map(|value| value.to_bytes())
            .collect::<Vec<_>>();
        assert_eq!(outputs[0], reference[0]);
    }

    // Case 2: One below ceiling
    {
        let words = single_axis_ceiling - 1;
        let program = Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32)
                .with_count(words)
                .with_output_byte_range(0_usize..8)],
            [workgroup_size, 1, 1],
            vec![
                Node::if_then(
                    Expr::eq(Expr::gid_x(), Expr::u32(0)),
                    vec![Node::store("out", Expr::u32(0), Expr::u32(33))],
                ),
                Node::if_then(
                    Expr::eq(Expr::gid_x(), Expr::u32(single_axis_ceiling - 2)),
                    vec![Node::store("out", Expr::u32(1), Expr::u32(44))],
                ),
            ],
        );
        let outputs = backend
            .dispatch(&program, &[], &DispatchConfig::default())
            .expect("Fix: one-below device ceiling launch must dispatch.");
        let expected = [33_u32, 44_u32]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        assert_eq!(outputs[0], expected);
        let reference = vyre_reference::ReferenceRequest::standard(&program, &[])
            .outputs()
            .expect("Fix: reference must evaluate one-below ceiling program.")
            .into_iter()
            .map(|value| value.to_bytes())
            .collect::<Vec<_>>();
        assert_eq!(outputs[0], reference[0]);
    }

    // Case 3: One above ceiling (folds to 2 rows on y)
    {
        let words = single_axis_ceiling + 1;
        let program = Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32)
                .with_count(words)
                .with_output_byte_range(0_usize..12)],
            [workgroup_size, 1, 1],
            vec![
                Node::if_then(
                    Expr::eq(Expr::gid_x(), Expr::u32(0)),
                    vec![Node::store("out", Expr::u32(0), Expr::u32(55))],
                ),
                Node::if_then(
                    Expr::eq(Expr::gid_x(), Expr::u32(single_axis_ceiling - 1)),
                    vec![Node::store("out", Expr::u32(1), Expr::u32(66))],
                ),
                Node::if_then(
                    Expr::eq(Expr::gid_x(), Expr::u32(single_axis_ceiling)),
                    vec![Node::store("out", Expr::u32(2), Expr::u32(77))],
                ),
            ],
        );
        let outputs = backend
            .dispatch(&program, &[], &DispatchConfig::default())
            .expect("Fix: one-above device ceiling launch must fold and dispatch.");
        let expected = [55_u32, 66_u32, 77_u32]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        assert_eq!(outputs[0], expected);
        let reference = vyre_reference::ReferenceRequest::standard(&program, &[])
            .outputs()
            .expect("Fix: reference must evaluate one-above ceiling program.")
            .into_iter()
            .map(|value| value.to_bytes())
            .collect::<Vec<_>>();
        assert_eq!(outputs[0], reference[0]);
    }
}
