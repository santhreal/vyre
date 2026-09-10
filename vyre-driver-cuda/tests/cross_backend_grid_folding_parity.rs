//! Cross-backend and reference interpreter parity contracts for grid folding.
//!
//! WHY: A 1D program whose launch grid exceeds the single-axis workgroup ceiling
//! must fold across grid axes (x -> y -> z) while computing the exact same
//! buffer as the single-axis dispatch and the independent reference interpreter.
//!
//! This test exercises:
//! 1. Pinned ceiling folding (`DispatchConfig::max_workgroups_per_axis`).
//! 2. Unfolded vs folded output byte-for-byte identity on CUDA.
//! 3. Cross-backend output parity against the independent reference interpreter oracle.
//! 4. A program that reads two grid axes, over every dispatch route that can
//!    launch it.
//!
//! A folded launch and a two-axis launch are the same question asked from both
//! ends: which axes carry the element index. Folding moves a one-axis index
//! across three axes, and a two-axis program addresses two axes directly, so a
//! route that publishes a grid for one and not the other computes a buffer no
//! oracle agrees with.

#![cfg(feature = "device-tests")]

use crate::harness;
use vyre_driver::DispatchConfig;
use vyre_driver_cuda::CudaBackend;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// A 1D program that stores its own element index at that index, over `words` elements.
fn identity_program(words: u32) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("out", 0, BufferAccess::WriteOnly, DataType::U32)
                .with_count(words)
                .with_output_byte_range(0_usize..(words as usize * 4)),
        ],
        [256, 1, 1],
        vec![Node::if_then(
            Expr::lt(Expr::gid_x(), Expr::u32(words)),
            vec![Node::store("out", Expr::gid_x(), Expr::gid_x())],
        )],
    )
}

fn identity_bytes(words: u32) -> Vec<u8> {
    (0..words).flat_map(u32::to_le_bytes).collect()
}

#[test]
fn launch_past_the_pinned_per_axis_ceiling_folds_and_matches_cuda_and_reference() {
    let backend = CudaBackend::acquire()
        .expect("Fix: live CUDA backend is required for grid folding parity contracts");

    // 4096 elements with workgroup_size = [256, 1, 1] needs 16 workgroups.
    // Pinning max_workgroups_per_axis to [4, 65535, 65535] forces a 2D grid: 4 on x, 4 on y.
    const WORDS: u32 = 4096;
    let program = identity_program(WORDS);
    let expected = identity_bytes(WORDS);

    // 1. Reference interpreter oracle evaluation
    let ref_outputs = vyre_reference::reference_eval(&program, &[])
        .expect("Fix: reference_eval must evaluate the identity program");
    let ref_bytes: Vec<u8> = ref_outputs[0].to_bytes();
    assert_eq!(
        ref_bytes, expected,
        "Fix: reference interpreter must match explicit byte oracle"
    );

    // 2. Default CUDA dispatch (single axis on x: 16 workgroups on x, 1 on y)
    let default_config = DispatchConfig::default();
    let default_outputs = backend
        .dispatch(&program, &[], &default_config)
        .expect("Fix: default single-axis CUDA dispatch must succeed");
    assert_eq!(
        default_outputs,
        vec![expected.clone()],
        "Fix: default single-axis CUDA dispatch must match explicit byte oracle"
    );

    // 3. Folded CUDA dispatch (folded across axes: 4 on x, 4 on y)
    let mut folded_config = DispatchConfig::default();
    folded_config.max_workgroups_per_axis = Some([4, 65_535, 65_535]);
    let folded_outputs = backend
        .dispatch(&program, &[], &folded_config)
        .expect("Fix: folded CUDA dispatch must succeed");

    assert_eq!(
        folded_outputs,
        vec![expected.clone()],
        "Fix: folded CUDA dispatch must match explicit byte oracle"
    );

    // 4. Parity assertion: folded matches unfolded and reference byte-for-byte
    assert_eq!(
        folded_outputs, default_outputs,
        "Fix: folded CUDA dispatch must match default single-axis CUDA dispatch byte-for-byte"
    );
    assert_eq!(
        folded_outputs[0], ref_bytes,
        "Fix: folded CUDA dispatch must match reference interpreter oracle byte-for-byte"
    );
}

#[test]
fn an_inferred_launch_past_the_device_single_axis_ceiling_folds_and_matches_reference() {
    let backend = CudaBackend::acquire()
        .expect("Fix: live CUDA backend is required for grid folding parity contracts");

    // Exercise folded dispatch through grid inference on CUDA
    const WORDS: u32 = 8192;
    let program = identity_program(WORDS);
    let expected = identity_bytes(WORDS);

    let mut folded_config = DispatchConfig::default();
    folded_config.max_workgroups_per_axis = Some([8, 65_535, 65_535]);

    let outputs = backend
        .dispatch(&program, &[], &folded_config)
        .expect("Fix: inferred folded dispatch must succeed");

    assert_eq!(
        outputs,
        vec![expected],
        "Fix: inferred folded CUDA launch must match expected byte oracle"
    );
}

/// Side of the square element space [`two_axis_program`] addresses.
const TWO_AXIS_SIDE: u32 = 32;

/// A program whose element index comes from two grid axes.
///
/// `row` is the x axis and `col` is the y axis, and the stored value is the
/// element's own linear index, so a launch that covers the space writes the
/// identity buffer and one that covers part of it leaves the rest at whatever
/// the route staged. The workgroup is square, which is what makes the y axis
/// load-bearing: under a grid published on x alone every lane reads
/// `col = tid.y`, so only the first `workgroup[1]` columns are ever written.
fn two_axis_program(side: u32) -> Program {
    let count = side * side;
    Program::wrapped(
        vec![
            BufferDecl::storage("out", 0, BufferAccess::WriteOnly, DataType::U32)
                .with_count(count)
                .with_output_byte_range(0_usize..(count as usize * 4)),
        ],
        [16, 16, 1],
        vec![
            Node::let_bind("row", Expr::gid_x()),
            Node::let_bind("col", Expr::gid_y()),
            Node::if_then(
                Expr::and(
                    Expr::lt(Expr::var("row"), Expr::u32(side)),
                    Expr::lt(Expr::var("col"), Expr::u32(side)),
                ),
                vec![
                    Node::let_bind(
                        "index",
                        Expr::add(
                            Expr::mul(Expr::var("row"), Expr::u32(side)),
                            Expr::var("col"),
                        ),
                    ),
                    Node::store("out", Expr::var("index"), Expr::var("index")),
                ],
            ),
        ],
    )
}

#[test]
fn a_two_axis_program_writes_the_same_buffer_on_every_route_that_launches_it() {
    let backend = CudaBackend::acquire()
        .expect("Fix: live CUDA backend is required for grid parity contracts");

    let program = two_axis_program(TWO_AXIS_SIDE);
    let count = TWO_AXIS_SIDE * TWO_AXIS_SIDE;
    let expected: Vec<u8> = (0..count).flat_map(u32::to_le_bytes).collect();

    let reference = vyre_reference::reference_eval(&program, &[])
        .expect("Fix: reference_eval must evaluate the two-axis program");
    assert_eq!(
        reference[0].to_bytes(),
        expected,
        "Fix: reference interpreter must match the explicit byte oracle for a two-axis program"
    );

    let mut pinned = DispatchConfig::default();
    pinned.grid_override = Some([TWO_AXIS_SIDE / 16, TWO_AXIS_SIDE / 16, 1]);
    let pinned_outputs = backend
        .dispatch(&program, &[], &pinned)
        .expect("Fix: a pinned two-axis grid must dispatch on CUDA");
    assert_eq!(
        pinned_outputs[0], expected,
        "Fix: a pinned two-axis CUDA grid must write every element the program addresses"
    );

    let artifact_outputs =
        harness::compiled_cuda_outputs(&backend, &program, &[], "two-axis-identity");
    assert_eq!(
        artifact_outputs[0], expected,
        "Fix: the authenticated CUDA artifact route must publish a grid covering both axes a program reads"
    );
}
