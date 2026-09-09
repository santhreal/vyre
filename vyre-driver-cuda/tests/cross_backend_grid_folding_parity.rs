//! Cross-backend and reference interpreter parity contracts for grid folding (Row 124).
//!
//! WHY: A 1D program whose launch grid exceeds the single-axis workgroup ceiling
//! must fold across grid axes (x -> y -> z) while computing the exact same
//! buffer as the single-axis dispatch and the independent reference interpreter.
//!
//! This test exercises:
//! 1. Pinned ceiling folding (`DispatchConfig::max_workgroups_per_axis`).
//! 2. Unfolded vs folded output byte-for-byte identity on CUDA.
//! 3. Cross-backend output parity against the independent reference interpreter oracle.

#![cfg(feature = "device-tests")]

use crate::harness;
use crate::CudaBackend;
use vyre_driver::{DispatchConfig, VyreBackend};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_reference::value::Value;

/// A 1D program that stores its own element index at that index, over `words` elements.
fn identity_program(words: u32) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("out", 0, BufferAccess::WriteOnly, DataType::U32)
                .with_count(words)
                .with_output_byte_range(0..(words as usize * 4)),
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
    let ref_bytes: Vec<u8> = match &ref_outputs[0] {
        Value::U32(val) => val.to_le_bytes().to_vec(),
        Value::U32Vec(vals) => vals.iter().flat_map(|v| v.to_le_bytes()).collect(),
        other => panic!("Fix: unexpected reference value {other:?}"),
    };
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
