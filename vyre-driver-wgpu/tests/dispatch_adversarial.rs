//! P1 inventory #88  -  adversarial tests for every dispatch path.
//!
//! Hostile inputs against `WgpuBackend::dispatch` and friends. The
//! test suite asserts each adversarial input produces a structured
//! `BackendError` rather than a panic, a hang, or undefined behavior.
//!
//! Coverage targets (the 6 dispatch paths the audit calls out):
//!   - direct synchronous dispatch
//!   - compiled-pipeline dispatch
//!   - async dispatch
//!   - compound dispatch (multi-stage)
//!   - persistent dispatch
//!   - megakernel dispatch
//!
//! GPU-required: each test acquires a real adapter; no silent skip.
//! `scripts/check_gpu_test_loudness.sh` enforces the loudness rule.

use vyre::ir::{BufferDecl, DataType, Program};
use vyre_driver::{BackendError, VyreBackend};

fn no_op_program() -> Program {
    Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        Vec::new(),
    )
}

#[test]
fn no_op_program_returns_zero_initialized_output() {
    let program = no_op_program();

    let backend = vyre_driver_wgpu::WgpuBackend::acquire()
        .expect("Fix: live WGPU backend is required for empty-program dispatch coverage");

    let inputs: Vec<Vec<u8>> = vec![];
    let config = vyre_driver::DispatchConfig::default();
    let result = backend.dispatch(&program, &inputs, &config);
    assert_eq!(
        result.expect("a no-op program with one sized output must dispatch"),
        vec![vec![0, 0, 0, 0]],
        "backend-allocated no-op output must be deterministically zero initialized"
    );
}

#[test]
fn dispatch_with_mismatched_inputs_yields_structured_error() {
    // Program declares one read buffer; we pass zero inputs  -  the
    // backend must structurally reject before submitting.
    let program = Program::wrapped(
        vec![
            BufferDecl::read("a", 0, DataType::U32).with_count(1),
            BufferDecl::output("out", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        Vec::new(),
    );
    let backend = vyre_driver_wgpu::WgpuBackend::acquire()
        .expect("Fix: live WGPU backend is required for adversarial dispatch coverage");
    let inputs: Vec<Vec<u8>> = vec![]; // mismatched: 0 < 1 expected
    let config = vyre_driver::DispatchConfig::default();
    let result = backend.dispatch(&program, &inputs, &config);
    assert!(
        matches!(result, Err(BackendError::Other(_))),
        "missing-input dispatch must return the owner-local actionable error, got {result:?}"
    );
}
