//! Hostile-input closure obligations every backend owes, asserted once.
//!
//! A backend must never silently accept input it cannot honour. Two obligations
//! carry that: a hostile byte slice either dispatches or fails with a message
//! carrying a `Fix:` hint, and a caller who supplies more input buffers than the
//! program declares is rejected rather than having the extra buffers ignored.
//!
//! Each backend crate had written those obligations out again against its own
//! backend type, so the assertion text and the probe programs drifted per crate
//! while the contract did not. The programs and the assertions live here; a
//! backend's test supplies the backend and names itself.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use crate::{DispatchConfig, VyreBackend};

/// Smallest well-formed program with one `ReadWrite` output slot.
#[must_use]
pub fn single_output_program(output_init: u32) -> Program {
    Program::wrapped(
        vec![BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(output_init))],
    )
}

/// Well-formed program that reads one `ReadOnly` input into its output slot.
#[must_use]
pub fn read_one_write_one_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::read("input", 0, DataType::U32),
            BufferDecl::storage("out", 1, BufferAccess::ReadWrite, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![
            Node::let_bind("v", Expr::load("input", Expr::u32(0))),
            Node::store("out", Expr::u32(0), Expr::var("v")),
        ],
    )
}

/// Program whose X workgroup dimension is zero, which no backend may launch.
#[must_use]
pub fn zero_workgroup_program() -> Program {
    Program::wrapped(
        vec![BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32).with_count(1)],
        [0, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(42))],
    )
}

/// Assert a hostile byte slice never produces an unactionable failure.
///
/// Success is allowed: a hostile slice may happen to be a valid initializer.
/// What is not allowed is a failure the operator cannot act on.
///
/// # Panics
/// Panics when a dispatch fails with a message that carries no `Fix:` hint.
pub fn assert_hostile_bytes_stay_actionable(backend: &dyn VyreBackend, label: &str) {
    let hostile_inputs: &[&[u8]] = &[
        b"",
        b"\x00",
        b"\xff",
        &(0u32..64).flat_map(|w| w.to_le_bytes()).collect::<Vec<_>>(),
    ];
    let program = single_output_program(7);
    let config = DispatchConfig::default();
    for (index, hostile) in hostile_inputs.iter().enumerate() {
        if let Err(error) = backend.dispatch(&program, &[hostile.to_vec()], &config) {
            let message = error.to_string();
            assert!(
                message.contains("Fix:"),
                "Fix: {label} error for hostile case {index} must carry a Fix: hint. Got: {message}"
            );
        }
    }
}

/// Assert extra trailing input buffers are rejected, not ignored.
///
/// The program declares one `ReadOnly` input and one `ReadWrite` slot, so two
/// buffers is the correct call and three is the hostile one.
///
/// # Panics
/// Panics when the over-supplied dispatch succeeds, or fails without a `Fix:`
/// hint.
pub fn assert_trailing_inputs_rejected(backend: &dyn VyreBackend, label: &str) {
    let result = backend.dispatch(
        &read_one_write_one_program(),
        &[
            1u32.to_le_bytes().to_vec(),
            2u32.to_le_bytes().to_vec(),
            3u32.to_le_bytes().to_vec(),
        ],
        &DispatchConfig::default(),
    );
    let error = result.err().unwrap_or_else(|| {
        panic!("Fix: {label} must reject extra trailing input buffers instead of ignoring them.")
    });
    let message = error.to_string();
    assert!(
        message.contains("Fix:"),
        "Fix: {label} trailing-input rejection must carry a Fix: hint. Got: {message}"
    );
}

/// Assert a zero workgroup dimension is refused before any device work starts.
///
/// # Panics
/// Panics when the dispatch succeeds, or fails without a `Fix:` hint.
pub fn assert_zero_workgroup_rejected(backend: &dyn VyreBackend, label: &str) {
    let result = backend.dispatch(&zero_workgroup_program(), &[], &DispatchConfig::default());
    let error = result.err().unwrap_or_else(|| {
        panic!("Fix: {label} must reject a zero workgroup dimension before dispatch.")
    });
    let message = error.to_string();
    assert!(
        message.contains("Fix:"),
        "Fix: {label} zero-workgroup rejection must carry a Fix: hint. Got: {message}"
    );
}
