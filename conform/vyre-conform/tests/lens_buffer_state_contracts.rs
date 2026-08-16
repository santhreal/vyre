//! Contracts for `vyre_conform::lens::buffer_state`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_conform::lens::buffer_state::project_output_buffers;
use vyre_foundation::fp_parity::{compare_output_buffers, BufferParity};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Program};

/// Mirrors the real `vyre-libs::security::flows_to` layout: many
/// read-only graph/input buffers plus a single read-write frontier.
/// The convergence loop carries the FULL state (7 buffers), but
/// `compare_output_buffers` is contractually sized to the program's
/// declared outputs. Before the projection fix the lens fed the
/// 7-buffer state straight into the comparator, which rejected it with
/// "program declares 1 output buffer(s), compared 7 result buffer(s)"
/// even when CPU and GPU agreed byte-for-byte, a false parity
/// failure. These two assertions pin both halves: the raw full state
/// is rejected, the projected state is accepted.
fn flows_to_shaped_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("pg_nodes", 0, BufferAccess::ReadOnly, DataType::U32).with_count(4),
            BufferDecl::storage("pg_edges", 1, BufferAccess::ReadOnly, DataType::U32).with_count(4),
            BufferDecl::storage("edge_offsets", 2, BufferAccess::ReadOnly, DataType::U32)
                .with_count(4),
            BufferDecl::storage("sources", 3, BufferAccess::ReadOnly, DataType::U32).with_count(4),
            BufferDecl::storage("dims", 4, BufferAccess::ReadOnly, DataType::U32).with_count(1),
            BufferDecl::storage("reach_in", 5, BufferAccess::ReadOnly, DataType::U32).with_count(4),
            BufferDecl::storage("reach_out", 6, BufferAccess::ReadWrite, DataType::U32)
                .with_count(4),
        ],
        [1, 1, 1],
        vec![],
    )
}

#[test]
fn project_output_buffers_selects_only_the_declared_output() {
    let program = flows_to_shaped_program();
    // Full convergence state: a distinct value per buffer so a wrong
    // projection is detectable, not coincidentally equal.
    let full_state: Vec<Vec<u8>> = (0..7u32)
        .map(|index| {
            let byte = (index + 1) as u8;
            vec![byte; if index == 4 { 4 } else { 16 }]
        })
        .collect();
    assert_eq!(program.output_buffer_indices(), &[6]);

    let projected =
        project_output_buffers(&program, &full_state).expect("Fix: projection must succeed");
    assert_eq!(
        projected.len(),
        1,
        "exactly one declared output (reach_out) must survive projection"
    );
    assert_eq!(
        projected[0], full_state[6],
        "the surviving buffer must be reach_out (the RW frontier), byte-for-byte"
    );
}

#[test]
fn full_state_breaks_comparator_but_projection_restores_parity() {
    let program = flows_to_shaped_program();
    let full_state: Vec<Vec<u8>> = (0..7u32)
        .map(|index| vec![(index + 1) as u8; if index == 4 { 4 } else { 16 }])
        .collect();

    // The raw full state (CPU == GPU, both 7 buffers) is STILL rejected
    // by the comparator because it is sized to declared outputs (1).
    // This is the exact false failure the GPU parity test hit.
    match compare_output_buffers(&program, &full_state, &full_state) {
        BufferParity::Mismatch(detail) => assert!(
            detail.contains("declares 1 output buffer(s), compared 7 result buffer(s)"),
            "expected the output-count mismatch, got: {detail}"
        ),
        BufferParity::Ok => {
            panic!("full 7-buffer state must NOT satisfy a 1-output comparator")
        }
    }

    // Projecting to declared outputs first makes identical states pass.
    let cpu = project_output_buffers(&program, &full_state).expect("Fix: projection");
    let gpu = project_output_buffers(&program, &full_state).expect("Fix: projection");
    assert!(
        matches!(
            compare_output_buffers(&program, &cpu, &gpu),
            BufferParity::Ok
        ),
        "projected identical outputs must compare equal"
    );
}

#[test]
fn project_output_buffers_errs_loudly_on_missing_output_slot() {
    let program = flows_to_shaped_program();
    // State truncated before the RW output index (6), a malformed
    // fixture. Projection must surface this, never silently drop it.
    let short_state: Vec<Vec<u8>> = vec![vec![0u8; 16]; 3];
    let err = project_output_buffers(&program, &short_state)
        .expect_err("missing output slot must be an error, not a silent short vector");
    assert!(
        err.contains("declares an output at index 6"),
        "error must name the missing output index: {err}"
    );
}
