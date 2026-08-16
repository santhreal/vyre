//! Counted controls for the refused declaration forms, and the host-abort path
//! a counted buffer must return an error for.

use vyre::ir::{BufferAccess, BufferDecl, DataType};

use super::harness::{
    assert_all_three_return, expected_bytes, inputs_for, run_wgpu, xor_program, N,
};

// ---------------------------------------------------------------------------
// Counted controls for the refused declaration forms, so the refusals above are
// provably about the missing count and not about the declaration form itself.
// ---------------------------------------------------------------------------

/// A COUNTED `BufferDecl::output` works on all three paths.
#[test]
fn counted_output_declaration_agrees_on_exact_bytes_across_three_backends() {
    let program = xor_program(BufferDecl::output("out", 1, DataType::U32).with_count(N), N);
    let inputs = inputs_for(&program, 0, N);
    assert_all_three_return(&program, &inputs, &expected_bytes(N), "counted output");
}

/// A COUNTED `WriteOnly` buffer works on all three paths.
#[test]
fn counted_write_only_declaration_agrees_on_exact_bytes_across_three_backends() {
    let program = xor_program(
        BufferDecl::storage("out", 1, BufferAccess::WriteOnly, DataType::U32).with_count(N),
        N,
    );
    let inputs = inputs_for(&program, 0, N);
    assert_all_three_return(&program, &inputs, &expected_bytes(N), "counted WriteOnly");
}

/// A COUNTED `pipeline_live_out` `ReadWrite` works on all three paths.
#[test]
fn counted_pipeline_live_out_read_write_agrees_on_exact_bytes_across_three_backends() {
    let program = xor_program(
        BufferDecl::read_write("out", 1, DataType::U32)
            .with_pipeline_live_out(true)
            .with_count(N),
        N,
    );
    let inputs = inputs_for(&program, 0, N);
    assert_all_three_return(&program, &inputs, &expected_bytes(N), "counted live-out");
}

// ---------------------------------------------------------------------------
// The host-abort path.
// ---------------------------------------------------------------------------

/// Oversupplying a COUNTED buffer returns a vyre error, never a host abort.
///
/// Locks out the second symptom of the original defect: supplying more bytes than
/// the destination buffer holds reached `Queue::write_buffer`, which raised a wgpu
/// validation error and took the host process down instead of returning a
/// `Result`. A library must not abort its host over a buffer declaration, so this
/// asserts an `Err` whose message names the two sizes.
#[test]
fn oversupplying_a_counted_read_write_returns_an_error_not_a_process_abort() {
    let program = xor_program(
        BufferDecl::read_write("out", 1, DataType::U32).with_count(1),
        N,
    );
    // The declaration says one u32, four bytes. Supply sixteen.
    let inputs = inputs_for(&program, 16, N);
    let error = run_wgpu(&program, &inputs)
        .expect_err("WGPU must refuse an upload that would overrun the destination buffer");
    assert!(
        error.contains("overrun"),
        "the refusal must say the upload would overrun, got: {error}"
    );
    assert!(
        error.contains("16"),
        "the refusal must name the supplied length, got: {error}"
    );
    assert!(
        error.contains(".with_count(n)"),
        "the refusal must name the remedy, got: {error}"
    );
}
