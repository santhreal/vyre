//! Declarations no path can size, so every path must refuse them and name the
//! remedy.

use vyre::ir::{BufferAccess, BufferDecl, DataType};

use super::harness::{a_bytes, assert_all_three_refuse, inputs_for, xor_program, N};

// ---------------------------------------------------------------------------
// The un-inferable declarations. These have no host bytes, so every path must
// refuse them and say how to fix it.
// ---------------------------------------------------------------------------

/// A countless `BufferDecl::output` is refused everywhere, naming `.with_count(n)`.
///
/// Locks out the CPU reference's certification hole: both GPU backends already
/// refused this, while the oracle answered it with an empty buffer, so a program
/// could pass the reference and be rejected by every real target.
#[test]
fn countless_output_declaration_is_refused_on_every_path_naming_the_remedy() {
    let program = xor_program(BufferDecl::output("out", 1, DataType::U32), N);
    let inputs = inputs_for(&program, 0, N);
    let [reference, cuda, wgpu] =
        assert_all_three_refuse(&program, &inputs, "countless BufferDecl::output");

    assert!(
        reference.contains(".with_count(n)"),
        "reference refusal must name the remedy, got: {reference}"
    );
    assert!(
        reference.contains("out"),
        "reference refusal must name the buffer, got: {reference}"
    );
    assert!(
        cuda.contains("with_count"),
        "CUDA refusal must name the remedy, got: {cuda}"
    );
    assert!(
        wgpu.contains(".with_count(n)"),
        "WGPU refusal must name the remedy, got: {wgpu}"
    );
    assert!(
        wgpu.contains("out"),
        "WGPU refusal must name the buffer, got: {wgpu}"
    );
}

/// A countless `WriteOnly` buffer is refused everywhere, naming `.with_count(n)`.
///
/// Locks out the SECOND silent cell found while sweeping this defect class:
/// `WriteOnly` is backend-allocated exactly like `BufferDecl::output`, yet WGPU
/// and the reference both answered a countless one with an empty buffer while
/// CUDA refused it. Same missing size, same absent host bytes, so the same
/// refusal.
#[test]
fn countless_write_only_declaration_is_refused_on_every_path_naming_the_remedy() {
    let program = xor_program(
        BufferDecl::storage("out", 1, BufferAccess::WriteOnly, DataType::U32),
        N,
    );
    let inputs = inputs_for(&program, 0, N);
    let [reference, _cuda, wgpu] =
        assert_all_three_refuse(&program, &inputs, "countless WriteOnly");

    assert!(
        reference.contains(".with_count(n)"),
        "reference refusal must name the remedy, got: {reference}"
    );
    assert!(
        wgpu.contains(".with_count(n)"),
        "WGPU refusal must name the remedy, got: {wgpu}"
    );
}

/// A countless `pipeline_live_out` `ReadWrite` is refused, naming `.with_count(n)`.
///
/// Locks out the third member of the backend-allocated set. Marking a `ReadWrite`
/// buffer live-out moves it from "seeded by the caller" to "allocated by the
/// backend", which removes the only source its size could have come from, so it
/// must refuse rather than inherit the plain `ReadWrite` inference path.
#[test]
fn countless_pipeline_live_out_read_write_is_refused_naming_the_remedy() {
    let program = xor_program(
        BufferDecl::read_write("out", 1, DataType::U32).with_pipeline_live_out(true),
        N,
    );
    let inputs = inputs_for(&program, 0, N);
    let [reference, _cuda, wgpu] =
        assert_all_three_refuse(&program, &inputs, "countless live-out read_write");

    assert!(
        reference.contains(".with_count(n)"),
        "reference refusal must name the remedy, got: {reference}"
    );
    assert!(
        wgpu.contains("count"),
        "WGPU refusal must mention the missing count, got: {wgpu}"
    );
}

/// A countless `read_write` with no input slice supplied is refused, not answered empty.
///
/// Locks out the other half of the original WGPU behavior: with the seed omitted
/// entirely, WGPU returned `Ok` with an empty buffer while the reference and CUDA
/// both refused for a missing input. Nothing can size the buffer in that state, so
/// `Ok` is the one answer that must not appear.
#[test]
fn countless_read_write_without_its_input_slice_is_refused_on_every_path() {
    let program = xor_program(BufferDecl::read_write("out", 1, DataType::U32), N);
    // Deliberately one short: only `a`, no seed for `out`.
    let inputs = vec![a_bytes(N)];
    assert_all_three_refuse(&program, &inputs, "countless read_write, seed omitted");
}
