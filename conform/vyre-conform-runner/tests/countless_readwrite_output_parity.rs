//! Writable buffer declarations must be honored or refused, never silently dropped.
//!
//! # The defect this suite locks out
//!
//! One program, an element-wise XOR whose result buffer was declared
//! `BufferDecl::read_write("out", ..)` with NO `.with_count(n)`, returned three
//! different answers and no error:
//!
//! ```text
//! reference = [255, 254, 253, 252]   4 words
//! cuda      = [255]                  1 word
//! wgpu      = []                     0 words
//! ```
//!
//! `Ok` asserts success, so a consumer reading zero elements could not tell "the
//! kernel wrote nothing" apart from "the backend discarded my output". Supplying
//! the result buffer at its full length instead made WGPU abort the host process
//! from inside `Queue::write_buffer`, and the same omission on
//! `BufferDecl::output` was refused loudly by both GPU backends while the CPU
//! reference answered it with an empty buffer.
//!
//! # The rule these tests hold
//!
//! A buffer declared without `.with_count(n)` has `count == 0`, which the IR
//! defines as runtime-sized. Two cases follow, and they are NOT the same case:
//!
//! - A buffer that receives host bytes (a plain `ReadWrite`) takes its element
//!   count from those bytes. Every path must honor it and return the same bytes.
//! - A buffer the backend allocates itself, selected by
//!   `BufferDecl::is_backend_allocated_output` (`BufferDecl::output`, any
//!   `WriteOnly`, or a `pipeline_live_out` `ReadWrite`), receives no host bytes,
//!   so nothing can size it. Every path must REFUSE it and name `.with_count(n)`.
//!
//! Every assertion below compares EXACT bytes or EXACT error text. Asserting that
//! each backend merely returned something would have passed on the original bug,
//! because `[255, 254, 253, 252]`, `[255]`, and `[]` are all "a result".
#![forbid(unsafe_code)]

use vyre::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre::{DispatchConfig, VyreBackend};
use vyre_reference::value::Value;

/// Element count for the main cases. Matches the `[64, 1, 1]` workgroup exactly.
const N: u32 = 64;

/// Build the XOR program whose single writable buffer is declared by `out`.
///
/// The store is gated on `idx < len` so the boundary cases (one element, zero
/// elements) do not depend on a backend absorbing out-of-range stores. The
/// expected bytes are a pure function of the input, so any disagreement between
/// backends is a backend defect and not an ambiguity in the program.
fn xor_program(out: BufferDecl, len: u32) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::read("a", 0, DataType::U32).with_count(len.max(1)),
            out,
        ],
        [64, 1, 1],
        vec![
            Node::let_bind("idx", Expr::gid_x()),
            Node::if_then(
                Expr::lt(Expr::var("idx"), Expr::u32(len)),
                vec![Node::store(
                    "out",
                    Expr::var("idx"),
                    Expr::bitxor(Expr::load("a", Expr::var("idx")), Expr::u32(0x0F)),
                )],
            ),
        ],
    )
}

/// Input bytes for `a`: the words `0..len`.
fn a_bytes(len: u32) -> Vec<u8> {
    (0..len).flat_map(u32::to_le_bytes).collect()
}

/// The bytes every path must return for [`xor_program`] over `len` elements.
fn expected_bytes(len: u32) -> Vec<u8> {
    (0..len).flat_map(|i| (i ^ 0x0F).to_le_bytes()).collect()
}

/// Inputs in the ABI every path shares: one slice per buffer the backend does
/// NOT allocate itself, in `Program::buffers()` order.
///
/// `vyre_reference::is_reference_input` is that predicate. Deriving the input
/// vector from it rather than hand-rolling it is what keeps the three backends
/// comparable: handing the reference a different seed length than the GPUs is
/// exactly the mistake that made one backend look correct and another look
/// truncated when both were applying the same rule.
fn inputs_for(program: &Program, seed_bytes: usize, len: u32) -> Vec<Vec<u8>> {
    program
        .buffers()
        .iter()
        .filter(|decl| vyre_reference::is_reference_input(decl))
        .map(|decl| match decl.name() {
            "a" => a_bytes(len),
            _ => vec![0u8; seed_bytes],
        })
        .collect()
}

fn run_reference(program: &Program, inputs: &[Vec<u8>]) -> Result<Vec<Vec<u8>>, String> {
    let values: Vec<Value> = inputs.iter().map(|b| Value::from(b.as_slice())).collect();
    vyre_reference::reference_eval(program, &values)
        .map(|outputs| outputs.into_iter().map(|value| value.to_bytes()).collect())
        .map_err(|error| error.to_string())
}

fn run_cuda(program: &Program, inputs: &[Vec<u8>]) -> Result<Vec<Vec<u8>>, String> {
    let backend = vyre_driver_cuda::CudaBackend::acquire()
        .map_err(|error| format!("CUDA acquire failed: {error}"))?;
    let borrowed: Vec<&[u8]> = inputs.iter().map(Vec::as_slice).collect();
    backend
        .dispatch_borrowed(program, &borrowed, &DispatchConfig::default())
        .map_err(|error| error.to_string())
}

fn run_wgpu(program: &Program, inputs: &[Vec<u8>]) -> Result<Vec<Vec<u8>>, String> {
    let backend = vyre_driver_wgpu::WgpuBackend::acquire()
        .map_err(|error| format!("WGPU acquire failed: {error}"))?;
    let borrowed: Vec<&[u8]> = inputs.iter().map(Vec::as_slice).collect();
    backend
        .dispatch_borrowed(program, &borrowed, &DispatchConfig::default())
        .map_err(|error| error.to_string())
}

/// Assert the three paths return byte-identical first output buffers.
///
/// Compares against `expected` as well as against each other, so a shared wrong
/// answer cannot pass.
fn assert_all_three_return(program: &Program, inputs: &[Vec<u8>], expected: &[u8], case: &str) {
    let reference = run_reference(program, inputs)
        .unwrap_or_else(|error| panic!("{case}: reference must dispatch, got: {error}"));
    let cuda = run_cuda(program, inputs)
        .unwrap_or_else(|error| panic!("{case}: CUDA must dispatch, got: {error}"));
    let wgpu = run_wgpu(program, inputs)
        .unwrap_or_else(|error| panic!("{case}: WGPU must dispatch, got: {error}"));

    assert_eq!(
        reference[0].len(),
        expected.len(),
        "{case}: reference returned {} bytes, expected {}",
        reference[0].len(),
        expected.len()
    );
    assert_eq!(reference[0], expected, "{case}: reference bytes");
    assert_eq!(
        cuda[0].len(),
        expected.len(),
        "{case}: CUDA returned {} bytes, expected {}",
        cuda[0].len(),
        expected.len()
    );
    assert_eq!(cuda[0], expected, "{case}: CUDA bytes");
    assert_eq!(
        wgpu[0].len(),
        expected.len(),
        "{case}: WGPU returned {} bytes, expected {}. A zero length here is the original defect.",
        wgpu[0].len(),
        expected.len()
    );
    assert_eq!(wgpu[0], expected, "{case}: WGPU bytes");
    assert_eq!(
        reference[0], cuda[0],
        "{case}: reference and CUDA must agree"
    );
    assert_eq!(cuda[0], wgpu[0], "{case}: CUDA and WGPU must agree");
}

/// Assert every path refuses `program`, and that each message names the remedy.
fn assert_all_three_refuse(program: &Program, inputs: &[Vec<u8>], case: &str) -> [String; 3] {
    let reference = run_reference(program, inputs)
        .err()
        .unwrap_or_else(|| panic!("{case}: reference must refuse, it returned Ok"));
    let cuda = run_cuda(program, inputs)
        .err()
        .unwrap_or_else(|| panic!("{case}: CUDA must refuse, it returned Ok"));
    let wgpu = run_wgpu(program, inputs)
        .err()
        .unwrap_or_else(|| panic!("{case}: WGPU must refuse, it returned Ok"));
    [reference, cuda, wgpu]
}

// ---------------------------------------------------------------------------
// The headline differential, and its control.
// ---------------------------------------------------------------------------

/// A countless `read_write` result buffer returns identical bytes on all three paths.
///
/// Locks out the reported defect: WGPU answering `Ok` with a zero-length buffer
/// while CUDA and the reference returned the full result. If this regresses, a
/// caller who omits `.with_count(n)` silently receives nothing from WGPU with no
/// error to key off, and a wrong-but-successful pipeline ships.
#[test]
fn countless_read_write_output_agrees_on_exact_bytes_across_three_backends() {
    let program = xor_program(BufferDecl::read_write("out", 1, DataType::U32), N);
    let inputs = inputs_for(&program, N as usize * 4, N);
    assert_all_three_return(
        &program,
        &inputs,
        &expected_bytes(N),
        "countless read_write",
    );
}

/// A COUNTED `read_write` result buffer still works on all three paths.
///
/// Locks out a blanket rejection: refusing every countless writable buffer would
/// "fix" the bug above while breaking every correct caller. If this regresses,
/// the refusal has over-applied.
#[test]
fn counted_read_write_output_agrees_on_exact_bytes_across_three_backends() {
    let program = xor_program(
        BufferDecl::read_write("out", 1, DataType::U32).with_count(N),
        N,
    );
    let inputs = inputs_for(&program, N as usize * 4, N);
    assert_all_three_return(&program, &inputs, &expected_bytes(N), "counted read_write");
}

/// A countless `read_write` takes its element count from the bytes supplied.
///
/// Locks out a divergence in WHERE the count comes from. The reference and CUDA
/// both read it off the caller's bytes; WGPU read it off the declaration, which
/// is why it produced a different length for the same program. Under-supplying
/// the seed pins that rule: every path must return exactly the supplied element
/// count, not the count the kernel could have written.
#[test]
fn countless_read_write_takes_its_element_count_from_the_supplied_bytes() {
    let program = xor_program(BufferDecl::read_write("out", 1, DataType::U32), N);
    // Supply one element for a buffer the kernel would happily write 64 of.
    let inputs = inputs_for(&program, 4, N);
    assert_all_three_return(
        &program,
        &inputs,
        &expected_bytes(1),
        "countless read_write, one-element seed",
    );
}

/// Two countless `read_write` buffers are sized independently of each other.
///
/// Locks out a resolution that applies one buffer's length to all of them. If
/// this regresses, a program with two differently sized runtime buffers gets one
/// of them silently truncated or over-read.
#[test]
fn two_countless_read_write_buffers_are_sized_independently() {
    let program = Program::wrapped(
        vec![
            BufferDecl::read("a", 0, DataType::U32).with_count(8),
            BufferDecl::read_write("small", 1, DataType::U32),
            BufferDecl::read_write("large", 2, DataType::U32),
        ],
        [64, 1, 1],
        vec![
            Node::let_bind("idx", Expr::gid_x()),
            Node::if_then(
                Expr::lt(Expr::var("idx"), Expr::u32(2)),
                vec![Node::store("small", Expr::var("idx"), Expr::u32(0xAA))],
            ),
            Node::if_then(
                Expr::lt(Expr::var("idx"), Expr::u32(8)),
                vec![Node::store("large", Expr::var("idx"), Expr::u32(0xBB))],
            ),
        ],
    );
    let inputs = vec![a_bytes(8), vec![0u8; 2 * 4], vec![0u8; 8 * 4]];
    let small_expected: Vec<u8> = (0..2u32).flat_map(|_| 0xAAu32.to_le_bytes()).collect();
    let large_expected: Vec<u8> = (0..8u32).flat_map(|_| 0xBBu32.to_le_bytes()).collect();

    for (label, outputs) in [
        ("reference", run_reference(&program, &inputs)),
        ("cuda", run_cuda(&program, &inputs)),
        ("wgpu", run_wgpu(&program, &inputs)),
    ] {
        let outputs = outputs.unwrap_or_else(|error| panic!("{label} must dispatch: {error}"));
        assert_eq!(
            outputs[0], small_expected,
            "{label}: the 2-element runtime buffer must return exactly 8 bytes"
        );
        assert_eq!(
            outputs[1], large_expected,
            "{label}: the 8-element runtime buffer must return exactly 32 bytes"
        );
    }
}

// ---------------------------------------------------------------------------
// Boundaries. A zero-length output is the exact value the bug produced, so it
// must remain reachable ON PURPOSE and be distinguishable from the defect.
// ---------------------------------------------------------------------------

/// A single-element countless `read_write` returns exactly four bytes everywhere.
///
/// Locks out an off-by-one in the resolved layout at the smallest non-empty size,
/// where the `max(1)` word-count floor in the layout math could mask a wrong count.
#[test]
fn single_element_countless_read_write_agrees_across_three_backends() {
    let program = xor_program(BufferDecl::read_write("out", 1, DataType::U32), 1);
    let inputs = inputs_for(&program, 4, 1);
    assert_all_three_return(&program, &inputs, &expected_bytes(1), "one element");
}

/// An EMPTY seed yields an empty output, and that is a different fact from the bug.
///
/// Locks out the two being conflated. The defect returned zero bytes for a caller
/// who supplied 256; asking for zero elements must still return zero. The
/// discriminator is the pair: the same declaration returns 0 bytes for an empty
/// seed and exactly 256 for a 256-byte seed. If a regression makes everything
/// empty again, the second half of this test fails while the first still passes.
#[test]
fn zero_length_seed_returns_empty_while_a_full_seed_returns_every_byte() {
    let program = xor_program(BufferDecl::read_write("out", 1, DataType::U32), N);

    let empty_inputs = inputs_for(&program, 0, N);
    for (label, outputs) in [
        ("reference", run_reference(&program, &empty_inputs)),
        ("cuda", run_cuda(&program, &empty_inputs)),
        ("wgpu", run_wgpu(&program, &empty_inputs)),
    ] {
        let outputs = outputs.unwrap_or_else(|error| panic!("{label} must dispatch: {error}"));
        assert_eq!(
            outputs[0].len(),
            0,
            "{label}: a zero-element seed must return zero bytes, got {:?}",
            outputs[0]
        );
    }

    let full_inputs = inputs_for(&program, N as usize * 4, N);
    assert_all_three_return(
        &program,
        &full_inputs,
        &expected_bytes(N),
        "full seed after empty seed",
    );
}

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
        .err()
        .expect("WGPU must refuse an upload that would overrun the destination buffer");
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

// ---------------------------------------------------------------------------
// The launch grid. Same compile-time zero, second symptom.
// ---------------------------------------------------------------------------

/// A countless `read_write` launches enough threads to write every element.
///
/// Locks out the second defect the readback fix uncovered. Sizing the buffer
/// correctly is only half the request: the launch grid was inferred from the
/// same compile-time count, and a countless declaration reported zero words,
/// which rounds up to exactly one workgroup. The compiler also shrinks the
/// workgroup to 32 lanes when it believes the output is that small, so a 64
/// element dispatch ran 32 threads and returned a correctly sized buffer whose
/// second half was zeros under an `Ok`. That is the same silent wrong answer as
/// the empty readback, just harder to notice, so it is asserted the same way:
/// exact bytes, at lengths that straddle the 32 lane and 64 lane boundaries.
///
/// A single length cannot catch this. At 32 elements and below the truncated
/// launch happens to cover the whole buffer, so the bug is invisible; the
/// lengths above 32 are the ones that fail if it returns.
#[test]
fn countless_read_write_launches_enough_threads_for_every_element() {
    for len in [31u32, 32, 33, 48, 64, 65, 127, 128, 129, 256] {
        let program = xor_program(BufferDecl::read_write("out", 1, DataType::U32), len);
        let inputs = inputs_for(&program, len as usize * 4, len);
        assert_all_three_return(
            &program,
            &inputs,
            &expected_bytes(len),
            &format!("countless read_write, {len} elements"),
        );
    }
}

/// A caller who pins `grid_override` keeps exactly the launch they asked for.
///
/// Locks out the grid re-inference over-applying. Re-deriving the grid from the
/// resolved element count must happen only when the backend inferred that grid
/// in the first place. If it starts overwriting an explicit `grid_override`, a
/// caller who deliberately launched a partial or oversized grid silently gets a
/// different dispatch than the one they wrote.
#[test]
fn an_explicit_grid_override_is_not_replaced_by_the_resolved_element_count() {
    // 4096 elements against a single workgroup. No workgroup size this backend
    // picks can cover that in one launch, so the assertions below do not depend
    // on which lane count the compiler chose.
    const WIDE: u32 = 4096;
    let program = xor_program(BufferDecl::read_write("out", 1, DataType::U32), WIDE);
    let inputs = inputs_for(&program, WIDE as usize * 4, WIDE);
    let borrowed: Vec<&[u8]> = inputs.iter().map(Vec::as_slice).collect();
    let backend = vyre_driver_wgpu::WgpuBackend::acquire().expect("WGPU adapter required");

    let mut config = DispatchConfig::default();
    config.grid_override = Some([1, 1, 1]);
    let pinned = backend
        .dispatch_borrowed(&program, &borrowed, &config)
        .expect("a pinned grid must still dispatch");

    assert_eq!(
        pinned[0].len(),
        WIDE as usize * 4,
        "the readback length is set by the resolved element count, not by the pinned grid"
    );
    // Element 0 is inside the one workgroup that was launched: 0 ^ 0x0F == 15.
    assert_eq!(
        pinned[0][..4],
        [15, 0, 0, 0],
        "the workgroup that was launched must still write its own elements"
    );
    // The last element is far outside it and must keep the zero it was seeded
    // with. If the override were silently widened to cover all 4096 elements,
    // this would hold 4095 ^ 0x0F.
    assert_eq!(
        pinned[0][WIDE as usize * 4 - 4..],
        [0, 0, 0, 0],
        "a one-workgroup override must NOT have been widened into a full launch"
    );
    assert_ne!(
        pinned[0],
        expected_bytes(WIDE),
        "a pinned partial grid must not produce the full-launch result"
    );
}
