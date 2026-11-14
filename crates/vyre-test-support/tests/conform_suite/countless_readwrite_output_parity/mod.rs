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

use vyre::ir::{BufferDecl, DataType, Expr, Node, Program};

mod counted;
mod harness;
mod launch_grid;
mod refusals;

use self::harness::{
    a_bytes, assert_all_three_return, expected_bytes, inputs_for, run_cuda, run_reference,
    run_wgpu, xor_program, N,
};

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
