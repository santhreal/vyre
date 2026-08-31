//! Hashmap reference interpreter buffer-size contracts.
//!
//! Every buffer a caller hands the interpreter must be at least as large as
//! its declaration says, and the caller supplies exactly one Value per buffer
//! accepted by `vyre_reference::is_reference_input`. This file owns both
//! halves of that ABI: the arity and the per-buffer size.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_reference::{reference_eval, value::Value};

#[test]
fn huge_declared_buffer_size_returns_structured_error() {
    let program = Program::wrapped(
        vec![
            BufferDecl::read("huge", 0, DataType::Vec4U32).with_count(u32::MAX),
            BufferDecl::output("out", 1, DataType::Vec4U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::load("huge", Expr::u32(0)),
        )],
    );

    let error = reference_eval(&program, &[Value::from(vec![0u8; 16])])
        .expect_err("oversized declared input must not panic or allocate implicitly");
    let message = error.to_string();
    assert!(
        message.contains("huge") && message.contains("requires at least"),
        "buffer size diagnostic must name the buffer and required byte count, got: {message}"
    );
}

// ---------------------------------------------------------------------------
// Input arity and sizing
//
// The interpreter takes one Value per `is_reference_input` buffer, in
// `Program::buffers` order, and allocates every backend-allocated output
// itself. It once also accepted a vector sized to every non-workgroup buffer,
// treating the trailing entries as output initializers whose contents were
// discarded. That convention let a fixture carry a placeholder the artifact
// ABI rejects on a device, so the CPU oracle certified a program the device
// refused, which is the wrong place and the wrong run to find that out.
//
// What breaks if this regresses: a fixture that is one Value long passes every
// CPU lens, and the same program fails on a real GPU with an argument-count
// error that names neither the fixture nor the buffer.
// ---------------------------------------------------------------------------

/// Two read-only inputs and one backend-allocated output, all `count` u32s.
///
/// Mirrors the shape of the `vyre-libs::logical` element-wise ops whose guard
/// tests first exposed the fail-open. Built by hand rather than imported
/// because `vyre-reference` sits below `vyre-libs` in the dependency order.
fn elementwise_program(count: u32) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("a", 0, BufferAccess::ReadOnly, DataType::U32).with_count(count),
            BufferDecl::storage("b", 1, BufferAccess::ReadOnly, DataType::U32).with_count(count),
            BufferDecl::output("out", 2, DataType::U32).with_count(count),
        ],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::gid_x(),
            Expr::bitand(
                Expr::load("a", Expr::gid_x()),
                Expr::load("b", Expr::gid_x()),
            ),
        )],
    )
}

/// An undersized input is rejected, not silently replaced.
///
/// Four u32 elements declared (16 bytes), eight bytes supplied for `b`.
#[test]
fn an_undersized_input_is_an_error() {
    let error = reference_eval(
        &elementwise_program(4),
        &[Value::from(vec![0u8; 16]), Value::from(vec![0u8; 8])],
    )
    .expect_err("an undersized input must be rejected");

    let message = error.to_string();
    assert!(
        message.contains("`b`"),
        "the diagnostic must name the offending buffer, got: {message}"
    );
    assert!(
        message.contains("8 bytes") && message.contains("16 bytes"),
        "the diagnostic must state both the supplied and the required size, got: {message}"
    );
    assert!(
        message.contains("Fix:"),
        "the diagnostic must carry an actionable fix, got: {message}"
    );
}

/// The rejection is on SIZE: an exactly-sized input set runs.
///
/// Without this, a fix that simply refused every input Value would make the
/// test above pass while breaking every caller.
#[test]
fn an_exactly_sized_input_set_is_accepted() {
    let outputs = reference_eval(
        &elementwise_program(4),
        &[Value::from(vec![0xFFu8; 16]), Value::from(vec![0x0Fu8; 16])],
    )
    .expect("an exactly-sized input set must be accepted");

    assert_eq!(outputs.len(), 1, "the program declares one output buffer");
    assert_eq!(
        outputs[0].to_bytes(),
        vec![0x0Fu8; 16],
        "0xFFFFFFFF & 0x0F0F0F0F is 0x0F0F0F0F in every lane"
    );
}

/// A single missing byte is caught.
///
/// Boundary case. An off-by-one in the comparison (`<=` written for `<`) would
/// let the most common real undersize slip through while every coarse test
/// still passed.
#[test]
fn an_input_one_byte_short_is_an_error() {
    let error = reference_eval(
        &elementwise_program(4),
        &[Value::from(vec![0u8; 16]), Value::from(vec![0u8; 15])],
    )
    .expect_err("an input one byte short must be rejected");
    assert!(
        error.to_string().contains("15 bytes"),
        "the diagnostic must report the actual supplied size, got: {error}"
    );
}

/// An empty input is caught.
///
/// The degenerate end of the same range. `vec![]` is the placeholder a caller
/// reaches for once they have concluded the Value is ignored.
#[test]
fn an_empty_input_is_an_error() {
    let error = reference_eval(
        &elementwise_program(4),
        &[Value::from(vec![0u8; 16]), Value::from(Vec::<u8>::new())],
    )
    .expect_err("an empty input must be rejected");
    assert!(error.to_string().contains("`b`"), "got: {error}");
}

/// A Value for the backend-allocated output is refused, and the diagnostic
/// says which value went unused.
///
/// This is the placeholder convention the artifact ABI rejects on a device.
/// Accepting it here is what let a malformed fixture reach hardware.
#[test]
fn an_output_placeholder_is_refused() {
    let error = reference_eval(
        &elementwise_program(4),
        &[
            Value::from(vec![0xFFu8; 16]),
            Value::from(vec![0x0Fu8; 16]),
            Value::from(vec![0u8; 16]),
        ],
    )
    .expect_err("a Value for a backend-allocated output must be refused");
    let message = error.to_string();
    assert!(
        message.contains("unused input"),
        "the diagnostic must say the extra value went unused, got: {message}"
    );
    assert!(
        message.contains("is_reference_input"),
        "the diagnostic must name the predicate that selects inputs, got: {message}"
    );
}

/// The interpreter allocates its own output.
///
/// The size check must apply to supplied inputs only. If it leaked onto the
/// output it would demand a Value the ABI says the caller never supplies.
#[test]
fn the_interpreter_allocates_its_own_output() {
    let outputs = reference_eval(
        &elementwise_program(4),
        &[Value::from(vec![0xFFu8; 16]), Value::from(vec![0x0Fu8; 16])],
    )
    .expect("one Value per input buffer is the whole ABI");
    assert_eq!(outputs[0].to_bytes(), vec![0x0Fu8; 16]);
}

/// The required size tracks the declaration, it is not a fixed constant.
///
/// A hardcoded 16 would satisfy every test above. Sweeping the element count
/// proves the requirement is derived from `count * size_of(element)`.
#[test]
fn the_required_size_tracks_the_declared_element_count() {
    for count in [1u32, 2, 4, 16, 257] {
        let required = count as usize * 4;
        let error = reference_eval(
            &elementwise_program(count),
            &[
                Value::from(vec![0u8; required]),
                Value::from(vec![0u8; required - 1]),
            ],
        )
        .expect_err("an undersized input must be rejected at every element count");
        let message = error.to_string();
        assert!(
            message.contains(&format!("{required} bytes")),
            "count={count}: the diagnostic must state the declared size, got: {message}"
        );
        assert!(
            message.contains(&format!("{} bytes", required - 1)),
            "count={count}: the diagnostic must state the supplied size, got: {message}"
        );
    }
}
