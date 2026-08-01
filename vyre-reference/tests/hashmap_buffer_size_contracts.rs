//! Hashmap reference interpreter buffer-size contracts.
//!
//! Every buffer a caller hands the interpreter must be at least as large as
//! its declaration says. This file owns that contract for both ways a Value
//! reaches the interpreter: an ordinary input, and a legacy output
//! initializer.

use vyre::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
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

    let error = reference_eval(
        &program,
        &[Value::from(vec![0u8; 16]), Value::from(vec![0u8; 4])],
    )
    .expect_err("oversized declared input must not panic or allocate implicitly");
    let message = error.to_string();
    assert!(
        message.contains("huge") && message.contains("requires at least"),
        "buffer size diagnostic must name the buffer and required byte count, got: {message}"
    );
}

// ---------------------------------------------------------------------------
// Legacy output-initializer sizing
//
// The interpreter accepts two input conventions. In LOGICAL mode a caller
// passes one Value per non-output buffer and the interpreter allocates the
// outputs itself. In LEGACY mode a caller passes one Value per non-workgroup
// buffer, outputs included, and those output Values are placeholders: the
// interpreter still zero-fills backend-allocated outputs, so their CONTENTS
// are unused.
//
// Their SIZE was unused too, and that was the bug. The legacy placeholder was
// bound to `_legacy_output_initializer` and dropped without a single check, so
// a caller who passed a 12-byte output buffer for a 16-byte declaration got a
// full 16-byte result and no diagnostic at all. That is a Law 10 fail-open:
// the caller's contract was violated and the interpreter silently substituted
// its own buffer. Five `vyre-libs` guard tests (`and`/`or`/`xor`/`nand`/`nor`
// output-size mismatch) asserted the error and had been failing unnoticed.
//
// What breaks if this regresses: an undersized output goes undetected on the
// CPU reference, then the same program on a real GPU writes past the caller's
// allocation. The reference oracle exists to catch exactly that before it
// reaches a device.
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

/// An undersized legacy output initializer is rejected, not silently replaced.
///
/// This is the exact case the five `logical_should_panic` guards assert: four
/// u32 elements declared (16 bytes), twelve bytes supplied.
#[test]
fn an_undersized_legacy_output_initializer_is_an_error() {
    let error = reference_eval(
        &elementwise_program(4),
        &[
            Value::from(vec![0u8; 16]),
            Value::from(vec![0u8; 16]),
            Value::from(vec![0u8; 12]),
        ],
    )
    .expect_err("an undersized legacy output initializer must be rejected");

    let message = error.to_string();
    assert!(
        message.contains("`out`"),
        "the diagnostic must name the offending buffer, got: {message}"
    );
    assert!(
        message.contains("12 bytes") && message.contains("16 bytes"),
        "the diagnostic must state both the supplied and the required size, got: {message}"
    );
    assert!(
        message.contains("Fix:"),
        "the diagnostic must carry an actionable fix, got: {message}"
    );
}

/// The rejection is on SIZE, not on the mode: an exactly-sized initializer runs.
///
/// Without this, a fix that simply refused every legacy output initializer
/// would make the test above pass while breaking every existing legacy caller.
#[test]
fn an_exactly_sized_legacy_output_initializer_is_accepted() {
    let outputs = reference_eval(
        &elementwise_program(4),
        &[
            Value::from(vec![0xFFu8; 16]),
            Value::from(vec![0x0Fu8; 16]),
            Value::from(vec![0u8; 16]),
        ],
    )
    .expect("an exactly-sized legacy output initializer must be accepted");

    assert_eq!(outputs.len(), 1, "the program declares one output buffer");
    assert_eq!(
        outputs[0].to_bytes(),
        vec![0x0Fu8; 16],
        "0xFFFFFFFF & 0x0F0F0F0F is 0x0F0F0F0F in every lane"
    );
}

/// An oversized legacy output initializer is accepted.
///
/// The contract is a MINIMUM. A caller who reuses one large scratch Value
/// across several programs is not doing anything wrong, and rejecting that
/// would be a gratuitous break.
#[test]
fn an_oversized_legacy_output_initializer_is_accepted() {
    let outputs = reference_eval(
        &elementwise_program(4),
        &[
            Value::from(vec![0xFFu8; 16]),
            Value::from(vec![0x0Fu8; 16]),
            Value::from(vec![0u8; 64]),
        ],
    )
    .expect("an oversized legacy output initializer must be accepted");
    assert_eq!(outputs[0].to_bytes(), vec![0x0Fu8; 16]);
}

/// A single missing byte is caught.
///
/// Boundary case. An off-by-one in the comparison (`<=` written for `<`) would
/// let the most common real undersize slip through while every coarse test
/// still passed.
#[test]
fn a_legacy_output_initializer_one_byte_short_is_an_error() {
    let error = reference_eval(
        &elementwise_program(4),
        &[
            Value::from(vec![0u8; 16]),
            Value::from(vec![0u8; 16]),
            Value::from(vec![0u8; 15]),
        ],
    )
    .expect_err("a legacy output initializer one byte short must be rejected");
    assert!(
        error.to_string().contains("15 bytes"),
        "the diagnostic must report the actual supplied size, got: {error}"
    );
}

/// An empty legacy output initializer is caught.
///
/// The degenerate end of the same range. `vec![]` is the placeholder a caller
/// reaches for once they have concluded the Value is ignored, which is
/// precisely the misunderstanding the old code encouraged.
#[test]
fn an_empty_legacy_output_initializer_is_an_error() {
    let error = reference_eval(
        &elementwise_program(4),
        &[
            Value::from(vec![0u8; 16]),
            Value::from(vec![0u8; 16]),
            Value::from(Vec::<u8>::new()),
        ],
    )
    .expect_err("an empty legacy output initializer must be rejected");
    assert!(error.to_string().contains("`out`"), "got: {error}");
}

/// Logical mode is unaffected: no output Value is passed, so none is checked.
///
/// The size check must live on the legacy branch only. If it leaked onto the
/// logical path it would demand an output Value that the convention says the
/// caller never supplies.
#[test]
fn logical_mode_still_allocates_its_own_output() {
    let outputs = reference_eval(
        &elementwise_program(4),
        &[Value::from(vec![0xFFu8; 16]), Value::from(vec![0x0Fu8; 16])],
    )
    .expect("logical mode passes one Value per input buffer only");
    assert_eq!(outputs[0].to_bytes(), vec![0x0Fu8; 16]);
}

/// An undersized ORDINARY input is still rejected with the same wording.
///
/// The two paths now share one diagnostic helper. This pins that they report
/// the same contract the same way, so a reader who has seen one message can
/// recognise the other.
#[test]
fn an_undersized_ordinary_input_reports_the_same_contract() {
    let error = reference_eval(
        &elementwise_program(4),
        &[
            Value::from(vec![0u8; 16]),
            Value::from(vec![0u8; 8]),
            Value::from(vec![0u8; 16]),
        ],
    )
    .expect_err("an undersized ordinary input must be rejected");

    let message = error.to_string();
    assert!(
        message.contains("`b`") && message.contains("8 bytes") && message.contains("16 bytes"),
        "input and output undersize must read identically, got: {message}"
    );
    assert!(message.contains("Fix:"), "got: {message}");
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
                Value::from(vec![0u8; required]),
                Value::from(vec![0u8; required - 1]),
            ],
        )
        .expect_err("an undersized output must be rejected at every element count");
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
