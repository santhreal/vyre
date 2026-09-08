//! Parity for the public u32 elementwise builders against the reference oracle.
//!
//! `u32_elementwise_unary` and `u32_elementwise_binary` are the infallible
//! wrappers: they build the program and, when the tensor refs do not agree,
//! substitute a trap program instead of returning the error. Both halves are
//! asserted. The success half is checked against `reference_eval` rather than
//! against the shape of the program, because a builder that emits a well formed
//! program computing the wrong thing is the defect these wrap.
//!
//! The mismatch half is what the wrapper adds over `try_*`, and it is the half
//! a shape assertion cannot see: the trap has to survive to the point of
//! evaluation, so it is evaluated and required to fail.
#![cfg(feature = "builder")]

use vyre_foundation::ir::Expr;
use vyre_libs::elementwise::{u32_elementwise_binary, u32_elementwise_unary};
use vyre_reference::value::Value;

const OP_UNARY: &str = "test::elementwise_u32_unary";
const OP_BINARY: &str = "test::elementwise_u32_binary";

fn bytes(words: &[u32]) -> Value {
    Value::from(vyre_primitives::wire::pack_u32_slice(words))
}

fn words(value: &Value) -> Vec<u32> {
    value
        .to_bytes()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().expect("a u32 lane is four bytes")))
        .collect()
}

#[test]
fn unary_builder_matches_the_oracle_lane_for_lane() {
    let input: Vec<u32> = (0..64u32).map(|i| i.wrapping_mul(2_654_435_761)).collect();
    let size = u32::try_from(input.len()).expect("lane count fits in u32");
    let program = u32_elementwise_unary(OP_UNARY, "input", "out", size, |x| {
        Expr::add(x, Expr::u32(7))
    });

    let outputs = vyre_reference::reference_eval(&program, &[bytes(&input)])
        .expect("the unary elementwise program must evaluate");

    let expected: Vec<u32> = input.iter().map(|x| x.wrapping_add(7)).collect();
    assert_eq!(words(&outputs[0]), expected);
}

#[test]
fn binary_builder_matches_the_oracle_lane_for_lane() {
    let a: Vec<u32> = (0..64u32).map(|i| i.wrapping_mul(2_246_822_519)).collect();
    let b: Vec<u32> = (0..64u32).map(|i| i.wrapping_mul(3_266_489_917)).collect();
    let size = u32::try_from(a.len()).expect("lane count fits in u32");
    let program = u32_elementwise_binary(OP_BINARY, "a", "b", "out", size, Expr::add);

    let outputs = vyre_reference::reference_eval(&program, &[bytes(&a), bytes(&b)])
        .expect("the binary elementwise program must evaluate");

    let expected: Vec<u32> = a.iter().zip(&b).map(|(x, y)| x.wrapping_add(*y)).collect();
    assert_eq!(words(&outputs[0]), expected);
}

/// A zero-lane request is a program, not a panic, and it writes nothing.
#[test]
fn a_zero_lane_request_evaluates_and_writes_no_lane() {
    let program = u32_elementwise_unary(OP_UNARY, "input", "out", 0, |x| x);
    let outputs = vyre_reference::reference_eval(&program, &[bytes(&[])])
        .expect("a zero-lane elementwise program must evaluate");
    assert!(words(&outputs[0]).is_empty());
}

/// The infallible wrapper substitutes a trap, and the trap must actually trap.
///
/// The three operands are given one name, which `check_unique_names` rejects
/// and this wrapper swallows in favour of a trap program. Asserting the program
/// merely exists would pass against a wrapper that returned an empty program,
/// so it is evaluated and required to refuse. The message is asserted too: a
/// refusal for some unrelated reason, an operand count among them, would
/// otherwise read as this contract holding.
#[test]
fn a_mismatched_operand_yields_a_program_that_refuses_to_evaluate() {
    let program = u32_elementwise_binary(OP_BINARY, "a", "a", "a", 4, Expr::add);
    let error = vyre_reference::reference_eval(&program, &[])
        .expect_err("a builder that could not honour its operands must hand back a refusal");
    let message = error.to_string();
    assert!(
        message.contains("Fix:"),
        "the trap must carry the builder's own diagnosis, not an unrelated refusal: {message}"
    );
}
