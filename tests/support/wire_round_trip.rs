//! The one owner of the VIR0 wire round-trip oracle.
//!
//! Three suites asserted the same thing about a program's wire form and each
//! spelled it out again: the terminal variant suite in `vyre-foundation`
//! checked encode and decode, the hostile sweep in `vyre-spec` checked byte
//! identity across a second and a third encode, and the atomic tag sweep beside
//! it checked byte identity but not the decoded value. Three readings of one
//! contract means the weakest of them decides what a variant is actually held
//! to.
//!
//! What the round trip must preserve is here. Which programs are worth running
//! through it stays with each suite.

#![allow(dead_code)]

use vyre_foundation::ir::Program;

/// Encode `program`, decode it, and encode the decoded value twice more.
///
/// Holds the full contract in one place: the bytes are non-empty, the decoded
/// program equals the original, and the canonical form is a fixed point, so a
/// re-encode and a decode of that re-encode both reproduce the first bytes
/// exactly. `case` names the program in every failure message.
///
/// Returns the canonical bytes, for a caller that pins a length or a prefix.
pub(crate) fn assert_canonical_wire_round_trip(program: &Program, case: &str) -> Vec<u8> {
    let encoded = program
        .to_wire()
        .unwrap_or_else(|error| panic!("Fix: wire case {case} must encode: {error}"));
    assert!(
        !encoded.is_empty(),
        "Fix: wire case {case} encoded to no bytes"
    );

    let decoded = Program::from_wire(&encoded)
        .unwrap_or_else(|error| panic!("Fix: wire case {case} must decode: {error}"));
    assert_eq!(
        &decoded, program,
        "Fix: wire case {case} decoded to a different program"
    );

    let reencoded = decoded
        .to_wire()
        .unwrap_or_else(|error| panic!("Fix: wire case {case} must re-encode: {error}"));
    assert_eq!(
        reencoded, encoded,
        "Fix: wire case {case} canonical bytes drifted after one round trip"
    );

    let redecoded = Program::from_wire(&reencoded)
        .unwrap_or_else(|error| panic!("Fix: wire case {case} must decode canonical bytes: {error}"));
    let third = redecoded
        .to_wire()
        .unwrap_or_else(|error| panic!("Fix: wire case {case} must triple-encode: {error}"));
    assert_eq!(
        third, encoded,
        "Fix: wire case {case} lost byte identity on the second canonical encode"
    );

    encoded
}
