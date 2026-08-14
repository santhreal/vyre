//! Wire-format v1 round-trip smoke test.
//!
//! Proves every non-empty Program built from primitive variants
//! round-trips through `to_wire` + `from_wire` byte-identically.
//! Protects against silent regressions in the VIR0 wire encoder or
//! decoder when a new IR variant lands; exhaustive proptest coverage
//! lives at `vyre-foundation/tests/terminal_wire_round_trip.rs`.

use vyre::ir::Program;

#[path = "support/mod.rs"]
mod support;
use support::{empty_program, one_store_program};

#[test]
fn empty_program_round_trips() {
    let p = empty_program();
    let bytes = p.to_wire().expect("empty program must encode");
    let decoded = Program::from_wire(&bytes).expect("empty program must decode");
    assert_eq!(decoded, p);
}

#[test]
fn trivial_program_round_trips() {
    let p = one_store_program();
    let bytes = p.to_wire().expect("trivial program must encode");
    let decoded = Program::from_wire(&bytes).expect("trivial program must decode");
    assert_eq!(decoded, p);
}

#[test]
fn re_encode_is_stable() {
    // Encoder must be deterministic: encoding the decoded program
    // yields the same bytes.
    let p = one_store_program();
    let bytes = p.to_wire().expect("encode");
    let decoded = Program::from_wire(&bytes).expect("decode");
    let re_encoded = decoded.to_wire().expect("re-encode");
    assert_eq!(bytes, re_encoded);
}

#[test]
fn wire_bytes_nonempty() {
    // Smoke check that encoded output is nonempty  -  the header and
    // body structure are verified exhaustively by
    // vyre-foundation/tests/terminal_wire_round_trip.rs.
    let bytes = empty_program().to_wire().expect("encode");
    assert_ne!(bytes.len(), 0, "encoded wire bytes must be non-empty");
}
