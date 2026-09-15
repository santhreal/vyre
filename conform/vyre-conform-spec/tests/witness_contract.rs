//! Contract tests for witness-set determinism, edge-case coverage,
//! and Program wire-format fingerprint stability.

use vyre_conform_spec::{U32Witness, WitnessSet};
use vyre_foundation::ir::{BufferDecl, Expr, Node, Program};
use vyre_spec::DataType;

#[test]
fn u32_witness_is_deterministic() {
    let a = U32Witness::enumerate();
    let b = U32Witness::enumerate();
    assert_eq!(
        a, b,
        "U32Witness::enumerate must be deterministic across calls"
    );
}

#[test]
fn u32_witness_uses_the_stable_semantic_data_type() {
    assert_eq!(<U32Witness as WitnessSet>::DATA_TYPE, DataType::U32);
}

#[test]
fn u32_witness_canonical_fingerprint_is_stable_across_calls() {
    let a = U32Witness::fingerprint_canonical();
    let b = U32Witness::fingerprint_canonical();
    assert_eq!(
        a, b,
        "U32Witness::fingerprint_canonical must be stable across calls"
    );
}

#[test]
fn u32_witness_contains_critical_edge_cases() {
    let w = U32Witness::enumerate();
    assert!(w.contains(&0), "witness set must contain 0");
    assert!(w.contains(&1), "witness set must contain 1");
    assert!(w.contains(&u32::MAX), "witness set must contain u32::MAX");
    assert!(
        w.contains(&(u32::MAX - 1)),
        "witness set must contain u32::MAX - 1"
    );
    assert!(
        w.contains(&0x8000_0000),
        "witness set must contain the sign-bit boundary 0x8000_0000"
    );
}

/// The program every fingerprint and wire-format test below builds.
///
/// Each call constructs a separate value with cold memos, which is what makes
/// a recomputation claim testable: `Program::fingerprint` memoizes into a
/// `OnceLock` and `Program::clone` carries an already-computed fingerprint
/// across, so two reads of one value compare a cached scalar with itself.
fn sample_program() -> Program {
    Program::wrapped(
        vec![BufferDecl::read_write("out", 0, DataType::U32)],
        [64, 1, 1],
        vec![
            Node::let_bind("idx", Expr::gid_x()),
            Node::store("out", Expr::var("idx"), Expr::u32(42)),
            Node::Return,
        ],
    )
}

#[test]
fn program_fingerprint_stable_across_clones() {
    let program = sample_program();
    let clone = program.clone();
    assert_eq!(
        program.fingerprint(),
        clone.fingerprint(),
        "fingerprint must be stable across clones"
    );
}

#[test]
fn program_fingerprint_stable_across_recomputation() {
    // Two separately constructed programs, so both sides hash from scratch. A
    // fingerprint that reached a map iteration order, a pointer address, or
    // uninitialized padding in the wire encoding separates them here and
    // cannot separate two reads of one memo.
    let first = sample_program();
    let second = sample_program();
    assert_eq!(
        first.fingerprint(),
        second.fingerprint(),
        "fingerprint must be stable across repeated computation"
    );
}

#[test]
fn program_wire_bytes_stable_across_serializations() {
    let program = sample_program();
    let bytes1 = program.canonical_wire_bytes().unwrap();
    let bytes2 = program.canonical_wire_bytes().unwrap();
    assert_eq!(
        bytes1, bytes2,
        "wire-format bytes must be identical across serializations"
    );
}

#[test]
fn program_wire_bytes_match_fingerprint_derivation() {
    let program = sample_program();
    let bytes = program.canonical_wire_bytes().unwrap();
    let expected_fp = *blake3::hash(&bytes).as_bytes();
    let actual_fp = program.fingerprint();
    assert_eq!(
        expected_fp, actual_fp,
        "fingerprint must equal blake3 hash of canonical wire bytes"
    );
}
