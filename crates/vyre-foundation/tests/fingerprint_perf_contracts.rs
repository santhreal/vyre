//! Pipeline fingerprint + VSA content-addressing performance tests.
//!
//! The pipeline fingerprint is the content-addressed key that prevents
//! redundant GPU pipeline compilations. These tests verify:
//! 1. Determinism  -  same input always produces same fingerprint
//! 2. Sensitivity  -  different programs produce different fingerprints
//! 3. Canonicalization  -  semantically equivalent programs (differing only
//!    in buffer declaration order) share the same fingerprint
//! 4. Stability across optimization  -  optimize(P) has a stable fingerprint

use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::optimizer::optimize;
use vyre_foundation::optimizer::{fingerprint_program, pipeline_fingerprint_bytes};
use vyre_test_support::pass_programs::sum_of_two_loads;

// ── Determinism ──────────────────────────────────────────────────────
//
// Every case below fingerprints two separately built copies of one program.
// Asking one `Program` twice proves nothing: `Program::fingerprint` memoizes
// into a `OnceLock`, so the second call reads the first call's answer and the
// comparison holds however the hash was computed. Two values each compute
// once, so a hash that reached a map iteration order, a pointer address, or
// uninitialized padding in the wire encoding separates them.

fn constant_folded_store() -> Program {
    Program::wrapped(
        vec![BufferDecl::read_write("out", 0, DataType::U32)],
        [64, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::add(Expr::u32(1), Expr::u32(2)),
        )],
    )
}

fn literal_store() -> Program {
    Program::wrapped(
        vec![BufferDecl::read_write("out", 0, DataType::U32)],
        [64, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(42))],
    )
}

#[test]
fn fingerprint_is_deterministic() {
    let a = fingerprint_program(&constant_folded_store());
    let b = fingerprint_program(&constant_folded_store());
    assert_eq!(a, b, "Fix: fingerprint_program must be deterministic.");
}

#[test]
fn pipeline_fingerprint_bytes_is_deterministic() {
    let a = pipeline_fingerprint_bytes(&literal_store());
    let b = pipeline_fingerprint_bytes(&literal_store());
    assert_eq!(
        a, b,
        "Fix: pipeline_fingerprint_bytes must be deterministic."
    );
}

/// The memoized answer is the answer a fresh computation gives.
///
/// A `OnceLock` filled from somewhere other than the canonical wire hash, or
/// filled before a mutation the program later applies, would serve a
/// fingerprint no rebuild reproduces, and every content-addressed cache keyed
/// on it would hand back another program's artifact.
#[test]
fn a_memoized_fingerprint_matches_a_fresh_one() {
    let warmed = literal_store();
    let memoized = pipeline_fingerprint_bytes(&warmed);
    assert_eq!(memoized, pipeline_fingerprint_bytes(&warmed));
    assert_eq!(memoized, pipeline_fingerprint_bytes(&literal_store()));
}

// ── Sensitivity ──────────────────────────────────────────────────────

#[test]
fn different_programs_have_different_fingerprints() {
    let p1 = Program::wrapped(
        vec![BufferDecl::read_write("out", 0, DataType::U32)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(42))],
    );
    let p2 = Program::wrapped(
        vec![BufferDecl::read_write("out", 0, DataType::U32)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(99))],
    );
    assert_ne!(
        fingerprint_program(&p1),
        fingerprint_program(&p2),
        "Fix: different programs must produce different fingerprints."
    );
}

#[test]
fn different_workgroup_sizes_produce_different_fingerprints() {
    let p1 = Program::wrapped(
        vec![BufferDecl::read_write("out", 0, DataType::U32)],
        [64, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
    );
    let p2 = Program::wrapped(
        vec![BufferDecl::read_write("out", 0, DataType::U32)],
        [128, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
    );
    assert_ne!(
        fingerprint_program(&p1),
        fingerprint_program(&p2),
        "Fix: different workgroup sizes must produce different fingerprints."
    );
}

// ── Canonicalization ─────────────────────────────────────────────────

#[test]
fn buffer_declaration_order_does_not_affect_pipeline_fingerprint() {
    // Same program, different buffer declaration order.
    let p1 = sum_of_two_loads(vec![
        BufferDecl::read("a", 0, DataType::U32),
        BufferDecl::read("b", 1, DataType::U32),
        BufferDecl::read_write("out", 2, DataType::U32),
    ]);
    let p2 = sum_of_two_loads(vec![
        BufferDecl::read_write("out", 2, DataType::U32),
        BufferDecl::read("b", 1, DataType::U32),
        BufferDecl::read("a", 0, DataType::U32),
    ]);
    assert_eq!(
        pipeline_fingerprint_bytes(&p1),
        pipeline_fingerprint_bytes(&p2),
        "Fix: buffer declaration order must not affect pipeline fingerprint (canonicalization)."
    );
}

// ── Commutative canonicalization ─────────────────────────────────────

#[test]
fn commutative_operand_order_canonicalized() {
    // lit + var vs var + lit should canonicalize to the same form
    let p1 = Program::wrapped(
        vec![BufferDecl::read_write("out", 0, DataType::U32)],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::add(Expr::u32(5), Expr::var("x")),
        )],
    );
    let p2 = Program::wrapped(
        vec![BufferDecl::read_write("out", 0, DataType::U32)],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::add(Expr::var("x"), Expr::u32(5)),
        )],
    );
    assert_eq!(
        pipeline_fingerprint_bytes(&p1),
        pipeline_fingerprint_bytes(&p2),
        "Fix: commutative operand reordering must be canonicalized for pipeline fingerprint."
    );
}

// ── Optimization stability ──────────────────────────────────────────

#[test]
fn optimized_program_fingerprint_is_stable() {
    let make = || {
        Program::wrapped(
            vec![BufferDecl::read_write("out", 0, DataType::U32)],
            [64, 1, 1],
            vec![
                Node::let_bind("x", Expr::add(Expr::u32(10), Expr::u32(20))),
                Node::store("out", Expr::u32(0), Expr::var("x")),
            ],
        )
    };

    let opt1 = optimize(make()).expect("registered optimizer must converge");
    let opt2 = optimize(make()).expect("registered optimizer must converge");
    assert_eq!(
        fingerprint_program(&opt1),
        fingerprint_program(&opt2),
        "Fix: optimized programs from identical sources must have identical fingerprints."
    );
}

// ── Full 32-byte fingerprint is non-trivial ──────────────────────────

#[test]
fn pipeline_fingerprint_is_32_bytes() {
    let program = Program::wrapped(
        vec![BufferDecl::read_write("out", 0, DataType::U32)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
    );
    let fp = pipeline_fingerprint_bytes(&program);
    assert_eq!(fp.len(), 32);
    // Not all zeros  -  actually hashed
    assert!(
        fp.iter().any(|&b| b != 0),
        "Fix: pipeline fingerprint must not be all zeros."
    );
}

// ── Empty program has a valid fingerprint ────────────────────────────

#[test]
fn empty_program_has_valid_fingerprint() {
    let program = Program::wrapped(Vec::new(), [1, 1, 1], Vec::new());
    let fp = fingerprint_program(&program);
    assert!(
        fp != 0,
        "Fix: empty program must still have a non-zero fingerprint."
    );
}
