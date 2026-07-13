//! Cross-backend PARITY for the shipping GPU-IR typedef annotator.
//!
//! `c11_annotate_typedef_names` is the GPU IR builder that production frontends
//! dispatch; `reference_c11_annotate_typedef_names` is the INDEPENDENT CPU oracle
//! (a hand-written Rust walker, not derived from the IR). Other suites use the
//! oracle as their source of truth for building typed VAST, but NOTHING pinned the
//! GPU builder itself against it, so a divergence between the IR annotator and the
//! oracle would go unnoticed (and the OpEntry conformance registry does not register
//! ANNOTATE_TYPEDEF_OP_ID; see BACKLOG.md PARITY-typedef-annotate-opentry).
//!
//! This differential runs the GPU builder through `reference_eval` and asserts its
//! annotated VAST is BYTE-IDENTICAL to the oracle's, over the canonical VAST built by
//! the (already-covered) `reference_c11_build_vast_nodes`. It exercises the exact
//! typedef-visibility resolution that matters: a typedef name reused as a type, a
//! reuse across a brace scope, and a control case with no typedef so the flags stay
//! clear. It also PROVES the witness encoding a future OpEntry registration needs
//! (GPU builder ← expanded u32-per-byte haystack; oracle ← raw source bytes).
#![cfg(feature = "c-parser")]
#![allow(deprecated)]

use vyre::ir::Expr;
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::parse::vast::{
    c11_annotate_typedef_names, reference_c11_annotate_typedef_names,
    reference_c11_build_vast_nodes,
};
use vyre_reference::value::Value;

const VAST_NODE_STRIDE_U32: usize = 10;
const VAST_TYPEDEF_FLAGS_FIELD: usize = 7;

/// One source byte per u32 word (the base builder's unpacked-haystack layout).
fn expanded_haystack(source: &[u8]) -> Vec<u8> {
    source
        .iter()
        .flat_map(|b| u32::from(*b).to_le_bytes())
        .collect()
}

fn unpack(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Run the GPU-IR annotator through `reference_eval`; return the annotated VAST bytes.
fn gpu_annotate(raw_vast: &[u8], source: &[u8], n: u32) -> Vec<u8> {
    let program = c11_annotate_typedef_names(
        "vast_nodes",
        "haystack",
        Expr::u32(source.len() as u32),
        Expr::u32(n),
        "out_annotated",
    );
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(raw_vast.to_vec()),
            Value::from(expanded_haystack(source)),
            Value::from(vec![0u8; raw_vast.len()]),
        ],
    )
    .expect("GPU annotate program must execute under reference_eval");
    outputs[0].to_bytes()
}

/// Assert the GPU-IR annotator and the CPU oracle agree byte-for-byte on `source`
/// with the given token stream; returns the annotated flag column for extra checks.
fn assert_parity(source: &[u8], tokens: &[(u32, u32, u32)]) -> Vec<u32> {
    let n = tokens.len() as u32;
    let tok_types: Vec<u32> = tokens.iter().map(|t| t.0).collect();
    let tok_starts: Vec<u32> = tokens.iter().map(|t| t.1).collect();
    let tok_lens: Vec<u32> = tokens.iter().map(|t| t.2).collect();

    let raw_vast = reference_c11_build_vast_nodes(&tok_types, &tok_starts, &tok_lens);
    let gpu = gpu_annotate(&raw_vast, source, n);
    let oracle = reference_c11_annotate_typedef_names(&raw_vast, source);

    assert_eq!(
        gpu,
        oracle,
        "GPU-IR `c11_annotate_typedef_names` must be byte-identical to the CPU oracle for `{}`",
        String::from_utf8_lossy(source)
    );

    let words = unpack(&gpu);
    (0..tokens.len())
        .map(|i| words[i * VAST_NODE_STRIDE_U32 + VAST_TYPEDEF_FLAGS_FIELD])
        .collect()
}

#[test]
fn gpu_annotate_matches_cpu_oracle_on_typedef_reuse() {
    // `typedef int foo; foo bar;`: foo is declared, then reused as a type for bar.
    let flags = assert_parity(
        b"typedef int foo; foo bar;",
        &[
            (TOK_TYPEDEF, 0, 7),
            (TOK_INT, 8, 3),
            (TOK_IDENTIFIER, 12, 3), // foo (typedef declarator)
            (TOK_SEMICOLON, 15, 1),
            (TOK_IDENTIFIER, 17, 3), // foo (type use)
            (TOK_IDENTIFIER, 21, 3), // bar (ordinary declarator)
            (TOK_SEMICOLON, 24, 1),
        ],
    );
    // Beyond byte-parity: the annotation must be NON-TRIVIAL (some flag set), else a
    // degenerate all-zero agreement would pass vacuously.
    assert!(
        flags.iter().any(|&f| f != 0),
        "typedef reuse must set at least one typedef flag, got {flags:?}"
    );
}

#[test]
fn gpu_annotate_matches_cpu_oracle_on_braced_scope_reuse() {
    // `typedef int foo; { foo bar; }`: the reuse happens inside a nested brace scope,
    // exercising the scope walker in both the GPU builder and the oracle.
    assert_parity(
        b"typedef int foo; { foo bar; }",
        &[
            (TOK_TYPEDEF, 0, 7),
            (TOK_INT, 8, 3),
            (TOK_IDENTIFIER, 12, 3), // foo (declared)
            (TOK_SEMICOLON, 15, 1),
            (TOK_LBRACE, 17, 1),
            (TOK_IDENTIFIER, 19, 3), // foo (type use in inner scope)
            (TOK_IDENTIFIER, 23, 3), // bar (declarator)
            (TOK_SEMICOLON, 26, 1),
            (TOK_RBRACE, 28, 1),
        ],
    );
}

#[test]
fn gpu_annotate_resolves_chained_typedef_correctly() {
    // `typedef int A; typedef A B; B x;`: A is a typedef, B is typedef'd FROM A (a
    // typedef-name used as the type in a NEW typedef), then B types the variable x.
    //
    // The CPU oracle `reference_c11_annotate_typedef_names` USED to be wrong here
    // (marked B ORDINARY and missed B-as-a-type), fixed in declarations.rs
    // (BACKLOG.md ORACLE-chained-typedef-bug). So this asserts FULL GPU==oracle byte
    // parity AND pins the exact correct per-node flags. B is a typedef declarator,
    // then a visible type for x.
    const TYPEDEF_DECLARATOR: u32 = 1 << 1; // 2
    const ORDINARY_DECLARATOR: u32 = 1 << 2; // 4
    const VISIBLE_TYPEDEF_NAME: u32 = 1; // 1
    let flags = assert_parity(
        b"typedef int A; typedef A B; B x;",
        &[
            (TOK_TYPEDEF, 0, 7),
            (TOK_INT, 8, 3),
            (TOK_IDENTIFIER, 12, 1), // node2: A (typedef declarator)
            (TOK_SEMICOLON, 13, 1),
            (TOK_TYPEDEF, 15, 7),
            (TOK_IDENTIFIER, 23, 1), // node5: A (type use in second typedef)
            (TOK_IDENTIFIER, 25, 1), // node6: B (typedef declarator, its type A is a typedef-name)
            (TOK_SEMICOLON, 26, 1),
            (TOK_IDENTIFIER, 28, 1), // node8: B (type use)
            (TOK_IDENTIFIER, 30, 1), // node9: x (ordinary declarator)
            (TOK_SEMICOLON, 31, 1),
        ],
    );
    assert_eq!(
        flags,
        vec![
            0,
            0,
            TYPEDEF_DECLARATOR,   // node2 A
            0,
            0,
            VISIBLE_TYPEDEF_NAME, // node5 A used as type
            TYPEDEF_DECLARATOR,   // node6 B is itself a typedef (type = typedef-name A)
            0,
            VISIBLE_TYPEDEF_NAME, // node8 B used as type
            ORDINARY_DECLARATOR,  // node9 x
            0,
        ],
        "GPU annotator must resolve chained typedefs: B is a typedef declarator, then a visible type for x"
    );
}

#[test]
fn gpu_annotate_matches_cpu_oracle_on_typedef_pointer_declaration() {
    // `typedef int T; T *p;`: a pointer declarator whose type is a typedef-name, with
    // a `*` between the type and the declarator (the declaration-prefix scan must skip
    // the star and still resolve T as the type in both the GPU builder and the oracle).
    assert_parity(
        b"typedef int T; T *p;",
        &[
            (TOK_TYPEDEF, 0, 7),
            (TOK_INT, 8, 3),
            (TOK_IDENTIFIER, 12, 1), // T (typedef declarator)
            (TOK_SEMICOLON, 13, 1),
            (TOK_IDENTIFIER, 15, 1), // T (type use)
            (TOK_STAR, 17, 1),
            (TOK_IDENTIFIER, 18, 1), // p (pointer declarator)
            (TOK_SEMICOLON, 19, 1),
        ],
    );
}

#[test]
fn gpu_annotate_matches_cpu_oracle_on_multiple_declarators() {
    // `typedef int T; T a, b;`: one typedef, then TWO declarators sharing the type T
    // across a comma. Both a and b must annotate identically as ordinary declarators
    // in GPU and oracle.
    assert_parity(
        b"typedef int T; T a, b;",
        &[
            (TOK_TYPEDEF, 0, 7),
            (TOK_INT, 8, 3),
            (TOK_IDENTIFIER, 12, 1), // T (typedef declarator)
            (TOK_SEMICOLON, 13, 1),
            (TOK_IDENTIFIER, 15, 1), // T (type use)
            (TOK_IDENTIFIER, 17, 1), // a (declarator)
            (TOK_COMMA, 18, 1),
            (TOK_IDENTIFIER, 20, 1), // b (declarator)
            (TOK_SEMICOLON, 21, 1),
        ],
    );
}

#[test]
fn gpu_annotate_matches_cpu_oracle_on_two_independent_typedefs() {
    // `typedef int A; typedef int B; A x; B y;`: two independent typedefs each used
    // in its own declaration; the annotator must resolve A and B independently in both
    // the GPU builder and the oracle without cross-contaminating the two chains.
    assert_parity(
        b"typedef int A; typedef int B; A x; B y;",
        &[
            (TOK_TYPEDEF, 0, 7),
            (TOK_INT, 8, 3),
            (TOK_IDENTIFIER, 12, 1), // A (typedef declarator)
            (TOK_SEMICOLON, 13, 1),
            (TOK_TYPEDEF, 15, 7),
            (TOK_INT, 23, 3),
            (TOK_IDENTIFIER, 27, 1), // B (typedef declarator)
            (TOK_SEMICOLON, 28, 1),
            (TOK_IDENTIFIER, 30, 1), // A (type use)
            (TOK_IDENTIFIER, 32, 1), // x (declarator)
            (TOK_SEMICOLON, 33, 1),
            (TOK_IDENTIFIER, 35, 1), // B (type use)
            (TOK_IDENTIFIER, 37, 1), // y (declarator)
            (TOK_SEMICOLON, 38, 1),
        ],
    );
}

#[test]
fn gpu_annotate_matches_cpu_oracle_on_plain_declaration() {
    // `int x;`: no typedef anywhere; the flag column must stay clear in BOTH, a
    // negative control proving the parity holds when nothing should be annotated.
    let flags = assert_parity(
        b"int x;",
        &[
            (TOK_INT, 0, 3),
            (TOK_IDENTIFIER, 4, 1), // x (ordinary declarator, not a typedef)
            (TOK_SEMICOLON, 5, 1),
        ],
    );
    assert!(
        flags.iter().all(|&f| f & 1 == 0),
        "no identifier is a VISIBLE_TYPEDEF_NAME in a plain declaration, got {flags:?}"
    );
}
