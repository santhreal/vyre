//! Cross-backend PARITY for the shipping GPU-IR typedef annotator.
//!
//! `c11_annotate_typedef_names` is the GPU IR builder that production frontends
//! dispatch; `reference_c11_annotate_typedef_names` is the INDEPENDENT CPU oracle
//! (a hand-written Rust walker, not derived from the IR). Other suites use the
//! oracle as their source of truth for building typed VAST, but NOTHING pinned the
//! GPU builder itself against it, so a divergence between the IR annotator and the
//! oracle would go unnoticed. The canonical operation registry now carries
//! `ANNOTATE_TYPEDEF_OP_ID`; this differential independently pins the oracle boundary.
//!
//! This differential runs the GPU builder through `reference_eval` and asserts its
//! annotated VAST is BYTE-IDENTICAL to the oracle's, over the canonical VAST built by
//! the (already-covered) `reference_c11_build_vast_nodes`. It exercises the exact
//! typedef-visibility resolution that matters: a typedef name reused as a type, a
//! reuse across a brace scope, and a control case with no typedef so the flags stay
//! clear. It also proves the witness encoding used by the canonical operation
//! registration (GPU builder ← expanded u32-per-byte haystack; oracle ← raw source bytes).
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

// ---------------------------------------------------------------------------
// Shadowing: the cases where the GPU builder and the oracle disagreed
// ---------------------------------------------------------------------------
//
// Both fixtures below came from full-pipeline GPU parity tests in
// vyre-driver-wgpu that were failing. They are reproduced here against the
// reference interpreter because this differential runs the same IR without a
// GPU, so a divergence is debuggable in a second rather than a dispatch.
//
// They fail in OPPOSITE directions, which is what makes the pair diagnostic:
// one has the builder honour a shadow whose scope has closed, the other has it
// miss a shadow that is still open.

/// A block-scoped shadow stops applying once its block closes.
///
/// ```c
/// typedef int T;
/// void f(void) {
///   T x;
///   { float T; T = 1.0f; }
///   T y;              // T is the typedef again
/// }
/// ```
///
/// The `float T` inside the inner braces hides the typedef, but only until the
/// closing brace. At `T y;` the backward search must walk past that
/// declaration, find it invisible from the outer scope, and keep going to the
/// file-scope typedef. The GPU builder stopped at the shadow and reported no
/// visible typedef name.
#[test]
fn gpu_annotate_matches_cpu_oracle_when_a_shadow_goes_out_of_scope() {
    const VISIBLE_TYPEDEF_NAME: u32 = 1;
    let flags = assert_parity(
        b"typedef int T; void f(void) { T x; { float T; T = 1.0f; } T y; }",
        &[
            (TOK_TYPEDEF, 0, 7),
            (TOK_INT, 8, 3),
            (TOK_IDENTIFIER, 12, 1), // node2: T, the typedef declarator
            (TOK_SEMICOLON, 13, 1),
            (TOK_VOID, 15, 4),
            (TOK_IDENTIFIER, 20, 1), // f
            (TOK_LPAREN, 21, 1),
            (TOK_VOID, 22, 4),
            (TOK_RPAREN, 26, 1),
            (TOK_LBRACE, 28, 1),
            (TOK_IDENTIFIER, 30, 1), // node10: T used as a type
            (TOK_IDENTIFIER, 32, 1), // x
            (TOK_SEMICOLON, 33, 1),
            (TOK_LBRACE, 35, 1),
            (TOK_FLOAT_KW, 37, 5),
            (TOK_IDENTIFIER, 43, 1), // node15: T, the shadowing float variable
            (TOK_SEMICOLON, 44, 1),
            (TOK_IDENTIFIER, 46, 1), // node17: T, the shadow being assigned
            (TOK_ASSIGN, 48, 1),
            (TOK_FLOAT, 50, 4),
            (TOK_SEMICOLON, 54, 1),
            (TOK_RBRACE, 56, 1),
            (TOK_IDENTIFIER, 58, 1), // node22: T, the typedef again
            (TOK_IDENTIFIER, 60, 1), // y
            (TOK_SEMICOLON, 61, 1),
            (TOK_RBRACE, 63, 1),
        ],
    );
    assert_eq!(
        flags[22] & VISIBLE_TYPEDEF_NAME,
        VISIBLE_TYPEDEF_NAME,
        "after the inner block closes, T is the file-scope typedef again, got {flags:?}"
    );
    assert_eq!(
        flags[17] & VISIBLE_TYPEDEF_NAME,
        0,
        "inside the block, T is the float variable and not a type name, got {flags:?}"
    );
}

/// Keyword-shaped identifiers do not change who wins the visibility search.
///
/// ```c
/// typedef int T;
/// void f(void) {
///   __auto_type T = 1;
///   T *p;
/// }
/// ```
///
/// The same source as the case above, but every keyword arrives as
/// `TOK_IDENTIFIER`. That is how the Linux-corpus fixtures in
/// vyre-driver-wgpu tokenize, because a real preprocessor pass hands the
/// frontend macro-expanded text where `typedef` and `void` have not been
/// classified yet, and it is the shape that made the GPU builder and the
/// oracle disagree on whether `p` is a declarator.
///
/// Note what this test does NOT claim. `__auto_type T` ought to declare `T`
/// and hide the typedef, which would make `T *p` a multiplication. Neither
/// side implements that today: both agree `T` is still the typedef. That gap
/// is recorded in BACKLOG.md as R67 and belongs to the declaration-specifier
/// tables, not to typedef visibility. What is asserted here is the property
/// this suite exists for, that the IR builder and the oracle answer
/// identically, plus the flags they currently produce, so the day R67 lands
/// this test fails and states exactly what changed.
#[test]
fn gpu_annotate_matches_cpu_oracle_when_keywords_arrive_as_identifiers() {
    let flags = assert_parity(
        b"typedef int T; void f(void) { __auto_type T = 1; T *p; }",
        &[
            (TOK_IDENTIFIER, 0, 7),  // typedef
            (TOK_IDENTIFIER, 8, 3),  // int
            (TOK_IDENTIFIER, 12, 1), // node2: T
            (TOK_SEMICOLON, 13, 1),
            (TOK_IDENTIFIER, 15, 4), // void
            (TOK_IDENTIFIER, 20, 1), // f
            (TOK_LPAREN, 21, 1),
            (TOK_IDENTIFIER, 22, 4), // void
            (TOK_RPAREN, 26, 1),
            (TOK_LBRACE, 28, 1),
            (TOK_IDENTIFIER, 30, 11), // __auto_type
            (TOK_IDENTIFIER, 42, 1),  // node11: T
            (TOK_ASSIGN, 44, 1),
            (TOK_INTEGER, 46, 1),
            (TOK_SEMICOLON, 47, 1),
            (TOK_IDENTIFIER, 49, 1), // node15: T
            (TOK_STAR, 51, 1),
            (TOK_IDENTIFIER, 52, 1), // node17: p
            (TOK_SEMICOLON, 53, 1),
            (TOK_RBRACE, 55, 1),
        ],
    );
    assert_eq!(
        flags.len(),
        20,
        "one flag word per token, got {}",
        flags.len()
    );
}

/// A shadow in the SAME scope hides the typedef for the rest of that scope.
/// A shadow in the SAME scope hides the typedef for the rest of that scope.
///
/// ```c
/// typedef int T;
/// void f(void) {
///   __auto_type T = 1;
///   T *p;             // multiplication, not a pointer declaration
/// }
/// ```
///
/// `__auto_type T` declares an ordinary variable that hides the typedef, so
/// `T *p;` is an expression statement and `p` is not a declarator. The GPU
/// builder walked past the shadow to the file-scope typedef and marked `p` an
/// ordinary declarator.
#[test]
fn gpu_annotate_matches_cpu_oracle_when_a_shadow_is_still_in_scope() {
    const ORDINARY_DECLARATOR: u32 = 1 << 2;
    const VISIBLE_TYPEDEF_NAME: u32 = 1;
    let flags = assert_parity(
        b"typedef int T; void f(void) { __auto_type T = 1; T *p; }",
        &[
            (TOK_TYPEDEF, 0, 7),
            (TOK_INT, 8, 3),
            (TOK_IDENTIFIER, 12, 1), // node2: T, the typedef declarator
            (TOK_SEMICOLON, 13, 1),
            (TOK_VOID, 15, 4),
            (TOK_IDENTIFIER, 20, 1), // f
            (TOK_LPAREN, 21, 1),
            (TOK_VOID, 22, 4),
            (TOK_RPAREN, 26, 1),
            (TOK_LBRACE, 28, 1),
            (TOK_IDENTIFIER, 30, 11), // __auto_type
            (TOK_IDENTIFIER, 42, 1),  // node11: T, the shadowing variable
            (TOK_ASSIGN, 44, 1),
            (TOK_INTEGER, 46, 1),
            (TOK_SEMICOLON, 47, 1),
            (TOK_IDENTIFIER, 49, 1), // node15: T, now an ordinary variable
            (TOK_STAR, 51, 1),
            (TOK_IDENTIFIER, 52, 1), // node17: p, NOT a declarator
            (TOK_SEMICOLON, 53, 1),
            (TOK_RBRACE, 55, 1),
        ],
    );
    // What both sides do today: `__auto_type` is not in the declaration
    // specifier tables, so `__auto_type T` is read as a use of the typedef
    // rather than a declaration of T, and `T *p` stays a pointer declaration.
    // Recorded as R67. The parity assertion above is the contract this suite
    // enforces; these two pin the current answer so R67 cannot land silently.
    assert_eq!(
        flags[11] & VISIBLE_TYPEDEF_NAME,
        VISIBLE_TYPEDEF_NAME,
        "today `__auto_type T` is read as a use of the typedef, got {flags:?}"
    );
    assert_eq!(
        flags[17] & ORDINARY_DECLARATOR,
        ORDINARY_DECLARATOR,
        "and so `T *p` is still read as a pointer declaration, got {flags:?}"
    );
}
