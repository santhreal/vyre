//! Failure-oriented hostile-parser tests for full-C constructs.
//!
//! Targets the VYRE C AST parser (VAST builder + classifier + PG lowerer)
//! with table-driven edge cases that historically break C parsers:
//!
//!   * typedef/expression ambiguity (the "most vexing parse" family)
//!   * nested declarators (function-pointer arrays, parenthesised names)
//!   * compound literals vs casts
//!   * designated initialisers (nested, mixed dot/array)
//!   * GNU attributes and inline asm
//!   * nested structs / enums / unions
//!
//! Every case asserts concrete VAST/PG node kinds.  Where a GPU program
//! exists for the stage under test we also assert CPU/GPU parity.

#![cfg(feature = "c-parser")]
#![allow(clippy::erasing_op)]
#![allow(deprecated)]

use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::lower::reference_ast_to_pg_nodes;
use vyre_libs::parsing::c::parse::vast::{
    reference_c11_build_vast_nodes, reference_c11_classify_vast_node_kinds, C_AST_KIND_ARRAY_DECL,
    C_AST_KIND_ARRAY_SUBSCRIPT_EXPR, C_AST_KIND_CAST_EXPR, C_AST_KIND_COMPOUND_LITERAL_EXPR,
    C_AST_KIND_ENUMERATOR_DECL, C_AST_KIND_FIELD_DECL, C_AST_KIND_FUNCTION_DEFINITION,
    C_AST_KIND_GNU_ATTRIBUTE, C_AST_KIND_INITIALIZER_LIST, C_AST_KIND_INLINE_ASM,
    C_AST_KIND_POINTER_DECL, C_AST_KIND_SIZEOF_EXPR,
};
use vyre_primitives::predicate::node_kind;

mod c_ast_gpu_parity_support;
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;

use c_ast_gpu_parity_support::{
    run_gpu_classifier_with_count as run_gpu_classifier,
    run_gpu_pg_lower_with_count as run_gpu_pg_lower,
    run_gpu_vast_builder_from_parts as run_gpu_vast_builder,
};
use c_frontend::rows::{
    assert_kind, assert_vast_row, pg_word_at, row_indices as typed_indices, starts_for_lens,
    word_at, PG_STRIDE_U32, VAST_STRIDE_U32,
};
use c_frontend::spelling::c_rows;

// The VAST builder / classifier / PG lowerer dispatch is owned by
// `c_ast_gpu_parity_support`, which every other C-AST parity root in this crate
// already drives its stages through. What stays here is the hostile fixture
// corpus and the node kinds each fixture must produce.
//
// Streams are spelled through `c_rows` rather than one `TOK_` per line. These
// six fixtures are distinct constructs that nonetheless share long runs of
// `LBRACE DOT IDENTIFIER ASSIGN` and `LPAREN IDENTIFIER RPAREN`, and one token
// per line made those runs read as copied text while burying the construct in
// 30 lines of scaffolding. Every token here is one source byte wide, which is
// what a bare kind name in a spelling means.

/// ```c
/// typedef int Foo;
/// void bar(void) {
///   Foo *a;            // typedef-name declarator
///   (Foo)*b;           // cast expression (type-name paren without decl context)
///   (Foo)-1;           // cast expression
///   c = (Foo){ .x=1 }; // compound literal + initializer list
/// }
/// ```
fn fixture_typedef_expr_ambiguity() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows(
        "TYPEDEF INT IDENTIFIER SEMICOLON \
         VOID IDENTIFIER LPAREN VOID RPAREN LBRACE \
         IDENTIFIER STAR IDENTIFIER SEMICOLON \
         LPAREN IDENTIFIER RPAREN STAR IDENTIFIER SEMICOLON \
         LPAREN IDENTIFIER RPAREN MINUS INTEGER SEMICOLON \
         IDENTIFIER ASSIGN LPAREN IDENTIFIER RPAREN \
         LBRACE DOT IDENTIFIER ASSIGN INTEGER RBRACE SEMICOLON \
         RBRACE",
    )
}

/// ```c
/// int (*(*f[4])(int))[2];
/// ```
///
/// Deeply nested: the classifier loses the decl context after the first
/// parenthesis, so only the outermost star is a POINTER_DECL.
fn fixture_deeply_nested_declarator() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows(
        "INT LPAREN STAR LPAREN STAR IDENTIFIER LBRACKET INTEGER RBRACKET RPAREN \
         LPAREN INT RPAREN RPAREN LBRACKET INTEGER RBRACKET SEMICOLON",
    )
}

/// ```c
/// struct S { int a; struct { int b; } nested; };
/// enum E { A, B };
/// ```
fn fixture_nested_struct_enum() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows(
        "STRUCT IDENTIFIER LBRACE INT IDENTIFIER SEMICOLON \
         STRUCT LBRACE INT IDENTIFIER SEMICOLON RBRACE IDENTIFIER SEMICOLON \
         RBRACE SEMICOLON \
         ENUM IDENTIFIER LBRACE IDENTIFIER COMMA IDENTIFIER RBRACE SEMICOLON",
    )
}

/// ```c
/// int x[] = { [0] = 1, [1] = { [2] = 3, [0] = 4 } };
/// ```
fn fixture_nested_designated_init() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows(
        "INT IDENTIFIER LBRACKET RBRACKET ASSIGN LBRACE \
         LBRACKET INTEGER RBRACKET ASSIGN INTEGER COMMA \
         LBRACKET INTEGER RBRACKET ASSIGN LBRACE \
         LBRACKET INTEGER RBRACKET ASSIGN INTEGER COMMA \
         LBRACKET INTEGER RBRACKET ASSIGN INTEGER RBRACE \
         RBRACE SEMICOLON",
    )
}

/// ```c
/// __attribute__((noreturn)) void die(int code) {
///   __asm__ volatile ("ud2" ::: "memory");
/// }
/// ```
fn fixture_gnu_attribute_inline_asm() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows(
        "GNU_ATTRIBUTE LPAREN LPAREN IDENTIFIER RPAREN RPAREN \
         VOID IDENTIFIER LPAREN INT IDENTIFIER RPAREN LBRACE \
         GNU_ASM VOLATILE LPAREN STRING COLON COLON COLON STRING RPAREN SEMICOLON \
         RBRACE",
    )
}

/// ```c
/// void f(void) {
///   int *p = (int []){ 1, 2, 3 };
///   struct S *s = (struct S){ .a = 1 };
/// }
/// ```
fn fixture_compound_literal_stress() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows(
        "VOID IDENTIFIER LPAREN VOID RPAREN LBRACE \
         INT STAR IDENTIFIER ASSIGN LPAREN INT LBRACKET RBRACKET RPAREN \
         LBRACE INTEGER COMMA INTEGER COMMA INTEGER RBRACE SEMICOLON \
         STRUCT IDENTIFIER STAR IDENTIFIER ASSIGN LPAREN STRUCT IDENTIFIER RPAREN \
         LBRACE DOT IDENTIFIER ASSIGN INTEGER RBRACE SEMICOLON \
         RBRACE",
    )
}

/// The spellings above produce exactly the streams the `TOK_` lists they replace
/// produced.
///
/// A spelling is a second way to say the same thing, so a mistyped kind name
/// would silently retarget a contract at a different construct while every
/// assertion below still passed against the new shape. Token counts and the two
/// kinds that carry the whole point of a fixture are pinned here as literals,
/// read from the stream rather than restated from the spelling.
#[test]
fn every_hostile_fixture_spells_the_stream_its_contracts_index() {
    let (types, starts, lens) = fixture_typedef_expr_ambiguity();
    assert_eq!(types.len(), 39, "typedef/expression ambiguity token count");
    assert_eq!(lens, vec![1; types.len()], "every hostile token is one byte");
    assert_eq!(
        starts,
        (0..types.len() as u32).collect::<Vec<u32>>(),
        "one-byte tokens lay out at consecutive offsets"
    );
    // Row 11 is the `*` in `Foo *a` that the POINTER_DECL contract indexes, and
    // row 17 the `*` in `(Foo)*b` that must NOT be one. Both are positional, so a
    // shifted stream would silently move the contract onto another token.
    assert_eq!(types[11], TOK_STAR, "row 11 is the `*` of `Foo *a`");
    assert_eq!(types[17], TOK_STAR, "row 17 is the `*` of `(Foo)*b`");
    assert_eq!(
        types[32], TOK_DOT,
        "row 32 is the `.` of the compound literal's designator"
    );

    assert_eq!(
        fixture_deeply_nested_declarator().0.len(),
        18,
        "nested declarator token count"
    );
    assert_eq!(
        fixture_nested_struct_enum().0.len(),
        24,
        "nested struct/enum token count"
    );
    assert_eq!(
        fixture_nested_designated_init().0.len(),
        31,
        "nested designated-init token count"
    );
    let (asm_types, _, _) = fixture_gnu_attribute_inline_asm();
    assert_eq!(asm_types.len(), 24, "GNU attribute / asm token count");
    assert_eq!(
        asm_types[16], TOK_STRING,
        "the asm template stays a STRING; a kind-name typo here would retarget the \
         INLINE_ASM contract at a different row"
    );
    assert_eq!(
        fixture_compound_literal_stress().0.len(),
        40,
        "compound-literal stress token count"
    );
}

// ---------------------------------------------------------------------------
// Table-driven CPU reference tests
// ---------------------------------------------------------------------------

#[path = "c11_parser_hostile_full_c/cpu_pg_and_gpu_parity.rs"]
mod cpu_pg_and_gpu_parity;
#[path = "c11_parser_hostile_full_c/cpu_reference_and_gpu_parity.rs"]
mod cpu_reference_and_gpu_parity;
