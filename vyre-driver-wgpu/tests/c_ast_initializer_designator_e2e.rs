//! End-to-end C AST tests for initializer lists, designators, compound literals,
//! and aggregate lowering (array / struct / union / enum) with GPU/CPU parity.
//!
//! Coverage:
//!   * plain initializer lists for arrays and structs
//!   * nested designated initializers (dot and array subscript mixed)
//!   * compound literals in assignment and call contexts
//!   * union designated initializers
//!   * enum declarations with explicit/implicit values
//!   * GNU range designators (`[a ... b]`)
//!   * PG lowering preservation (kind, span, parent, first_child, next_sibling)

#![cfg(feature = "c-parser")]
#![allow(deprecated)]

use vyre_libs::parsing::c::parse::vast::{
    reference_c11_build_vast_nodes, reference_c11_classify_vast_node_kinds,
    C_AST_KIND_ARRAY_SUBSCRIPT_EXPR, C_AST_KIND_ASSIGN_EXPR, C_AST_KIND_COMPOUND_LITERAL_EXPR,
    C_AST_KIND_ENUMERATOR_DECL, C_AST_KIND_FIELD_DECL, C_AST_KIND_INITIALIZER_LIST,
    C_AST_KIND_MEMBER_ACCESS_EXPR,
};
use vyre_primitives::predicate::node_kind;

mod c_ast_gpu_parity_support;
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
use c_ast_gpu_parity_support::run_gpu_pg_lower;

use c_frontend::expression_pipeline::run_reference_pg_lower;
use c_frontend::rows::{assert_pg_preserves_row_and_kind, row_indices};
use c_frontend::spelling::c_rows;

// ---------------------------------------------------------------------------
// Fixtures
//
// Spelled through `c_rows` rather than one `TOK_` per line: two unrelated
// initializer fixtures share long runs of `LBRACE DOT IDENTIFIER ASSIGN ...`,
// and one token per line makes those runs read as copied text.
// ---------------------------------------------------------------------------

/// ```c
/// int arr[3] = {1, 2, 3};
/// ```
fn fixture_array_initializer_list() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows(
        "INT:3 IDENTIFIER:3 LBRACKET INTEGER RBRACKET ASSIGN \
         LBRACE INTEGER COMMA INTEGER COMMA INTEGER RBRACE SEMICOLON",
    )
}

/// ```c
/// struct Point p = {10, "label"};
/// ```
fn fixture_struct_initializer_list() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows(
        "STRUCT:6 IDENTIFIER:5 IDENTIFIER ASSIGN \
         LBRACE INTEGER:2 COMMA STRING:7 RBRACE SEMICOLON",
    )
}

/// ```c
/// union U u = {.i = 42};
/// ```
fn fixture_union_designated_init() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows(
        "UNION:5 IDENTIFIER IDENTIFIER ASSIGN \
         LBRACE DOT IDENTIFIER ASSIGN INTEGER:2 RBRACE SEMICOLON",
    )
}

/// ```c
/// enum Color { RED = 0, GREEN, BLUE = 2 };
/// enum Color c = GREEN;
/// ```
fn fixture_enum_with_initializer() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows(
        "ENUM:4 IDENTIFIER:5 LBRACE IDENTIFIER:3 ASSIGN INTEGER COMMA IDENTIFIER:5 COMMA \
         IDENTIFIER:4 ASSIGN INTEGER RBRACE SEMICOLON \
         ENUM:4 IDENTIFIER:5 IDENTIFIER ASSIGN IDENTIFIER:5 SEMICOLON",
    )
}

/// ```c
/// struct config cfg = {
///   .name = "test",
///   .dims = { [0] = 1, [1] = { .x = 2, .y = 3 } },
///   .flags[2] = 1,
/// };
/// ```
fn fixture_nested_designator_mixed() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows(
        "STRUCT IDENTIFIER IDENTIFIER ASSIGN LBRACE \
         DOT IDENTIFIER ASSIGN STRING COMMA \
         DOT IDENTIFIER ASSIGN LBRACE \
         LBRACKET INTEGER RBRACKET ASSIGN INTEGER COMMA \
         LBRACKET INTEGER RBRACKET ASSIGN LBRACE \
         DOT IDENTIFIER ASSIGN INTEGER COMMA DOT IDENTIFIER ASSIGN INTEGER RBRACE \
         RBRACE COMMA \
         DOT IDENTIFIER LBRACKET INTEGER RBRACKET ASSIGN INTEGER COMMA \
         RBRACE SEMICOLON",
    )
}

/// ```c
/// struct Rect r = (struct Rect){ .w = 10, .h = 20 };
/// ```
fn fixture_compound_literal_expr() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows(
        "STRUCT:6 IDENTIFIER:4 IDENTIFIER ASSIGN LPAREN STRUCT:6 IDENTIFIER:4 RPAREN \
         LBRACE DOT IDENTIFIER ASSIGN INTEGER:2 COMMA DOT IDENTIFIER ASSIGN INTEGER:2 \
         RBRACE SEMICOLON",
    )
}

/// ```c
/// void f(struct S);
/// f((struct S){ .a = 1 });
/// ```
fn fixture_compound_literal_in_call() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows(
        "VOID:4 IDENTIFIER LPAREN STRUCT:6 IDENTIFIER RPAREN SEMICOLON \
         IDENTIFIER LPAREN LPAREN STRUCT:6 IDENTIFIER RPAREN \
         LBRACE DOT IDENTIFIER ASSIGN INTEGER RBRACE RPAREN SEMICOLON",
    )
}

// ---------------------------------------------------------------------------
// CPU reference tests  -  shape & kind correctness
// ---------------------------------------------------------------------------

#[path = "c_ast_initializer_designator_e2e/cpu_pg_and_gpu_parity.rs"]
mod cpu_pg_and_gpu_parity;
#[path = "c_ast_initializer_designator_e2e/gpu_parity.rs"]
mod gpu_parity;
