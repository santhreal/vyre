//! Deep contracts for C AST initializer designators, compound literals,
//! assignment suppression/classification, string initializers, and full
//! CPU / GPU / PG parity.
//!
//! Coverage:
//!   * nested designators (field → array, field → struct)
//!   * GNU range designators `[a ... b]`
//!   * field designators in unions and structs
//!   * mixed positional / designated initializers
//!   * compound literals inside initializer lists
//!   * declaration initializer assignment suppression
//!   * designator assignment classification
//!   * string / char array initialization in nested aggregates
//!   * CPU reference, PG lowering preservation, GPU parity for all of the above

#![cfg(feature = "c-parser")]
#![allow(deprecated)]

use vyre_libs::parsing::c::lower::reference_ast_to_pg_nodes;
use vyre_libs::parsing::c::parse::vast::{
    reference_c11_build_vast_nodes, reference_c11_classify_vast_node_kinds, C_AST_KIND_ARRAY_DECL,
    C_AST_KIND_ARRAY_SUBSCRIPT_EXPR, C_AST_KIND_ASSIGN_EXPR, C_AST_KIND_COMPOUND_LITERAL_EXPR,
    C_AST_KIND_FIELD_DECL, C_AST_KIND_INITIALIZER_LIST, C_AST_KIND_MEMBER_ACCESS_EXPR,
    C_AST_KIND_RANGE_DESIGNATOR_EXPR, C_AST_KIND_STRUCT_DECL, C_AST_KIND_UNION_DECL,
};
use vyre_primitives::predicate::node_kind;

mod c_ast_gpu_parity_support;
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
#[path = "../../tests/support/c_frontend/fixtures/initializer_designator_streams.rs"]
mod initializer_designator_streams;
use c_ast_gpu_parity_support::{
    run_gpu_classifier, run_gpu_pg_lower, run_gpu_vast_builder_from_parts as run_gpu_vast_builder,
};

use c_frontend::expression_pipeline::run_reference_pg_lower;
use c_frontend::rows::{assert_pg_preserves_row, kind_at, row_indices};
use c_frontend::spelling::c_rows;
use initializer_designator_streams::union_field_designator;

fn assert_full_pipeline_parity(
    tok_types: &[u32],
    tok_starts: &[u32],
    tok_lens: &[u32],
    label: &str,
) {
    let raw_cpu = reference_c11_build_vast_nodes(tok_types, tok_starts, tok_lens);
    let raw_gpu = run_gpu_vast_builder(tok_types, tok_starts, tok_lens);
    assert_eq!(
        raw_gpu, raw_cpu,
        "{label}: GPU VAST builder must match CPU oracle"
    );

    let typed_cpu = reference_c11_classify_vast_node_kinds(&raw_cpu);
    let typed_gpu = run_gpu_classifier(&raw_cpu);
    assert_eq!(
        typed_gpu, typed_cpu,
        "{label}: GPU classifier must match CPU oracle"
    );

    let pg_cpu = reference_ast_to_pg_nodes(&typed_cpu);
    let pg_gpu = run_gpu_pg_lower(&typed_cpu);
    assert_eq!(
        pg_gpu, pg_cpu,
        "{label}: GPU PG lowerer must match CPU oracle"
    );
}

// ---------------------------------------------------------------------------
// Fixtures
//
// Spelled through `c_rows` rather than one `TOK_` per line: two unrelated
// initializer fixtures share long runs of `LBRACE DOT IDENTIFIER ASSIGN ...`,
// and one token per line makes those runs read as copied text.
// ---------------------------------------------------------------------------

/// ```c
/// struct S s = { .a[0] = 1, .b = { .c = 2 } };
/// ```
fn fixture_nested_field_array_designator() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows(
        "STRUCT:6 IDENTIFIER IDENTIFIER ASSIGN LBRACE \
         DOT IDENTIFIER LBRACKET INTEGER RBRACKET ASSIGN INTEGER COMMA \
         DOT IDENTIFIER ASSIGN LBRACE DOT IDENTIFIER ASSIGN INTEGER RBRACE \
         RBRACE SEMICOLON",
    )
}

/// ```c
/// int arr[10] = { [0 ... 3] = 1, [5] = 2 };
/// ```
fn fixture_range_designator_array() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows(
        "INT:3 IDENTIFIER:3 LBRACKET INTEGER:2 RBRACKET ASSIGN LBRACE \
         LBRACKET INTEGER ELLIPSIS:3 INTEGER RBRACKET ASSIGN INTEGER COMMA \
         LBRACKET INTEGER RBRACKET ASSIGN INTEGER RBRACE SEMICOLON",
    )
}

/// ```c
/// struct S s = { 1, .b = 2, 3 };
/// ```
fn fixture_mixed_positional_designated() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows(
        "STRUCT:6 IDENTIFIER IDENTIFIER ASSIGN LBRACE \
         INTEGER COMMA DOT IDENTIFIER ASSIGN INTEGER COMMA INTEGER RBRACE SEMICOLON",
    )
}

/// ```c
/// struct T t = { .inner = (struct S){ .x = 1 } };
/// ```
fn fixture_compound_literal_nested() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows(
        "STRUCT:6 IDENTIFIER IDENTIFIER ASSIGN LBRACE DOT IDENTIFIER:5 ASSIGN \
         LPAREN STRUCT:6 IDENTIFIER RPAREN LBRACE DOT IDENTIFIER ASSIGN INTEGER RBRACE \
         RBRACE SEMICOLON",
    )
}

/// ```c
/// int x = {1};
/// ```
fn fixture_assignment_suppression() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows("INT:3 IDENTIFIER ASSIGN LBRACE INTEGER RBRACE SEMICOLON")
}

/// ```c
/// struct S s = { .a = 1 };
/// ```
fn fixture_designator_assignment_class() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows(
        "STRUCT:6 IDENTIFIER IDENTIFIER ASSIGN \
         LBRACE DOT IDENTIFIER ASSIGN INTEGER RBRACE SEMICOLON",
    )
}

/// ```c
/// struct Buf { char data[4]; } b = { .data = "abc" };
/// ```
fn fixture_string_char_array_nested() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows(
        "STRUCT:6 IDENTIFIER:3 LBRACE CHAR_KW:4 IDENTIFIER:4 LBRACKET INTEGER RBRACKET \
         SEMICOLON RBRACE IDENTIFIER ASSIGN \
         LBRACE DOT IDENTIFIER:4 ASSIGN STRING:5 RBRACE SEMICOLON",
    )
}

// ---------------------------------------------------------------------------
// CPU reference contracts
// ---------------------------------------------------------------------------

#[path = "c_ast_initializer_designator_deep_contracts/cpu_pg_and_gpu_parity.rs"]
mod cpu_pg_and_gpu_parity;
#[path = "c_ast_initializer_designator_deep_contracts/gpu_parity.rs"]
mod gpu_parity;
