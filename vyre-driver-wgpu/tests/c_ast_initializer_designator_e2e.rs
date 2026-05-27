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

use vyre::ir::Expr;
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::lower::c_lower_ast_to_pg_nodes;
use vyre_libs::parsing::c::parse::vast::{
    reference_c11_build_vast_nodes, reference_c11_classify_vast_node_kinds,
    C_AST_KIND_ARRAY_SUBSCRIPT_EXPR, C_AST_KIND_ASSIGN_EXPR, C_AST_KIND_COMPOUND_LITERAL_EXPR,
    C_AST_KIND_ENUMERATOR_DECL, C_AST_KIND_FIELD_DECL, C_AST_KIND_INITIALIZER_LIST,
    C_AST_KIND_MEMBER_ACCESS_EXPR,
};
use vyre_primitives::predicate::node_kind;
use vyre_reference::value::Value;

mod c_ast_gpu_parity_support;
use c_ast_gpu_parity_support::{run_gpu_pg_lower, starts_for_lens};

const VAST_STRIDE_U32: usize = 10;
const PG_STRIDE_U32: usize = 6;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn word_at(buf: &[u8], word: usize) -> u32 {
    let off = word * 4;
    u32::from_le_bytes(buf[off..off + 4].try_into().unwrap())
}

fn node_count_from_vast(vast: &[u8]) -> u32 {
    u32::try_from(vast.len() / (VAST_STRIDE_U32 * 4)).unwrap_or_default()
}

fn typed_indices(rows: &[u8], kind: u32) -> Vec<usize> {
    rows.chunks_exact(VAST_STRIDE_U32 * 4)
        .enumerate()
        .filter_map(|(idx, row)| {
            let row_kind = u32::from_le_bytes(row[0..4].try_into().unwrap());
            (row_kind == kind).then_some(idx)
        })
        .collect()
}

fn pg_word_at(buf: &[u8], idx: usize, field: usize) -> u32 {
    word_at(buf, idx * PG_STRIDE_U32 + field)
}

fn run_reference_pg_lower(typed_vast: &[u8]) -> Vec<u8> {
    let num_nodes = node_count_from_vast(typed_vast);
    let program = c_lower_ast_to_pg_nodes("vast_nodes", Expr::u32(num_nodes), "pg_nodes");
    let output_len = num_nodes.saturating_mul(PG_STRIDE_U32 as u32).max(1) as usize * 4;
    let values = [
        Value::from(typed_vast.to_vec()),
        Value::from(vec![0; output_len]),
    ];
    let outputs = vyre_reference::reference_eval(&program, &values)
        .unwrap_or_else(|error| panic!("Fix: C AST PG lowerer must execute on CPU: {error}"));
    assert_eq!(outputs.len(), 1, "Fix: PG lowerer must emit one buffer");
    outputs[0].to_bytes()
}

fn assert_pg_preserves_kind_span_and_links(
    typed_vast: &[u8],
    pg: &[u8],
    tok_starts: &[u32],
    tok_lens: &[u32],
    idx: usize,
    expected_kind: u32,
) {
    let pg_kind = pg_word_at(pg, idx, 0);
    let pg_start = pg_word_at(pg, idx, 1);
    let pg_end = pg_word_at(pg, idx, 2);
    let pg_parent = pg_word_at(pg, idx, 3);
    let pg_first_child = pg_word_at(pg, idx, 4);
    let pg_next_sibling = pg_word_at(pg, idx, 5);

    let vast_kind = word_at(typed_vast, idx * VAST_STRIDE_U32);
    let vast_parent = word_at(typed_vast, idx * VAST_STRIDE_U32 + 1);
    let vast_first_child = word_at(typed_vast, idx * VAST_STRIDE_U32 + 2);
    let vast_next_sibling = word_at(typed_vast, idx * VAST_STRIDE_U32 + 3);

    assert_eq!(pg_kind, expected_kind, "PG kind mismatch at row {idx}");
    assert_eq!(pg_kind, vast_kind, "PG/VAST kind drift at row {idx}");
    assert_eq!(
        pg_start, tok_starts[idx],
        "PG span_start mismatch at row {idx}"
    );
    assert_eq!(
        pg_end,
        tok_starts[idx] + tok_lens[idx],
        "PG span_end mismatch at row {idx}"
    );
    assert_eq!(pg_parent, vast_parent, "PG parent drift at row {idx}");
    assert_eq!(
        pg_first_child, vast_first_child,
        "PG first_child drift at row {idx}"
    );
    assert_eq!(
        pg_next_sibling, vast_next_sibling,
        "PG next_sibling drift at row {idx}"
    );
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// ```c
/// int arr[3] = {1, 2, 3};
/// ```
fn fixture_array_initializer_list() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_INT,
        TOK_IDENTIFIER,
        TOK_LBRACKET,
        TOK_INTEGER,
        TOK_RBRACKET,
        TOK_ASSIGN,
        TOK_LBRACE,
        TOK_INTEGER,
        TOK_COMMA,
        TOK_INTEGER,
        TOK_COMMA,
        TOK_INTEGER,
        TOK_RBRACE,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
    let tok_starts = starts_for_lens(&tok_lens);
    (tok_types, tok_starts, tok_lens)
}

/// ```c
/// struct Point p = {10, "label"};
/// ```
fn fixture_struct_initializer_list() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_STRUCT,
        TOK_IDENTIFIER,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_LBRACE,
        TOK_INTEGER,
        TOK_COMMA,
        TOK_STRING,
        TOK_RBRACE,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![6, 5, 1, 1, 1, 2, 1, 7, 1, 1];
    let tok_starts = starts_for_lens(&tok_lens);
    (tok_types, tok_starts, tok_lens)
}

/// ```c
/// union U u = {.i = 42};
/// ```
fn fixture_union_designated_init() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_UNION,
        TOK_IDENTIFIER,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_LBRACE,
        TOK_DOT,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_INTEGER,
        TOK_RBRACE,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![5, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1];
    let tok_starts = starts_for_lens(&tok_lens);
    (tok_types, tok_starts, tok_lens)
}

/// ```c
/// enum Color { RED = 0, GREEN, BLUE = 2 };
/// enum Color c = GREEN;
/// ```
fn fixture_enum_with_initializer() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_ENUM,
        TOK_IDENTIFIER,
        TOK_LBRACE,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_INTEGER,
        TOK_COMMA,
        TOK_IDENTIFIER,
        TOK_COMMA,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_INTEGER,
        TOK_RBRACE,
        TOK_SEMICOLON,
        TOK_ENUM,
        TOK_IDENTIFIER,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![4, 5, 1, 3, 1, 1, 1, 5, 1, 4, 1, 1, 1, 1, 4, 5, 1, 1, 5, 1];
    let tok_starts = starts_for_lens(&tok_lens);
    (tok_types, tok_starts, tok_lens)
}

/// ```c
/// struct config cfg = {
///   .name = "test",
///   .dims = { [0] = 1, [1] = { .x = 2, .y = 3 } },
///   .flags[2] = 1,
/// };
/// ```
fn fixture_nested_designator_mixed() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_STRUCT,
        TOK_IDENTIFIER,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_LBRACE,
        TOK_DOT,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_STRING,
        TOK_COMMA,
        TOK_DOT,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_LBRACE,
        TOK_LBRACKET,
        TOK_INTEGER,
        TOK_RBRACKET,
        TOK_ASSIGN,
        TOK_INTEGER,
        TOK_COMMA,
        TOK_LBRACKET,
        TOK_INTEGER,
        TOK_RBRACKET,
        TOK_ASSIGN,
        TOK_LBRACE,
        TOK_DOT,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_INTEGER,
        TOK_COMMA,
        TOK_DOT,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_INTEGER,
        TOK_RBRACE,
        TOK_RBRACE,
        TOK_COMMA,
        TOK_DOT,
        TOK_IDENTIFIER,
        TOK_LBRACKET,
        TOK_INTEGER,
        TOK_RBRACKET,
        TOK_ASSIGN,
        TOK_INTEGER,
        TOK_COMMA,
        TOK_RBRACE,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![1; tok_types.len()];
    let tok_starts = starts_for_lens(&tok_lens);
    (tok_types, tok_starts, tok_lens)
}

/// ```c
/// struct Rect r = (struct Rect){ .w = 10, .h = 20 };
/// ```
fn fixture_compound_literal_expr() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_STRUCT,
        TOK_IDENTIFIER,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_LPAREN,
        TOK_STRUCT,
        TOK_IDENTIFIER,
        TOK_RPAREN,
        TOK_LBRACE,
        TOK_DOT,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_INTEGER,
        TOK_COMMA,
        TOK_DOT,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_INTEGER,
        TOK_RBRACE,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![6, 4, 1, 1, 1, 6, 4, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1];
    let tok_starts = starts_for_lens(&tok_lens);
    (tok_types, tok_starts, tok_lens)
}

/// ```c
/// void f(struct S);
/// f((struct S){ .a = 1 });
/// ```
fn fixture_compound_literal_in_call() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_VOID,
        TOK_IDENTIFIER,
        TOK_LPAREN,
        TOK_STRUCT,
        TOK_IDENTIFIER,
        TOK_RPAREN,
        TOK_SEMICOLON,
        TOK_IDENTIFIER,
        TOK_LPAREN,
        TOK_LPAREN,
        TOK_STRUCT,
        TOK_IDENTIFIER,
        TOK_RPAREN,
        TOK_LBRACE,
        TOK_DOT,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_INTEGER,
        TOK_RBRACE,
        TOK_RPAREN,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![
        4, 1, 1, 6, 1, 1, 1, 1, 1, 1, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ];
    let tok_starts = starts_for_lens(&tok_lens);
    (tok_types, tok_starts, tok_lens)
}

// ---------------------------------------------------------------------------
// CPU reference tests  -  shape & kind correctness
// ---------------------------------------------------------------------------

mod c_ast_initializer_designator_e2e_part1 {

    include!("__split/c_ast_initializer_designator_e2e_part1.rs");
}
mod c_ast_initializer_designator_e2e_part2 {
    include!("__split/c_ast_initializer_designator_e2e_part2.rs");
}
