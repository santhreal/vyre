//! End-to-end GPU/CPU parity tests for C expression precedence and associativity.
//!
//! Covers: comma boundaries, assignment chains, nested ternary, logical/bitwise
//! precedence ladders, cast vs parenthesized expression typing, postfix
//! call/index/member, and unary chains.  Every fixture asserts both expression
//! shape rows and PG lowering (kind, span, and tree-link preservation).

#![cfg(feature = "c-parser")]
#![allow(clippy::too_many_arguments, clippy::erasing_op)]
#![allow(deprecated)]

use vyre::ir::Expr;
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::lower::{c_lower_ast_to_pg_nodes, reference_ast_to_pg_nodes};
use vyre_libs::parsing::c::parse::vast::{
    reference_c11_build_expression_shape_nodes, reference_c11_build_vast_nodes,
    reference_c11_classify_vast_node_kinds, C_AST_KIND_ARRAY_SUBSCRIPT_EXPR,
    C_AST_KIND_ASSIGN_EXPR, C_AST_KIND_CAST_EXPR, C_AST_KIND_CONDITIONAL_EXPR,
    C_AST_KIND_MEMBER_ACCESS_EXPR, C_AST_KIND_UNARY_EXPR, C_EXPR_ASSOC_LEFT, C_EXPR_ASSOC_NONE,
    C_EXPR_ASSOC_RIGHT, C_EXPR_SHAPE_BINARY, C_EXPR_SHAPE_CONDITIONAL, C_EXPR_SHAPE_NONE,
    C_EXPR_SHAPE_STRIDE_U32,
};
use vyre_primitives::predicate::node_kind;
use vyre_reference::value::Value;

mod c_ast_gpu_parity_support;
use c_ast_gpu_parity_support::{run_gpu_expr_shape, run_gpu_pg_lower};

const VAST_STRIDE_U32: usize = 10;
const PG_STRIDE_U32: usize = 6;
const SENTINEL: u32 = u32::MAX;

struct PipelineRows {
    tok_starts: Vec<u32>,
    tok_lens: Vec<u32>,
    typed_vast: Vec<u8>,
    expr_shape: Vec<u8>,
    pg_nodes: Vec<u8>,
}

fn bytes(words: &[u32]) -> Vec<u8> {
    vyre_primitives::wire::pack_u32_slice(words)
}

fn starts_for_lens(lens: &[u32]) -> Vec<u32> {
    let mut cursor = 0u32;
    lens.iter()
        .map(|len| {
            let start = cursor;
            cursor = cursor.saturating_add(*len).saturating_add(1);
            start
        })
        .collect()
}

fn word_at(bytes: &[u8], word: usize) -> u32 {
    let offset = word * 4;
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn node_count_from_vast(vast: &[u8]) -> u32 {
    u32::try_from(vast.len() / (VAST_STRIDE_U32 * 4)).unwrap_or_default()
}

fn row_indices(rows: &[u8], stride_words: usize, kind: u32) -> Vec<usize> {
    rows.chunks_exact(stride_words * 4)
        .enumerate()
        .filter_map(|(idx, row)| {
            let row_kind = u32::from_le_bytes(row[0..4].try_into().unwrap());
            (row_kind == kind).then_some(idx)
        })
        .collect()
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

fn run_pipeline(tok_types: &[u32], tok_lens: &[u32]) -> PipelineRows {
    let tok_starts = starts_for_lens(tok_lens);
    let raw_vast = reference_c11_build_vast_nodes(tok_types, &tok_starts, tok_lens);
    let typed_vast = reference_c11_classify_vast_node_kinds(&raw_vast);
    let expr_shape = reference_c11_build_expression_shape_nodes(&raw_vast, &typed_vast);
    let pg_nodes = run_reference_pg_lower(&typed_vast);
    assert_eq!(
        pg_nodes,
        reference_ast_to_pg_nodes(&typed_vast),
        "Fix: executable PG lowerer must match the byte oracle"
    );

    PipelineRows {
        tok_starts,
        tok_lens: tok_lens.to_vec(),
        typed_vast,
        expr_shape,
        pg_nodes,
    }
}

fn assert_kind(rows: &[u8], idx: usize, stride_words: usize, kind: u32) {
    assert_eq!(word_at(rows, idx * stride_words), kind, "kind at row {idx}");
}

fn assert_pg_preserves_row(rows: &PipelineRows, idx: usize, kind: u32) {
    assert_kind(&rows.typed_vast, idx, VAST_STRIDE_U32, kind);
    assert_kind(&rows.pg_nodes, idx, PG_STRIDE_U32, kind);
    assert_eq!(
        word_at(&rows.pg_nodes, idx * PG_STRIDE_U32 + 1),
        rows.tok_starts[idx],
        "PG span_start at row {idx}"
    );
    assert_eq!(
        word_at(&rows.pg_nodes, idx * PG_STRIDE_U32 + 2),
        rows.tok_starts[idx] + rows.tok_lens[idx],
        "PG span_end at row {idx}"
    );
}

fn assert_pg_links_match_vast(rows: &PipelineRows, idx: usize) {
    assert_eq!(
        word_at(&rows.pg_nodes, idx * PG_STRIDE_U32 + 3),
        word_at(&rows.typed_vast, idx * VAST_STRIDE_U32 + 1),
        "PG parent at row {idx}"
    );
    assert_eq!(
        word_at(&rows.pg_nodes, idx * PG_STRIDE_U32 + 4),
        word_at(&rows.typed_vast, idx * VAST_STRIDE_U32 + 2),
        "PG first_child at row {idx}"
    );
    assert_eq!(
        word_at(&rows.pg_nodes, idx * PG_STRIDE_U32 + 5),
        word_at(&rows.typed_vast, idx * VAST_STRIDE_U32 + 3),
        "PG next_sibling at row {idx}"
    );
}

fn assert_shape_row(
    rows: &[u8],
    idx: usize,
    shape_kind: u32,
    raw_operator: u32,
    precedence: u32,
    associativity: u32,
    first: u32,
    second: u32,
    third: u32,
) {
    let row = idx * C_EXPR_SHAPE_STRIDE_U32 as usize;
    assert_eq!(word_at(rows, row), shape_kind, "shape_kind[{idx}]");
    assert_eq!(
        word_at(rows, row + 1),
        if shape_kind == C_EXPR_SHAPE_NONE {
            SENTINEL
        } else {
            idx as u32
        },
        "source_idx[{idx}]"
    );
    assert_eq!(word_at(rows, row + 2), raw_operator, "raw_operator[{idx}]");
    assert_eq!(word_at(rows, row + 3), precedence, "precedence[{idx}]");
    assert_eq!(
        word_at(rows, row + 4),
        associativity,
        "associativity[{idx}]"
    );
    assert_eq!(word_at(rows, row + 5), first, "first[{idx}]");
    assert_eq!(word_at(rows, row + 6), second, "second[{idx}]");
    assert_eq!(word_at(rows, row + 7), third, "third[{idx}]");
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn comma_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_IDENTIFIER,
        TOK_COMMA,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_IDENTIFIER,
        TOK_COMMA,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

fn assignment_chain_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

fn ternary_nesting_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_IDENTIFIER,
        TOK_QUESTION,
        TOK_IDENTIFIER,
        TOK_QUESTION,
        TOK_IDENTIFIER,
        TOK_COLON,
        TOK_IDENTIFIER,
        TOK_COLON,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

fn logical_bitwise_fixture() -> (Vec<u32>, Vec<u32>) {
    // a || b && c | d ^ e & f == g < h + i * j;
    let tok_types = vec![
        TOK_IDENTIFIER,
        TOK_OR,
        TOK_IDENTIFIER,
        TOK_AND,
        TOK_IDENTIFIER,
        TOK_PIPE,
        TOK_IDENTIFIER,
        TOK_CARET,
        TOK_IDENTIFIER,
        TOK_AMP,
        TOK_IDENTIFIER,
        TOK_EQ,
        TOK_IDENTIFIER,
        TOK_LT,
        TOK_IDENTIFIER,
        TOK_PLUS,
        TOK_IDENTIFIER,
        TOK_STAR,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

fn cast_vs_paren_fixture() -> (Vec<u32>, Vec<u32>) {
    // (int)a; (b + c);
    let tok_types = vec![
        TOK_LPAREN,
        TOK_INT,
        TOK_RPAREN,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
        TOK_LPAREN,
        TOK_IDENTIFIER,
        TOK_PLUS,
        TOK_IDENTIFIER,
        TOK_RPAREN,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

fn postfix_fixture() -> (Vec<u32>, Vec<u32>) {
    // a(b); a[b]; a.c; a->d;
    let tok_types = vec![
        TOK_IDENTIFIER,
        TOK_LPAREN,
        TOK_IDENTIFIER,
        TOK_RPAREN,
        TOK_SEMICOLON,
        TOK_IDENTIFIER,
        TOK_LBRACKET,
        TOK_IDENTIFIER,
        TOK_RBRACKET,
        TOK_SEMICOLON,
        TOK_IDENTIFIER,
        TOK_DOT,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
        TOK_IDENTIFIER,
        TOK_ARROW,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

fn unary_chain_fixture() -> (Vec<u32>, Vec<u32>) {
    // !~-*&++a;
    let tok_types = vec![
        TOK_BANG,
        TOK_TILDE,
        TOK_MINUS,
        TOK_STAR,
        TOK_AMP,
        TOK_INC,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

// ---------------------------------------------------------------------------
// CPU shape + PG lowering tests
// ---------------------------------------------------------------------------

mod c_ast_expression_precedence_e2e_part1 {

    include!("__split/c_ast_expression_precedence_e2e_part1.rs");
}
mod c_ast_expression_precedence_e2e_part2 {
    include!("__split/c_ast_expression_precedence_e2e_part2.rs");
}
