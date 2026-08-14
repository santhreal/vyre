//! The CPU reference pipeline every C expression contract test runs.
//!
//! Tokens in, then VAST build, type-aware classification, expression-shape
//! rows, and property-graph lowering, with the executable lowerer checked
//! against the byte oracle on every call. Tests assert against the resulting
//! [`PipelineRows`]; nothing here touches a backend.

use vyre::ir::Expr;
use vyre_libs::parsing::c::lex::tokens::TOK_QUESTION;
use vyre_libs::parsing::c::lower::{c_lower_ast_to_pg_nodes, reference_ast_to_pg_nodes};
use vyre_libs::parsing::c::parse::vast::{
    reference_c11_build_expression_shape_nodes, reference_c11_build_vast_nodes,
    reference_c11_classify_vast_node_kinds, C_EXPR_ASSOC_NONE, C_EXPR_ASSOC_RIGHT,
    C_EXPR_SHAPE_BINARY, C_EXPR_SHAPE_CONDITIONAL, C_EXPR_SHAPE_NONE, C_EXPR_SHAPE_STRIDE_U32,
};
use vyre_reference::value::Value;

use super::rows::{
    node_count_from_vast, starts_for_lens, word_at, PG_STRIDE_U32, SENTINEL, VAST_STRIDE_U32,
};
use super::token_fixture::Fixture;

pub(crate) struct PipelineRows {
    pub(crate) tok_starts: Vec<u32>,
    pub(crate) tok_lens: Vec<u32>,
    pub(crate) raw_vast: Vec<u8>,
    pub(crate) typed_vast: Vec<u8>,
    pub(crate) expr_shape: Vec<u8>,
    pub(crate) pg_nodes: Vec<u8>,
}

/// `(tok_types, tok_lens)` for single-character lexemes.
pub(crate) fn unit_lens_fixture(tok_types: Vec<u32>) -> (Vec<u32>, Vec<u32>) {
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

pub(crate) fn run_reference_pg_lower(typed_vast: &[u8]) -> Vec<u8> {
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

/// The pipeline for a token stream whose lexemes are single characters laid out
/// one unit apart, which is what an expression-shape contract needs.
pub(crate) fn run_pipeline(tok_types: &[u32], tok_lens: &[u32]) -> PipelineRows {
    run_pipeline_from_parts(tok_types, &starts_for_lens(tok_lens), tok_lens)
}

/// The pipeline for a [`Fixture`], which carries the source offsets its own
/// lexemes occupy. A preprocessor or corpus contract needs those offsets, not
/// the unit spacing [`run_pipeline`] derives.
pub(crate) fn run_pipeline_for_fixture(fix: &Fixture) -> PipelineRows {
    run_pipeline_from_parts(&fix.tok_types, &fix.tok_starts, &fix.tok_lens)
}

pub(crate) fn run_pipeline_from_parts(
    tok_types: &[u32],
    tok_starts: &[u32],
    tok_lens: &[u32],
) -> PipelineRows {
    let raw_vast = reference_c11_build_vast_nodes(tok_types, tok_starts, tok_lens);
    let typed_vast = reference_c11_classify_vast_node_kinds(&raw_vast);
    let expr_shape = reference_c11_build_expression_shape_nodes(&raw_vast, &typed_vast);
    let pg_nodes = run_reference_pg_lower(&typed_vast);
    assert_eq!(
        pg_nodes,
        reference_ast_to_pg_nodes(&typed_vast),
        "Fix: executable PG lowerer must match the byte oracle"
    );

    PipelineRows {
        tok_starts: tok_starts.to_vec(),
        tok_lens: tok_lens.to_vec(),
        raw_vast,
        typed_vast,
        expr_shape,
        pg_nodes,
    }
}

pub(crate) fn assert_kind(rows: &[u8], idx: usize, stride_words: usize, kind: u32) {
    assert_eq!(word_at(rows, idx * stride_words), kind, "kind at row {idx}");
}

/// Assert the expression-shape row at `idx` is the "not an operator" row.
pub(crate) fn assert_shape_none(rows: &[u8], idx: usize, raw_operator: u32) {
    let row = idx * C_EXPR_SHAPE_STRIDE_U32 as usize;
    assert_eq!(word_at(rows, row), C_EXPR_SHAPE_NONE, "shape_kind[{idx}]");
    assert_eq!(word_at(rows, row + 1), SENTINEL, "source_idx[{idx}]");
    assert_eq!(word_at(rows, row + 2), raw_operator, "raw_operator[{idx}]");
    assert_eq!(word_at(rows, row + 3), 0, "precedence[{idx}]");
    assert_eq!(
        word_at(rows, row + 4),
        C_EXPR_ASSOC_NONE,
        "associativity[{idx}]"
    );
    assert_eq!(word_at(rows, row + 5), SENTINEL, "first[{idx}]");
    assert_eq!(word_at(rows, row + 6), SENTINEL, "second[{idx}]");
    assert_eq!(word_at(rows, row + 7), SENTINEL, "third[{idx}]");
}

pub(crate) fn assert_pg_preserves_row(rows: &PipelineRows, idx: usize, kind: u32) {
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

pub(crate) fn assert_pg_links_match_vast(rows: &PipelineRows, idx: usize) {
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn assert_shape_row(
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

/// One expected expression-shape row for [`assert_shape_rows`]:
/// `(idx, shape_kind, raw_operator, precedence, associativity, first, second, third)`.
pub(crate) type ShapeRow = (usize, u32, u32, u32, u32, u32, u32, u32);

/// The expected row for a token that only closes or separates an expression and
/// so carries no shape node of its own.
pub(crate) fn shape_none_row(idx: usize, raw_operator: u32) -> ShapeRow {
    (
        idx,
        C_EXPR_SHAPE_NONE,
        raw_operator,
        0,
        C_EXPR_ASSOC_NONE,
        SENTINEL,
        SENTINEL,
        SENTINEL,
    )
}

/// The expected row for a binary operator: two operands and no third.
pub(crate) fn binary_row(
    idx: usize,
    raw_operator: u32,
    precedence: u32,
    associativity: u32,
    first: u32,
    second: u32,
) -> ShapeRow {
    (
        idx,
        C_EXPR_SHAPE_BINARY,
        raw_operator,
        precedence,
        associativity,
        first,
        second,
        SENTINEL,
    )
}

/// Precedence band the C grammar gives the ternary conditional.
const C_CONDITIONAL_PRECEDENCE: u32 = 3;

/// The expected row for a ternary conditional. Its operator spelling,
/// precedence and associativity are fixed by the grammar, so only the three
/// operand links vary per fixture.
pub(crate) fn conditional_row(
    idx: usize,
    condition: u32,
    consequent: u32,
    alternative: u32,
) -> ShapeRow {
    (
        idx,
        C_EXPR_SHAPE_CONDITIONAL,
        TOK_QUESTION,
        C_CONDITIONAL_PRECEDENCE,
        C_EXPR_ASSOC_RIGHT,
        condition,
        consequent,
        alternative,
    )
}

/// Assert a whole table of expected shape rows in order.
///
/// A precedence or associativity contract is a table of rows, so it reads as
/// one, and each row still goes through [`assert_shape_row`] so a mismatch
/// names the offending field and row index.
pub(crate) fn assert_shape_rows(rows: &[u8], expected: &[ShapeRow]) {
    for &(idx, shape_kind, raw_operator, precedence, associativity, first, second, third) in
        expected
    {
        assert_shape_row(
            rows,
            idx,
            shape_kind,
            raw_operator,
            precedence,
            associativity,
            first,
            second,
            third,
        );
    }
}
