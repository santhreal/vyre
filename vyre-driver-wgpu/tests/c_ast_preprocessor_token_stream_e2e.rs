//! Preprocessing-token streams must survive lex → VAST → classification →
//! expression-shape → PG lowering **without** macro expansion: directive rows stay
//! `TOK_PREPROC` in raw VAST, `__LINE__` / `__FILE__` stay ordinary identifiers, and
//! macro-shaped calls stay `CALL` sites.

#![cfg(feature = "c-parser")]
#![allow(deprecated)]
#[path = "c_ast_gpu_parity_support/mod.rs"]
mod c_ast_gpu_parity_support;
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
mod c_token_support;

use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::predicate::node_kind;

use c_ast_gpu_parity_support::{
    run_gpu_classifier, run_gpu_expr_shape, run_gpu_pg_lower, word_at, VAST_STRIDE_U32,
};
use c_frontend::token_fixture::c_fixture;
use c_token_support::{
    assert_lex_matches_non_ws, assert_pg_row, assert_shape_none, find_row_for_lexeme,
    row_typed_kind, run_cpu_pipeline as cpu_pipeline_buffers, Fixture, PipelineRows,
};

/// The CPU pipeline plus this suite's own guard that the fixture's lexemes
/// re-lex to the kinds it declares.
fn run_cpu_pipeline(assembled: &Fixture) -> PipelineRows {
    assert_lex_matches_non_ws(assembled);
    cpu_pipeline_buffers(assembled)
}

#[test]
fn preprocessor_directive_rows_keep_preproc_raw_kind_and_survive_pg() {
    let a = c_fixture![
        ("#ifndef FOO", TOK_PREPROC),
        ("\n", TOK_WHITESPACE),
        ("#define FOO 1", TOK_PREPROC),
        ("\n", TOK_WHITESPACE),
        ("int", TOK_INT),
        ("x", TOK_IDENTIFIER),
        (";", TOK_SEMICOLON),
    ];
    let out = run_cpu_pipeline(&a);
    for idx in [0usize, 1] {
        assert_eq!(
            word_at(&out.raw_vast, idx * VAST_STRIDE_U32),
            TOK_PREPROC,
            "raw VAST must preserve TOK_PREPROC (no expansion)"
        );
        assert_eq!(row_typed_kind(&out.typed_vast, idx), 0);
        assert_pg_row(&a, &out.pg_nodes, &out.typed_vast, idx, 0);
        assert_shape_none(&out.expr_shape, idx);
    }
    assert_eq!(
        word_at(&out.raw_vast, 2 * VAST_STRIDE_U32),
        TOK_INT,
        "`int` must stay keyword-promoted in raw VAST"
    );
    assert_eq!(
        row_typed_kind(&out.typed_vast, 2),
        0,
        "type-keyword rows stay unclassified (kind 0) in typed VAST"
    );
    assert_pg_row(&a, &out.pg_nodes, &out.typed_vast, 2, 0);
    assert_pg_row(&a, &out.pg_nodes, &out.typed_vast, 3, node_kind::VARIABLE);
}

#[test]
fn conditional_directive_token_rows_survive_without_expansion() {
    let a = c_fixture![
        ("#if 0", TOK_PREPROC),
        ("\n", TOK_WHITESPACE),
        ("#elif 1", TOK_PREPROC),
        ("\n", TOK_WHITESPACE),
        ("#else", TOK_PREPROC),
        ("\n", TOK_WHITESPACE),
        ("#endif", TOK_PREPROC),
        ("\n", TOK_WHITESPACE),
        ("int", TOK_INT),
        ("q", TOK_IDENTIFIER),
        (";", TOK_SEMICOLON),
    ];
    let out = run_cpu_pipeline(&a);
    for idx in 0..4 {
        assert_eq!(
            word_at(&out.raw_vast, idx * VAST_STRIDE_U32),
            TOK_PREPROC,
            "conditional directive row {idx}"
        );
        assert_eq!(row_typed_kind(&out.typed_vast, idx), 0);
        assert_pg_row(&a, &out.pg_nodes, &out.typed_vast, idx, 0);
        assert_shape_none(&out.expr_shape, idx);
    }
}

#[test]
fn macro_shaped_call_survives_as_call_without_expansion() {
    // Split declaration from assignment so `SUM(` cannot inherit a declaration
    // prefix from `int y =` (which classifies the identifier as FUNCTION_DECL).
    let a = c_fixture![
        ("int", TOK_INT),
        ("y", TOK_IDENTIFIER),
        (";", TOK_SEMICOLON),
        ("y", TOK_IDENTIFIER),
        ("=", TOK_ASSIGN),
        ("SUM", TOK_IDENTIFIER),
        ("(", TOK_LPAREN),
        ("1", TOK_INTEGER),
        (",", TOK_COMMA),
        ("2", TOK_INTEGER),
        (")", TOK_RPAREN),
        (";", TOK_SEMICOLON),
    ];
    let out = run_cpu_pipeline(&a);
    let sum_idx = find_row_for_lexeme(&a, "SUM");
    assert_eq!(row_typed_kind(&out.typed_vast, sum_idx), node_kind::CALL);
    assert_pg_row(&a, &out.pg_nodes, &out.typed_vast, sum_idx, node_kind::CALL);
}

#[test]
fn line_and_file_spellings_remain_identifier_variables() {
    let a = c_fixture![
        ("int", TOK_INT),
        ("ln", TOK_IDENTIFIER),
        ("=", TOK_ASSIGN),
        ("__LINE__", TOK_IDENTIFIER),
        (";", TOK_SEMICOLON),
        ("const", TOK_CONST),
        ("char", TOK_CHAR_KW),
        ("*", TOK_STAR),
        ("fp", TOK_IDENTIFIER),
        ("=", TOK_ASSIGN),
        ("__FILE__", TOK_IDENTIFIER),
        (";", TOK_SEMICOLON),
    ];
    let out = run_cpu_pipeline(&a);
    let line_idx = find_row_for_lexeme(&a, "__LINE__");
    let file_idx = find_row_for_lexeme(&a, "__FILE__");
    let ls = a.tok_starts[line_idx] as usize;
    let fs = a.tok_starts[file_idx] as usize;
    assert_eq!(&a.source.as_bytes()[ls..ls + 8], b"__LINE__");
    assert_eq!(&a.source.as_bytes()[fs..fs + 8], b"__FILE__");
    assert_eq!(
        row_typed_kind(&out.typed_vast, line_idx),
        node_kind::VARIABLE
    );
    assert_eq!(
        row_typed_kind(&out.typed_vast, file_idx),
        node_kind::VARIABLE
    );
    assert_pg_row(
        &a,
        &out.pg_nodes,
        &out.typed_vast,
        line_idx,
        node_kind::VARIABLE,
    );
    assert_pg_row(
        &a,
        &out.pg_nodes,
        &out.typed_vast,
        file_idx,
        node_kind::VARIABLE,
    );
}

#[test]
fn macro_statement_call_inside_compound_survives_as_call() {
    let a = c_fixture![
        ("void", TOK_VOID),
        ("f", TOK_IDENTIFIER),
        ("(", TOK_LPAREN),
        ("void", TOK_VOID),
        (")", TOK_RPAREN),
        ("{", TOK_LBRACE),
        ("\n", TOK_WHITESPACE),
        ("LOCK", TOK_IDENTIFIER),
        ("(", TOK_LPAREN),
        (")", TOK_RPAREN),
        (";", TOK_SEMICOLON),
        ("\n", TOK_WHITESPACE),
        ("}", TOK_RBRACE),
    ];
    let out = run_cpu_pipeline(&a);
    let lock_idx = find_row_for_lexeme(&a, "LOCK");
    assert_eq!(row_typed_kind(&out.typed_vast, lock_idx), node_kind::CALL);
    assert_pg_row(
        &a,
        &out.pg_nodes,
        &out.typed_vast,
        lock_idx,
        node_kind::CALL,
    );
}

#[test]
fn gpu_matches_cpu_for_classify_expr_shape_and_pg_on_preproc_stream() {
    let a = c_fixture![
        ("#define M(x) x", TOK_PREPROC),
        ("\n", TOK_WHITESPACE),
        ("int", TOK_INT),
        ("z", TOK_IDENTIFIER),
        (";", TOK_SEMICOLON),
        ("z", TOK_IDENTIFIER),
        ("=", TOK_ASSIGN),
        ("M", TOK_IDENTIFIER),
        ("(", TOK_LPAREN),
        ("42", TOK_INTEGER),
        (")", TOK_RPAREN),
        (";", TOK_SEMICOLON),
    ];
    let out = run_cpu_pipeline(&a);
    let gpu_typed = run_gpu_classifier(&out.raw_vast);
    assert_eq!(gpu_typed, out.typed_vast, "GPU classifier must match CPU");

    assert_eq!(
        run_gpu_expr_shape(&out.raw_vast, &out.typed_vast),
        out.expr_shape,
        "GPU expression-shape must match CPU"
    );
    assert_eq!(
        run_gpu_pg_lower(&out.typed_vast),
        out.pg_nodes,
        "GPU PG lowering must match CPU"
    );

    let m_idx = find_row_for_lexeme(&a, "M");
    assert_eq!(row_typed_kind(&out.typed_vast, m_idx), node_kind::CALL);
    assert_eq!(
        word_at(&out.raw_vast, m_idx * VAST_STRIDE_U32),
        TOK_IDENTIFIER
    );
    assert_eq!(
        word_at(&out.raw_vast, 0),
        TOK_PREPROC,
        "directive row stays TOK_PREPROC in raw VAST"
    );
}
