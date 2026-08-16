//! End-to-end C parser coverage for container_of-style cast/member expressions.

#![cfg(feature = "c-parser")]
#![allow(deprecated)]
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::parse::vast::{
    C_AST_KIND_CAST_EXPR, C_AST_KIND_MEMBER_ACCESS_EXPR, C_AST_KIND_POINTER_DECL,
    C_EXPR_ASSOC_LEFT, C_EXPR_SHAPE_BINARY, C_EXPR_SHAPE_NONE, C_EXPR_SHAPE_STRIDE_U32,
};
use vyre_libs::predicate::node_kind;

#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;

use c_frontend::expression_pipeline::{
    assert_kind, assert_pg_links_match_vast, assert_pg_preserves_row, run_pipeline, PipelineRows,
};
use c_frontend::rows::{row_indices_by_stride as row_indices, word_at, VAST_STRIDE_U32};

/// Assert the typed VAST row at `idx` carries the token's own source span.
///
/// [`assert_pg_preserves_row`] pins the lowered property-graph span; this pins
/// the span the classifier wrote, which is where a cast that swallowed its
/// operand's span would show up first.
fn assert_vast_span(rows: &PipelineRows, idx: usize) {
    assert_eq!(
        word_at(&rows.typed_vast, idx * VAST_STRIDE_U32 + 5),
        rows.tok_starts[idx],
        "typed VAST span_start[{idx}]"
    );
    assert_eq!(
        word_at(&rows.typed_vast, idx * VAST_STRIDE_U32 + 6),
        rows.tok_lens[idx],
        "typed VAST span_len[{idx}]"
    );
}

fn assert_binary_shape(rows: &PipelineRows, idx: usize, raw_operator: u32, precedence: u32) {
    let row = idx * C_EXPR_SHAPE_STRIDE_U32 as usize;
    assert_eq!(
        word_at(&rows.expr_shape, row),
        C_EXPR_SHAPE_BINARY,
        "shape_kind[{idx}]"
    );
    assert_eq!(
        word_at(&rows.expr_shape, row + 1),
        idx as u32,
        "source_idx[{idx}]"
    );
    assert_eq!(
        word_at(&rows.expr_shape, row + 2),
        raw_operator,
        "raw_operator[{idx}]"
    );
    assert_eq!(
        word_at(&rows.expr_shape, row + 3),
        precedence,
        "precedence[{idx}]"
    );
    assert_eq!(
        word_at(&rows.expr_shape, row + 4),
        C_EXPR_ASSOC_LEFT,
        "associativity[{idx}]"
    );
}

fn cast_to_pointer_arrow_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_LPAREN,
        TOK_LPAREN,
        TOK_STRUCT,
        TOK_IDENTIFIER,
        TOK_STAR,
        TOK_RPAREN,
        TOK_IDENTIFIER,
        TOK_RPAREN,
        TOK_ARROW,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![1, 1, 6, 4, 1, 1, 1, 1, 2, 4, 1];
    (tok_types, tok_lens)
}

fn nested_char_cast_subtraction_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_LPAREN,
        TOK_LPAREN,
        TOK_CHAR_KW,
        TOK_STAR,
        TOK_RPAREN,
        TOK_IDENTIFIER,
        TOK_MINUS,
        TOK_LPAREN,
        TOK_CHAR_KW,
        TOK_STAR,
        TOK_RPAREN,
        TOK_AMP,
        TOK_LPAREN,
        TOK_LPAREN,
        TOK_STRUCT,
        TOK_IDENTIFIER,
        TOK_STAR,
        TOK_RPAREN,
        TOK_INTEGER,
        TOK_RPAREN,
        TOK_ARROW,
        TOK_IDENTIFIER,
        TOK_RPAREN,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![
        1, 1, 4, 1, 1, 4, 1, 1, 4, 1, 1, 1, 1, 1, 6, 4, 1, 1, 1, 1, 2, 6, 1, 1,
    ];
    (tok_types, tok_lens)
}

#[test]
fn cast_to_pointer_then_arrow_is_cast_pointer_decl_and_member_access() {
    let (tok_types, tok_lens) = cast_to_pointer_arrow_fixture();
    let rows = run_pipeline(&tok_types, &tok_lens);

    assert_eq!(
        row_indices(&rows.typed_vast, VAST_STRIDE_U32, C_AST_KIND_CAST_EXPR),
        vec![1],
        "Fix: ((struct node *)p)->member must classify the type-name paren as a cast"
    );
    assert_eq!(
        row_indices(&rows.typed_vast, VAST_STRIDE_U32, C_AST_KIND_POINTER_DECL),
        vec![4],
        "Fix: star inside the cast type-name must be a pointer declarator"
    );
    assert_eq!(
        row_indices(
            &rows.typed_vast,
            VAST_STRIDE_U32,
            C_AST_KIND_MEMBER_ACCESS_EXPR
        ),
        vec![8],
        "Fix: arrow after the casted pointer must be member access"
    );
    assert!(
        row_indices(&rows.typed_vast, VAST_STRIDE_U32, node_kind::CALL).is_empty(),
        "Fix: cast parentheses must not be typed as CALL nodes"
    );

    for (idx, kind) in [
        (1usize, C_AST_KIND_CAST_EXPR),
        (4, C_AST_KIND_POINTER_DECL),
        (8, C_AST_KIND_MEMBER_ACCESS_EXPR),
    ] {
        assert_vast_span(&rows, idx);
        assert_pg_preserves_row(&rows, idx, kind);
        assert_pg_links_match_vast(&rows, idx);
        assert_eq!(
            word_at(&rows.expr_shape, idx * C_EXPR_SHAPE_STRIDE_U32 as usize),
            C_EXPR_SHAPE_NONE,
            "postfix cast/member rows do not receive binary expression-shape rows"
        );
    }

    assert_eq!(tok_types[8], TOK_ARROW);
    assert_eq!(rows.tok_lens[8], 2);
}

#[test]
fn nested_char_pointer_cast_subtraction_preserves_casts_binary_and_arrow() {
    let (tok_types, tok_lens) = nested_char_cast_subtraction_fixture();
    let rows = run_pipeline(&tok_types, &tok_lens);

    assert_eq!(
        row_indices(&rows.typed_vast, VAST_STRIDE_U32, C_AST_KIND_CAST_EXPR),
        vec![1, 7, 13],
        "Fix: char* and nested struct-pointer casts must all classify as CAST_EXPR"
    );
    assert_eq!(
        row_indices(&rows.typed_vast, VAST_STRIDE_U32, C_AST_KIND_POINTER_DECL),
        vec![3, 9, 16],
        "Fix: every star in the cast type-names must be POINTER_DECL"
    );
    assert_eq!(
        row_indices(
            &rows.typed_vast,
            VAST_STRIDE_U32,
            C_AST_KIND_MEMBER_ACCESS_EXPR
        ),
        vec![20],
        "Fix: nested zero-pointer member selection must remain MEMBER_ACCESS_EXPR"
    );
    assert_kind(&rows.typed_vast, 6, VAST_STRIDE_U32, node_kind::BINARY);
    assert_binary_shape(&rows, 6, TOK_MINUS, 12);
    assert!(
        row_indices(&rows.typed_vast, VAST_STRIDE_U32, node_kind::CALL).is_empty(),
        "Fix: cast-heavy container_of expressions must not manufacture CALL nodes"
    );

    for (idx, kind) in [
        (1usize, C_AST_KIND_CAST_EXPR),
        (3, C_AST_KIND_POINTER_DECL),
        (6, node_kind::BINARY),
        (7, C_AST_KIND_CAST_EXPR),
        (9, C_AST_KIND_POINTER_DECL),
        (13, C_AST_KIND_CAST_EXPR),
        (16, C_AST_KIND_POINTER_DECL),
        (20, C_AST_KIND_MEMBER_ACCESS_EXPR),
    ] {
        assert_vast_span(&rows, idx);
        assert_pg_preserves_row(&rows, idx, kind);
        assert_pg_links_match_vast(&rows, idx);
    }

    assert_eq!(tok_types[6], TOK_MINUS);
    assert_eq!(tok_types[20], TOK_ARROW);
    assert_eq!(rows.tok_lens[20], 2);
}
