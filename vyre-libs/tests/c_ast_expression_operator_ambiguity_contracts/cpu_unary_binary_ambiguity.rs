// Contracts for C expression-operator ambiguity resolution.
//
// Covers the canonical ambiguities that make C expression parsing hard:
//   - `*`, `&`, `+`, `-` appearing as binary vs unary operators
//   - casts `(type-name)expr` vs parenthesized expressions `(expr)`
//   - `sizeof` / `typeof` followed by type-name vs expression
//
// Every fixture asserts both the CPU reference classification and the
// GPU/CPU parity for VAST building, classification, expression-shape rows,
// and PG lowering.  GPU acquisition failures are never silently skipped.

use super::expression_ambiguity::*;
use crate::c_frontend::expression_pipeline::{
    assert_pg_links_match_vast, assert_pg_preserves_row, assert_shape_none, assert_shape_rows,
    run_pipeline,
};
use crate::c_frontend::rows::{row_indices_by_stride, SENTINEL, VAST_STRIDE_U32};
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::parse::vast::{
    C_AST_KIND_UNARY_EXPR, C_EXPR_ASSOC_LEFT, C_EXPR_SHAPE_BINARY,
};
use vyre_primitives::predicate::node_kind;

// ---------------------------------------------------------------------------
// Ambiguity tests
// ---------------------------------------------------------------------------

#[test]
pub(crate) fn star_binary_is_binary_and_unary_is_unary() {
    let (tok_types, tok_lens) = star_binary_fixture();
    let rows = run_pipeline(&tok_types, &tok_lens);

    assert_eq!(
        row_indices_by_stride(&rows.typed_vast, VAST_STRIDE_U32, node_kind::BINARY),
        vec![1],
        "Fix: * in binary context must classify as BINARY"
    );
    assert_shape_rows(
        &rows.expr_shape,
        &[(
            1,
            C_EXPR_SHAPE_BINARY,
            TOK_STAR,
            13,
            C_EXPR_ASSOC_LEFT,
            0,
            2,
            SENTINEL,
        )],
    );
    assert_pg_preserves_row(&rows, 1, node_kind::BINARY);
    assert_pg_links_match_vast(&rows, 1);

    let (tok_types_u, tok_lens_u) = star_unary_fixture();
    let rows_u = run_pipeline(&tok_types_u, &tok_lens_u);
    assert_eq!(
        row_indices_by_stride(&rows_u.typed_vast, VAST_STRIDE_U32, C_AST_KIND_UNARY_EXPR),
        vec![0],
        "Fix: * in unary context must classify as UNARY_EXPR"
    );
    assert_shape_none(&rows_u.expr_shape, 0, TOK_STAR);
    assert_pg_preserves_row(&rows_u, 0, C_AST_KIND_UNARY_EXPR);
    assert_pg_links_match_vast(&rows_u, 0);
}

#[test]
pub(crate) fn amp_binary_is_binary_and_unary_is_unary() {
    let (tok_types, tok_lens) = amp_binary_fixture();
    let rows = run_pipeline(&tok_types, &tok_lens);

    assert_eq!(
        row_indices_by_stride(&rows.typed_vast, VAST_STRIDE_U32, node_kind::BINARY),
        vec![1],
        "Fix: & in binary context must classify as BINARY"
    );
    assert_shape_rows(
        &rows.expr_shape,
        &[(
            1,
            C_EXPR_SHAPE_BINARY,
            TOK_AMP,
            8,
            C_EXPR_ASSOC_LEFT,
            0,
            2,
            SENTINEL,
        )],
    );
    assert_pg_preserves_row(&rows, 1, node_kind::BINARY);
    assert_pg_links_match_vast(&rows, 1);

    let (tok_types_u, tok_lens_u) = amp_unary_fixture();
    let rows_u = run_pipeline(&tok_types_u, &tok_lens_u);
    assert_eq!(
        row_indices_by_stride(&rows_u.typed_vast, VAST_STRIDE_U32, C_AST_KIND_UNARY_EXPR),
        vec![0],
        "Fix: & in unary context must classify as UNARY_EXPR"
    );
    assert_shape_none(&rows_u.expr_shape, 0, TOK_AMP);
    assert_pg_preserves_row(&rows_u, 0, C_AST_KIND_UNARY_EXPR);
    assert_pg_links_match_vast(&rows_u, 0);
}
