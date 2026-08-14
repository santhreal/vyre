// Contracts for C builtin and generic-selection expression classification.
//
// Covers:
//   - `__builtin_constant_p`
//   - `__builtin_choose_expr`
//   - `__builtin_types_compatible_p`
//   - C11 `_Generic`
//   - Nested builtin/generic combinations
//
// Every test asserts that these expressions receive distinct VAST kinds and
// do NOT collapse into generic `CALL` or `BINARY`.  GPU/CPU parity and PG
// lowering preservation are asserted for every fixture.

use super::expression_builtin::*;
use crate::c_frontend::expression_pipeline::{
    assert_pg_links_match_vast, assert_pg_preserves_row, assert_shape_none, run_pipeline,
};
use crate::c_frontend::rows::{row_indices_by_stride, word_at, VAST_STRIDE_U32};
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::parse::vast::{
    C_AST_KIND_BUILTIN_CHOOSE_EXPR, C_AST_KIND_BUILTIN_CONSTANT_P_EXPR,
    C_AST_KIND_BUILTIN_TYPES_COMPATIBLE_P_EXPR, C_AST_KIND_GENERIC_SELECTION_EXPR,
};
use vyre_primitives::predicate::node_kind;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
pub(crate) fn builtin_constant_p_classifies_as_distinct_expr() {
    let (tok_types, tok_lens) = builtin_constant_p_fixture();
    let rows = run_pipeline(&tok_types, &tok_lens);

    assert_eq!(
        row_indices_by_stride(
            &rows.typed_vast,
            VAST_STRIDE_U32,
            C_AST_KIND_BUILTIN_CONSTANT_P_EXPR
        ),
        vec![0],
        "Fix: __builtin_constant_p must be a distinct expression kind"
    );
    assert_ne!(
        word_at(&rows.typed_vast, 0 * VAST_STRIDE_U32),
        node_kind::CALL,
        "Fix: __builtin_constant_p must not collapse into CALL"
    );
    assert_shape_none(&rows.expr_shape, 0, TOK_BUILTIN_CONSTANT_P);
    assert_pg_preserves_row(&rows, 0, C_AST_KIND_BUILTIN_CONSTANT_P_EXPR);
    assert_pg_links_match_vast(&rows, 0);
}

#[test]
pub(crate) fn builtin_choose_expr_classifies_as_distinct_expr() {
    let (tok_types, tok_lens) = builtin_choose_expr_fixture();
    let rows = run_pipeline(&tok_types, &tok_lens);

    assert_eq!(
        row_indices_by_stride(
            &rows.typed_vast,
            VAST_STRIDE_U32,
            C_AST_KIND_BUILTIN_CHOOSE_EXPR
        ),
        vec![0],
        "Fix: __builtin_choose_expr must be a distinct expression kind"
    );
    assert_ne!(
        word_at(&rows.typed_vast, 0 * VAST_STRIDE_U32),
        node_kind::CALL,
        "Fix: __builtin_choose_expr must not collapse into CALL"
    );
    assert_shape_none(&rows.expr_shape, 0, TOK_BUILTIN_CHOOSE_EXPR);
    assert_pg_preserves_row(&rows, 0, C_AST_KIND_BUILTIN_CHOOSE_EXPR);
    assert_pg_links_match_vast(&rows, 0);
}

#[test]
pub(crate) fn builtin_types_compatible_p_classifies_as_distinct_expr() {
    let (tok_types, tok_lens) = builtin_types_compatible_p_fixture();
    let rows = run_pipeline(&tok_types, &tok_lens);

    assert_eq!(
        row_indices_by_stride(
            &rows.typed_vast,
            VAST_STRIDE_U32,
            C_AST_KIND_BUILTIN_TYPES_COMPATIBLE_P_EXPR
        ),
        vec![0],
        "Fix: __builtin_types_compatible_p must be a distinct expression kind"
    );
    assert_ne!(
        word_at(&rows.typed_vast, 0 * VAST_STRIDE_U32),
        node_kind::CALL,
        "Fix: __builtin_types_compatible_p must not collapse into CALL"
    );
    assert_shape_none(&rows.expr_shape, 0, TOK_BUILTIN_TYPES_COMPATIBLE_P);
    assert_pg_preserves_row(&rows, 0, C_AST_KIND_BUILTIN_TYPES_COMPATIBLE_P_EXPR);
    assert_pg_links_match_vast(&rows, 0);
}

#[test]
pub(crate) fn generic_selection_classifies_as_distinct_expr() {
    let (tok_types, tok_lens) = generic_selection_fixture();
    let rows = run_pipeline(&tok_types, &tok_lens);

    assert_eq!(
        row_indices_by_stride(
            &rows.typed_vast,
            VAST_STRIDE_U32,
            C_AST_KIND_GENERIC_SELECTION_EXPR
        ),
        vec![0],
        "Fix: _Generic must be a distinct selection-expression kind"
    );
    assert_ne!(
        word_at(&rows.typed_vast, 0 * VAST_STRIDE_U32),
        node_kind::CALL,
        "Fix: _Generic must not collapse into CALL"
    );
    assert_ne!(
        word_at(&rows.typed_vast, 0 * VAST_STRIDE_U32),
        node_kind::BINARY,
        "Fix: _Generic must not collapse into BINARY"
    );
    assert_shape_none(&rows.expr_shape, 0, TOK_GENERIC);
    assert_pg_preserves_row(&rows, 0, C_AST_KIND_GENERIC_SELECTION_EXPR);
    assert_pg_links_match_vast(&rows, 0);
}

#[test]
pub(crate) fn nested_builtin_and_generic_expressions_classify_correctly() {
    let (tok_types, tok_lens) = nested_builtin_fixture();
    let rows = run_pipeline(&tok_types, &tok_lens);

    assert_eq!(
        row_indices_by_stride(
            &rows.typed_vast,
            VAST_STRIDE_U32,
            C_AST_KIND_BUILTIN_CHOOSE_EXPR
        ),
        vec![0],
        "Fix: outer __builtin_choose_expr must classify"
    );
    assert_eq!(
        row_indices_by_stride(
            &rows.typed_vast,
            VAST_STRIDE_U32,
            C_AST_KIND_BUILTIN_CONSTANT_P_EXPR
        ),
        vec![4],
        "Fix: nested __builtin_constant_p must classify"
    );

    for idx in [0usize, 4] {
        assert_ne!(
            word_at(&rows.typed_vast, idx * VAST_STRIDE_U32),
            node_kind::CALL,
            "Fix: builtin row {idx} must not collapse into CALL"
        );
    }

    assert_pg_preserves_row(&rows, 0, C_AST_KIND_BUILTIN_CHOOSE_EXPR);
    assert_pg_preserves_row(&rows, 4, C_AST_KIND_BUILTIN_CONSTANT_P_EXPR);
    assert_pg_links_match_vast(&rows, 0);
    assert_pg_links_match_vast(&rows, 4);
}
