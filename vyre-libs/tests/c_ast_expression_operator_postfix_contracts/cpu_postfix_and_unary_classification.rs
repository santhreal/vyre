// Contracts for C postfix and unary expression classification.
//
// Covers:
//   - chained member access (`.` and `->`)
//   - chained array subscript (`[]`)
//   - mixed postfix sequences (`a[i].b->c`)
//   - unary dereference (`*`) and address-of (`&`)
//   - GNU `__real__` and `__imag__`
//   - GNU label-address (`&&label`)
//   - postfix increment/decrement position contracts
//
// GPU/CPU parity and PG lowering preservation are asserted for every fixture.

use super::expression_postfix::*;
use crate::c_frontend::expression_pipeline::{
    assert_pg_links_match_vast, assert_pg_preserves_row, assert_shape_none, run_pipeline,
};
use crate::c_frontend::rows::{row_indices_by_stride as row_indices, word_at, VAST_STRIDE_U32};
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::parse::vast::{
    C_AST_KIND_ARRAY_SUBSCRIPT_EXPR, C_AST_KIND_GNU_LABEL_ADDRESS_EXPR,
    C_AST_KIND_MEMBER_ACCESS_EXPR, C_AST_KIND_UNARY_EXPR,
};
use vyre_primitives::predicate::node_kind;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
pub(crate) fn chained_member_access_classifies_each_dot() {
    let (tok_types, tok_lens) = chained_member_fixture();
    let rows = run_pipeline(&tok_types, &tok_lens);

    assert_eq!(
        row_indices(
            &rows.typed_vast,
            VAST_STRIDE_U32,
            C_AST_KIND_MEMBER_ACCESS_EXPR
        ),
        vec![1, 3],
        "Fix: each . in a.b.c must classify as MEMBER_ACCESS_EXPR"
    );
    assert_shape_none(&rows.expr_shape, 1, TOK_DOT);
    assert_shape_none(&rows.expr_shape, 3, TOK_DOT);
    assert_pg_preserves_row(&rows, 1, C_AST_KIND_MEMBER_ACCESS_EXPR);
    assert_pg_preserves_row(&rows, 3, C_AST_KIND_MEMBER_ACCESS_EXPR);
}

#[test]
pub(crate) fn chained_arrow_access_classifies_each_arrow() {
    let (tok_types, tok_lens) = chained_arrow_fixture();
    let rows = run_pipeline(&tok_types, &tok_lens);

    assert_eq!(
        row_indices(
            &rows.typed_vast,
            VAST_STRIDE_U32,
            C_AST_KIND_MEMBER_ACCESS_EXPR
        ),
        vec![1, 3],
        "Fix: each -> in a->b->c must classify as MEMBER_ACCESS_EXPR"
    );
    assert_shape_none(&rows.expr_shape, 1, TOK_ARROW);
    assert_shape_none(&rows.expr_shape, 3, TOK_ARROW);
    assert_pg_preserves_row(&rows, 1, C_AST_KIND_MEMBER_ACCESS_EXPR);
    assert_pg_preserves_row(&rows, 3, C_AST_KIND_MEMBER_ACCESS_EXPR);
}

#[test]
pub(crate) fn mixed_postfix_member_and_subscript_classifies() {
    let (tok_types, tok_lens) = mixed_postfix_fixture();
    let rows = run_pipeline(&tok_types, &tok_lens);

    assert_eq!(
        row_indices(
            &rows.typed_vast,
            VAST_STRIDE_U32,
            C_AST_KIND_ARRAY_SUBSCRIPT_EXPR
        ),
        vec![1],
        "Fix: [ in a[0].b->c must classify as ARRAY_SUBSCRIPT_EXPR"
    );
    assert_eq!(
        row_indices(
            &rows.typed_vast,
            VAST_STRIDE_U32,
            C_AST_KIND_MEMBER_ACCESS_EXPR
        ),
        vec![4, 6],
        "Fix: . and -> in a[0].b->c must classify as MEMBER_ACCESS_EXPR"
    );
    for idx in [1usize, 4, 6] {
        assert_pg_preserves_row(&rows, idx, word_at(&rows.typed_vast, idx * VAST_STRIDE_U32));
        assert_pg_links_match_vast(&rows, idx);
    }
}

#[test]
pub(crate) fn unary_deref_and_addressof_are_unary_expr() {
    let (tok_types_d, tok_lens_d) = unary_deref_fixture();
    let rows_d = run_pipeline(&tok_types_d, &tok_lens_d);
    assert_eq!(
        row_indices(&rows_d.typed_vast, VAST_STRIDE_U32, C_AST_KIND_UNARY_EXPR),
        vec![0],
        "Fix: *p must classify * as UNARY_EXPR"
    );
    assert_shape_none(&rows_d.expr_shape, 0, TOK_STAR);
    assert_pg_preserves_row(&rows_d, 0, C_AST_KIND_UNARY_EXPR);

    let (tok_types_a, tok_lens_a) = unary_addressof_fixture();
    let rows_a = run_pipeline(&tok_types_a, &tok_lens_a);
    assert_eq!(
        row_indices(&rows_a.typed_vast, VAST_STRIDE_U32, C_AST_KIND_UNARY_EXPR),
        vec![0],
        "Fix: &x must classify & as UNARY_EXPR"
    );
    assert_shape_none(&rows_a.expr_shape, 0, TOK_AMP);
    assert_pg_preserves_row(&rows_a, 0, C_AST_KIND_UNARY_EXPR);
}

#[test]
pub(crate) fn gnu_real_and_imag_are_unary_expr() {
    let (tok_types_r, tok_lens_r) = gnu_real_fixture();
    let rows_r = run_pipeline(&tok_types_r, &tok_lens_r);
    assert_eq!(
        row_indices(&rows_r.typed_vast, VAST_STRIDE_U32, C_AST_KIND_UNARY_EXPR),
        vec![0],
        "Fix: __real__ must classify as UNARY_EXPR"
    );
    assert_shape_none(&rows_r.expr_shape, 0, TOK_GNU_REAL);
    assert_pg_preserves_row(&rows_r, 0, C_AST_KIND_UNARY_EXPR);

    let (tok_types_i, tok_lens_i) = gnu_imag_fixture();
    let rows_i = run_pipeline(&tok_types_i, &tok_lens_i);
    assert_eq!(
        row_indices(&rows_i.typed_vast, VAST_STRIDE_U32, C_AST_KIND_UNARY_EXPR),
        vec![0],
        "Fix: __imag__ must classify as UNARY_EXPR"
    );
    assert_shape_none(&rows_i.expr_shape, 0, TOK_GNU_IMAG);
    assert_pg_preserves_row(&rows_i, 0, C_AST_KIND_UNARY_EXPR);
}

#[test]
pub(crate) fn label_address_expr_classifies_and_lowers() {
    let (tok_types, tok_lens) = label_address_fixture();
    let rows = run_pipeline(&tok_types, &tok_lens);

    assert_eq!(
        row_indices(
            &rows.typed_vast,
            VAST_STRIDE_U32,
            C_AST_KIND_GNU_LABEL_ADDRESS_EXPR
        ),
        vec![0],
        "Fix: &&label must classify as GNU_LABEL_ADDRESS_EXPR"
    );
    assert_ne!(
        word_at(&rows.typed_vast, 0 * VAST_STRIDE_U32),
        node_kind::BINARY,
        "Fix: &&label must not be confused with logical AND"
    );
    assert_shape_none(&rows.expr_shape, 0, TOK_AND);
    assert_pg_preserves_row(&rows, 0, C_AST_KIND_GNU_LABEL_ADDRESS_EXPR);
    assert_pg_links_match_vast(&rows, 0);
}
