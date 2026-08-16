// End-to-end C parser coverage for expression-shape rows and PG lowering.

use super::expression_shape_pg::*;
use crate::c_frontend::expression_pipeline::{
    assert_pg_preserves_row, assert_shape_rows, binary_row, conditional_row, run_pipeline,
    shape_none_row,
};
use crate::c_frontend::rows::{row_indices_by_stride, VAST_STRIDE_U32};
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::parse::vast::{
    C_AST_KIND_ARRAY_SUBSCRIPT_EXPR, C_AST_KIND_ASSIGN_EXPR, C_AST_KIND_CONDITIONAL_EXPR,
    C_AST_KIND_MEMBER_ACCESS_EXPR, C_AST_KIND_UNARY_EXPR, C_EXPR_ASSOC_LEFT, C_EXPR_ASSOC_RIGHT,
};
use vyre_libs::predicate::node_kind;

#[test]
pub(crate) fn assignment_chain_comma_conditional_member_and_unary_shapes_lower_to_pg() {
    let (tok_types, tok_lens) = expression_chain_fixture();
    let rows = run_pipeline(&tok_types, &tok_lens);

    assert_shape_rows(
        &rows.expr_shape,
        &[
            binary_row(1, TOK_ASSIGN, 2, C_EXPR_ASSOC_RIGHT, 0, 3),
            binary_row(3, TOK_ASSIGN, 2, C_EXPR_ASSOC_RIGHT, 2, 4),
            shape_none_row(5, TOK_COMMA),
            binary_row(7, TOK_ASSIGN, 2, C_EXPR_ASSOC_RIGHT, 6, 8),
            shape_none_row(9, TOK_COMMA),
            conditional_row(11, 10, 12, 14),
            shape_none_row(15, TOK_COMMA),
            binary_row(22, TOK_ASSIGN, 2, C_EXPR_ASSOC_RIGHT, 16, 25),
            binary_row(25, TOK_PLUS, 12, C_EXPR_ASSOC_LEFT, 23, 27),
            binary_row(27, TOK_STAR, 13, C_EXPR_ASSOC_LEFT, 26, 28),
        ],
    );

    assert_eq!(
        row_indices_by_stride(&rows.typed_vast, VAST_STRIDE_U32, C_AST_KIND_ASSIGN_EXPR),
        vec![1, 3, 7, 22]
    );
    assert_eq!(
        row_indices_by_stride(
            &rows.typed_vast,
            VAST_STRIDE_U32,
            C_AST_KIND_CONDITIONAL_EXPR
        ),
        vec![11]
    );
    assert_eq!(
        row_indices_by_stride(
            &rows.typed_vast,
            VAST_STRIDE_U32,
            C_AST_KIND_ARRAY_SUBSCRIPT_EXPR
        ),
        vec![17]
    );
    assert_eq!(
        row_indices_by_stride(
            &rows.typed_vast,
            VAST_STRIDE_U32,
            C_AST_KIND_MEMBER_ACCESS_EXPR
        ),
        vec![20]
    );
    assert_eq!(
        row_indices_by_stride(&rows.typed_vast, VAST_STRIDE_U32, C_AST_KIND_UNARY_EXPR),
        vec![23, 28]
    );
    assert_eq!(
        row_indices_by_stride(&rows.typed_vast, VAST_STRIDE_U32, node_kind::BINARY),
        vec![25, 27]
    );

    for (idx, kind) in [
        (1, C_AST_KIND_ASSIGN_EXPR),
        (3, C_AST_KIND_ASSIGN_EXPR),
        (7, C_AST_KIND_ASSIGN_EXPR),
        (11, C_AST_KIND_CONDITIONAL_EXPR),
        (17, C_AST_KIND_ARRAY_SUBSCRIPT_EXPR),
        (20, C_AST_KIND_MEMBER_ACCESS_EXPR),
        (22, C_AST_KIND_ASSIGN_EXPR),
        (23, C_AST_KIND_UNARY_EXPR),
        (25, node_kind::BINARY),
        (27, node_kind::BINARY),
        (28, C_AST_KIND_UNARY_EXPR),
    ] {
        assert_pg_preserves_row(&rows, idx, kind);
    }
}
