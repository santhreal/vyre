// End-to-end C parser coverage for expression-shape rows and PG lowering.

use super::expression_shape_pg::*;
use crate::c_frontend::expression_pipeline::{
    assert_pg_preserves_row, assert_shape_rows, run_pipeline, shape_none_row,
};
use crate::c_frontend::rows::{row_indices_by_stride, SENTINEL, VAST_STRIDE_U32};
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::parse::vast::{
    C_AST_KIND_ARRAY_SUBSCRIPT_EXPR, C_AST_KIND_ASSIGN_EXPR, C_AST_KIND_CONDITIONAL_EXPR,
    C_AST_KIND_MEMBER_ACCESS_EXPR, C_AST_KIND_UNARY_EXPR, C_EXPR_ASSOC_LEFT, C_EXPR_ASSOC_RIGHT,
    C_EXPR_SHAPE_BINARY, C_EXPR_SHAPE_CONDITIONAL,
};
use vyre_primitives::predicate::node_kind;

#[test]
pub(crate) fn assignment_chain_comma_conditional_member_and_unary_shapes_lower_to_pg() {
    let (tok_types, tok_lens) = expression_chain_fixture();
    let rows = run_pipeline(&tok_types, &tok_lens);

    assert_shape_rows(
        &rows.expr_shape,
        &[
            (
                1,
                C_EXPR_SHAPE_BINARY,
                TOK_ASSIGN,
                2,
                C_EXPR_ASSOC_RIGHT,
                0,
                3,
                SENTINEL,
            ),
            (
                3,
                C_EXPR_SHAPE_BINARY,
                TOK_ASSIGN,
                2,
                C_EXPR_ASSOC_RIGHT,
                2,
                4,
                SENTINEL,
            ),
            shape_none_row(5, TOK_COMMA),
            (
                7,
                C_EXPR_SHAPE_BINARY,
                TOK_ASSIGN,
                2,
                C_EXPR_ASSOC_RIGHT,
                6,
                8,
                SENTINEL,
            ),
            shape_none_row(9, TOK_COMMA),
            (
                11,
                C_EXPR_SHAPE_CONDITIONAL,
                TOK_QUESTION,
                3,
                C_EXPR_ASSOC_RIGHT,
                10,
                12,
                14,
            ),
            shape_none_row(15, TOK_COMMA),
            (
                22,
                C_EXPR_SHAPE_BINARY,
                TOK_ASSIGN,
                2,
                C_EXPR_ASSOC_RIGHT,
                16,
                25,
                SENTINEL,
            ),
            (
                25,
                C_EXPR_SHAPE_BINARY,
                TOK_PLUS,
                12,
                C_EXPR_ASSOC_LEFT,
                23,
                27,
                SENTINEL,
            ),
            (
                27,
                C_EXPR_SHAPE_BINARY,
                TOK_STAR,
                13,
                C_EXPR_ASSOC_LEFT,
                26,
                28,
                SENTINEL,
            ),
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
