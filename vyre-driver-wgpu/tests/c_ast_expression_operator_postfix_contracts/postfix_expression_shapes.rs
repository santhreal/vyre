use super::expression_postfix::*;
use crate::c_ast_gpu_parity_support::assert_expression_shape_parity;
use crate::c_frontend::expression_pipeline::{assert_shape_none, run_pipeline};
use crate::c_frontend::rows::{word_at, VAST_STRIDE_U32};
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::parse::vast::C_AST_KIND_UNARY_EXPR;
use vyre_libs::predicate::node_kind;

#[test]
fn postfix_inc_dec_are_not_unary_and_not_binary() {
    let (tok_types_i, tok_lens_i) = postfix_inc_fixture();
    let rows_i = run_pipeline(&tok_types_i, &tok_lens_i);

    let kind_i = word_at(&rows_i.typed_vast, VAST_STRIDE_U32);
    assert_ne!(
        kind_i, C_AST_KIND_UNARY_EXPR,
        "Fix: postfix ++ must NOT be classified as UNARY_EXPR"
    );
    assert_ne!(
        kind_i,
        node_kind::BINARY,
        "Fix: postfix ++ must NOT be classified as BINARY"
    );
    assert_shape_none(&rows_i.expr_shape, 1, TOK_INC);

    let (tok_types_d, tok_lens_d) = postfix_dec_fixture();
    let rows_d = run_pipeline(&tok_types_d, &tok_lens_d);

    let kind_d = word_at(&rows_d.typed_vast, VAST_STRIDE_U32);
    assert_ne!(
        kind_d, C_AST_KIND_UNARY_EXPR,
        "Fix: postfix -- must NOT be classified as UNARY_EXPR"
    );
    assert_ne!(
        kind_d,
        node_kind::BINARY,
        "Fix: postfix -- must NOT be classified as BINARY"
    );
    assert_shape_none(&rows_d.expr_shape, 1, TOK_DEC);
}

// ---------------------------------------------------------------------------
// GPU / CPU parity
// ---------------------------------------------------------------------------

#[test]
fn gpu_matches_cpu_for_postfix_fixtures() {
    assert_expression_shape_parity(&[
        chained_member_fixture(),
        chained_arrow_fixture(),
        mixed_postfix_fixture(),
        unary_deref_fixture(),
        unary_addressof_fixture(),
        gnu_real_fixture(),
        gnu_imag_fixture(),
        label_address_fixture(),
        postfix_inc_fixture(),
        postfix_dec_fixture(),
    ]);
}
