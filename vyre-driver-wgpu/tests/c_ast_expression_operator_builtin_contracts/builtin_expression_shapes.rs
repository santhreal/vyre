use super::expression_builtin::*;
use crate::c_ast_gpu_parity_support::{assert_expression_shape_parity, word_at};
use crate::c_frontend::expression_pipeline::run_pipeline;
use vyre_libs::parsing::c::parse::vast::C_EXPR_SHAPE_STRIDE_U32;

#[test]
fn builtin_shapes_are_none_not_binary() {
    let fixtures = [
        builtin_constant_p_fixture(),
        builtin_choose_expr_fixture(),
        builtin_types_compatible_p_fixture(),
        generic_selection_fixture(),
    ];

    for (fixture_idx, (tok_types, tok_lens)) in fixtures.iter().enumerate() {
        let rows = run_pipeline(tok_types, tok_lens);
        for idx in 0..tok_types.len() {
            let shape_kind = word_at(&rows.expr_shape, idx * C_EXPR_SHAPE_STRIDE_U32 as usize);
            assert_ne!(
                shape_kind,
                1, // C_EXPR_SHAPE_BINARY
                "Fix: builtin fixture {fixture_idx} row {idx} must not receive BINARY shape"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// GPU / CPU parity
// ---------------------------------------------------------------------------

#[test]
fn gpu_matches_cpu_for_builtin_fixtures() {
    assert_expression_shape_parity(&[
        builtin_constant_p_fixture(),
        builtin_choose_expr_fixture(),
        builtin_types_compatible_p_fixture(),
        generic_selection_fixture(),
        nested_builtin_fixture(),
    ]);
}
