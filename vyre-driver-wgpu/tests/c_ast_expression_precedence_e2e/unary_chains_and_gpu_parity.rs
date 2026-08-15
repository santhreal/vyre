use super::*;

#[test]
fn unary_chain_typing_and_pg_lower() {
    let (tok_types, tok_lens) = unary_chain_fixture();
    let rows = run_pipeline(&tok_types, &tok_lens);

    // Unary operators receive NONE shape rows.
    for idx in [0usize, 1, 2, 3, 4, 5] {
        assert_shape_row(
            &rows.expr_shape,
            idx,
            C_EXPR_SHAPE_NONE,
            tok_types[idx],
            0,
            C_EXPR_ASSOC_NONE,
            SENTINEL,
            SENTINEL,
            SENTINEL,
        );
    }

    assert_eq!(
        row_indices(&rows.typed_vast, VAST_STRIDE_U32, C_AST_KIND_UNARY_EXPR),
        vec![0, 1, 2, 3, 4, 5]
    );

    for idx in [0usize, 1, 2, 3, 4, 5] {
        assert_pg_preserves_row(&rows, idx, C_AST_KIND_UNARY_EXPR);
        assert_pg_links_match_vast(&rows, idx);
    }
}

// ---------------------------------------------------------------------------
// GPU / CPU parity test
// ---------------------------------------------------------------------------

#[test]
fn gpu_matches_cpu_for_expression_shape_and_pg_lowering() {
    assert_expression_shape_parity(&[
        comma_fixture(),
        assignment_chain_fixture(),
        ternary_nesting_fixture(),
        logical_bitwise_fixture(),
        cast_vs_paren_fixture(),
        postfix_fixture(),
        unary_chain_fixture(),
    ]);
}
