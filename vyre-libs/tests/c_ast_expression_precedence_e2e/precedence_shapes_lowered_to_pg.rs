use super::*;

#[test]
fn comma_boundary_preserves_assignment_shapes_and_lowers_to_pg() {
    let (tok_types, tok_lens) = comma_fixture();
    let rows = run_pipeline(&tok_types, &tok_lens);

    // Assignments at 1, 5 and 9; the commas between them are boundaries only.
    assert_shape_rows(
        &rows.expr_shape,
        &[
            binary_row(1, TOK_ASSIGN, 2, C_EXPR_ASSOC_RIGHT, 0, 2),
            shape_none_row(3, TOK_COMMA),
            binary_row(5, TOK_ASSIGN, 2, C_EXPR_ASSOC_RIGHT, 4, 6),
            shape_none_row(7, TOK_COMMA),
            binary_row(9, TOK_ASSIGN, 2, C_EXPR_ASSOC_RIGHT, 8, 10),
        ],
    );

    assert_eq!(
        row_indices(&rows.typed_vast, VAST_STRIDE_U32, C_AST_KIND_ASSIGN_EXPR),
        vec![1, 5, 9]
    );

    for idx in [1usize, 5, 9] {
        assert_pg_preserves_row(&rows, idx, C_AST_KIND_ASSIGN_EXPR);
        assert_pg_links_match_vast(&rows, idx);
    }
}

#[test]
fn assignment_chain_right_associativity_shapes_and_lowers_to_pg() {
    let (tok_types, tok_lens) = assignment_chain_fixture();
    let rows = run_pipeline(&tok_types, &tok_lens);

    // Right-associative: a = (b = (c = d))
    assert_shape_rows(
        &rows.expr_shape,
        &[
            binary_row(5, TOK_ASSIGN, 2, C_EXPR_ASSOC_RIGHT, 4, 6),
            binary_row(3, TOK_ASSIGN, 2, C_EXPR_ASSOC_RIGHT, 2, 5),
            binary_row(1, TOK_ASSIGN, 2, C_EXPR_ASSOC_RIGHT, 0, 3),
        ],
    );

    assert_eq!(
        row_indices(&rows.typed_vast, VAST_STRIDE_U32, C_AST_KIND_ASSIGN_EXPR),
        vec![1, 3, 5]
    );

    for idx in [1usize, 3, 5] {
        assert_pg_preserves_row(&rows, idx, C_AST_KIND_ASSIGN_EXPR);
        assert_pg_links_match_vast(&rows, idx);
    }
}

#[test]
fn ternary_nesting_right_associativity_shapes_and_lowers_to_pg() {
    let (tok_types, tok_lens) = ternary_nesting_fixture();
    let rows = run_pipeline(&tok_types, &tok_lens);

    // Inner conditional b ? c : d at 3, outer a ? (inner) : e at 1; the colons
    // are boundaries, not shape nodes.
    assert_shape_rows(
        &rows.expr_shape,
        &[
            conditional_row(3, 2, 4, 6),
            conditional_row(1, 0, 3, 8),
            shape_none_row(5, TOK_COLON),
            shape_none_row(7, TOK_COLON),
        ],
    );

    assert_eq!(
        row_indices(
            &rows.typed_vast,
            VAST_STRIDE_U32,
            C_AST_KIND_CONDITIONAL_EXPR
        ),
        vec![1, 3]
    );

    for idx in [1usize, 3] {
        assert_pg_preserves_row(&rows, idx, C_AST_KIND_CONDITIONAL_EXPR);
        assert_pg_links_match_vast(&rows, idx);
    }
}

#[test]
fn logical_and_bitwise_precedence_shapes_and_lowers_to_pg() {
    let (tok_types, tok_lens) = logical_bitwise_fixture();
    let rows = run_pipeline(&tok_types, &tok_lens);

    // Precedence ladder (tightest to loosest): * > + > < > == > & > ^ > | > && > ||
    assert_shape_rows(
        &rows.expr_shape,
        &[
            binary_row(17, TOK_STAR, 13, C_EXPR_ASSOC_LEFT, 16, 18),
            binary_row(15, TOK_PLUS, 12, C_EXPR_ASSOC_LEFT, 14, 17),
            binary_row(13, TOK_LT, 10, C_EXPR_ASSOC_LEFT, 12, 15),
            binary_row(11, TOK_EQ, 9, C_EXPR_ASSOC_LEFT, 10, 13),
            binary_row(9, TOK_AMP, 8, C_EXPR_ASSOC_LEFT, 8, 11),
            binary_row(7, TOK_CARET, 7, C_EXPR_ASSOC_LEFT, 6, 9),
            binary_row(5, TOK_PIPE, 6, C_EXPR_ASSOC_LEFT, 4, 7),
            binary_row(3, TOK_AND, 5, C_EXPR_ASSOC_LEFT, 2, 5),
            binary_row(1, TOK_OR, 4, C_EXPR_ASSOC_LEFT, 0, 3),
        ],
    );

    assert_eq!(
        row_indices(&rows.typed_vast, VAST_STRIDE_U32, node_kind::BINARY),
        vec![1, 3, 5, 7, 9, 11, 13, 15, 17]
    );

    for idx in [1usize, 3, 5, 7, 9, 11, 13, 15, 17] {
        assert_pg_preserves_row(&rows, idx, node_kind::BINARY);
        assert_pg_links_match_vast(&rows, idx);
    }
}

#[test]
fn cast_vs_parenthesized_expression_typing_and_pg_lower() {
    let (tok_types, tok_lens) = cast_vs_paren_fixture();
    let rows = run_pipeline(&tok_types, &tok_lens);

    // (int)a;  -> cast
    assert_eq!(
        word_at(&rows.typed_vast, 0 * VAST_STRIDE_U32),
        C_AST_KIND_CAST_EXPR,
        "Fix: (int) must classify as cast expression"
    );
    // (b + c); -> parenthesized expression, LPAREN stays raw.
    assert_eq!(
        word_at(&rows.typed_vast, 5 * VAST_STRIDE_U32),
        0,
        "Fix: (b + c) must NOT classify as cast"
    );

    // Neither LPAREN carries a shape node; only the plus inside does.
    assert_shape_rows(
        &rows.expr_shape,
        &[
            shape_none_row(0, TOK_LPAREN),
            shape_none_row(5, TOK_LPAREN),
            binary_row(7, TOK_PLUS, 12, C_EXPR_ASSOC_LEFT, 6, 8),
        ],
    );

    assert_pg_preserves_row(&rows, 0, C_AST_KIND_CAST_EXPR);
    assert_pg_preserves_row(&rows, 7, node_kind::BINARY);
    assert_pg_links_match_vast(&rows, 7);
}

#[test]
fn postfix_call_index_member_shapes_and_lowers_to_pg() {
    let (tok_types, tok_lens) = postfix_fixture();
    let rows = run_pipeline(&tok_types, &tok_lens);

    // Postfix operators do not receive expression-shape nodes.
    for idx in [0usize, 6, 11, 15] {
        assert_shape_none(&rows.expr_shape, idx, tok_types[idx]);
    }

    assert_eq!(
        row_indices(&rows.typed_vast, VAST_STRIDE_U32, node_kind::CALL),
        vec![0]
    );
    assert_eq!(
        row_indices(
            &rows.typed_vast,
            VAST_STRIDE_U32,
            C_AST_KIND_ARRAY_SUBSCRIPT_EXPR
        ),
        vec![6]
    );
    assert_eq!(
        row_indices(
            &rows.typed_vast,
            VAST_STRIDE_U32,
            C_AST_KIND_MEMBER_ACCESS_EXPR
        ),
        vec![11, 15]
    );

    for idx in [0usize, 6, 11, 15] {
        let kind = word_at(&rows.typed_vast, idx * VAST_STRIDE_U32);
        assert_pg_preserves_row(&rows, idx, kind);
        assert_pg_links_match_vast(&rows, idx);
    }
}
