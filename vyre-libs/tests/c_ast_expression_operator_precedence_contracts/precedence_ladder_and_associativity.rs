use super::*;

#[test]
fn shift_operators_precedence_between_additive_and_relational() {
    let (tok_types, tok_lens) = shift_precedence_fixture();
    let rows = run_pipeline(&tok_types, &tok_lens);

    // a << b + c  ->  << is root because it has lower precedence (11) than + (12)
    assert_shape_rows(
        &rows.expr_shape,
        &[
            binary_row(1, TOK_LSHIFT, 11, C_EXPR_ASSOC_LEFT, 0, 3),
            binary_row(3, TOK_PLUS, 12, C_EXPR_ASSOC_LEFT, 2, 4),
        ],
    );
    assert_pg_preserves_row(&rows, 1, node_kind::BINARY);
    assert_pg_preserves_row(&rows, 3, node_kind::BINARY);
}

#[test]
fn relational_operators_precedence_between_shift_and_equality() {
    let (tok_types, tok_lens) = relational_precedence_fixture();
    let rows = run_pipeline(&tok_types, &tok_lens);

    // a < b << c  ->  < is root (10) looser than << (11)
    assert_shape_rows(
        &rows.expr_shape,
        &[
            binary_row(1, TOK_LT, 10, C_EXPR_ASSOC_LEFT, 0, 3),
            binary_row(3, TOK_LSHIFT, 11, C_EXPR_ASSOC_LEFT, 2, 4),
        ],
    );
    assert_pg_preserves_row(&rows, 1, node_kind::BINARY);
    assert_pg_preserves_row(&rows, 3, node_kind::BINARY);
}

#[test]
fn equality_operators_precedence_between_relational_and_bitwise_and() {
    let (tok_types, tok_lens) = equality_precedence_fixture();
    let rows = run_pipeline(&tok_types, &tok_lens);

    // a == b < c  ->  == is root (9) looser than < (10)
    assert_shape_rows(
        &rows.expr_shape,
        &[
            binary_row(1, TOK_EQ, 9, C_EXPR_ASSOC_LEFT, 0, 3),
            binary_row(3, TOK_LT, 10, C_EXPR_ASSOC_LEFT, 2, 4),
        ],
    );
    assert_pg_preserves_row(&rows, 1, node_kind::BINARY);
    assert_pg_preserves_row(&rows, 3, node_kind::BINARY);
}

#[test]
fn equality_operators_are_left_associative() {
    let (tok_types, tok_lens) = equality_left_assoc_fixture();
    let rows = run_pipeline(&tok_types, &tok_lens);

    // a == b != c  ->  != is root because left-assoc chains group right-to-left:
    // !=(==(a,b), c)
    assert_shape_rows(
        &rows.expr_shape,
        &[
            binary_row(1, TOK_EQ, 9, C_EXPR_ASSOC_LEFT, 0, 2),
            binary_row(3, TOK_NE, 9, C_EXPR_ASSOC_LEFT, 1, 4),
        ],
    );
    assert_pg_preserves_row(&rows, 1, node_kind::BINARY);
    assert_pg_preserves_row(&rows, 3, node_kind::BINARY);
}

#[test]
fn compound_assignment_operators_are_right_associative() {
    let (tok_types, tok_lens) = compound_assignment_fixture();
    let rows = run_pipeline(&tok_types, &tok_lens);

    // a += b -= c  ->  += is root (right-assoc)
    assert_shape_rows(
        &rows.expr_shape,
        &[
            binary_row(1, TOK_PLUS_EQ, 2, C_EXPR_ASSOC_RIGHT, 0, 3),
            binary_row(3, TOK_MINUS_EQ, 2, C_EXPR_ASSOC_RIGHT, 2, 4),
        ],
    );
    assert_eq!(
        row_indices(&rows.typed_vast, VAST_STRIDE_U32, C_AST_KIND_ASSIGN_EXPR),
        vec![1, 3],
        "Fix: compound assignment operators must classify as ASSIGN_EXPR"
    );
    assert_pg_preserves_row(&rows, 1, C_AST_KIND_ASSIGN_EXPR);
    assert_pg_preserves_row(&rows, 3, C_AST_KIND_ASSIGN_EXPR);
}

#[test]
fn ternary_precedence_looser_than_assignment() {
    let (tok_types, tok_lens) = ternary_looser_than_assignment_fixture();
    let rows = run_pipeline(&tok_types, &tok_lens);

    // a = b ? c : d  ->  = is root (prec 2) looser than ? (prec 3).
    // The right operand of = must be the conditional-expression root.
    assert_shape_rows(
        &rows.expr_shape,
        &[
            binary_row(1, TOK_ASSIGN, 2, C_EXPR_ASSOC_RIGHT, 0, 3),
            conditional_row(3, 2, 4, 6),
        ],
    );
    assert_pg_preserves_row(&rows, 1, C_AST_KIND_ASSIGN_EXPR);
    assert_pg_preserves_row(&rows, 3, C_AST_KIND_CONDITIONAL_EXPR);
}

#[test]
fn ternary_right_associativity_chains_correctly() {
    let (tok_types, tok_lens) = ternary_right_assoc_fixture();
    let rows = run_pipeline(&tok_types, &tok_lens);

    // Outer ? at 1 is a ? b : (inner); inner ? at 5 is c ? d : e.
    assert_shape_rows(
        &rows.expr_shape,
        &[conditional_row(1, 0, 2, 5), conditional_row(5, 4, 6, 8)],
    );
    assert_pg_preserves_row(&rows, 1, C_AST_KIND_CONDITIONAL_EXPR);
    assert_pg_preserves_row(&rows, 5, C_AST_KIND_CONDITIONAL_EXPR);
}

#[test]
fn comma_is_expression_boundary_with_no_shape() {
    let (tok_types, tok_lens) = comma_boundary_fixture();
    let rows = run_pipeline(&tok_types, &tok_lens);

    assert_shape_rows(
        &rows.expr_shape,
        &[
            binary_row(1, TOK_ASSIGN, 2, C_EXPR_ASSOC_RIGHT, 0, 2),
            shape_none_row(3, TOK_COMMA),
            binary_row(5, TOK_ASSIGN, 2, C_EXPR_ASSOC_RIGHT, 4, 6),
        ],
    );
    assert_pg_preserves_row(&rows, 1, C_AST_KIND_ASSIGN_EXPR);
    assert_pg_preserves_row(&rows, 5, C_AST_KIND_ASSIGN_EXPR);
}

#[test]
fn full_precedence_ladder_from_star_to_or() {
    let (tok_types, tok_lens) = full_precedence_ladder_fixture();
    let rows = run_pipeline(&tok_types, &tok_lens);

    // Tightest to loosest, root selection validates every band.
    assert_shape_rows(
        &rows.expr_shape,
        &[
            binary_row(19, TOK_STAR, 13, C_EXPR_ASSOC_LEFT, 18, 20),
            binary_row(17, TOK_LSHIFT, 11, C_EXPR_ASSOC_LEFT, 15, 19),
            binary_row(15, TOK_PLUS, 12, C_EXPR_ASSOC_LEFT, 14, 16),
            binary_row(13, TOK_LT, 10, C_EXPR_ASSOC_LEFT, 12, 17),
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
        vec![1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    );

    for idx in [1usize, 3, 5, 7, 9, 11, 13, 15, 17, 19] {
        assert_pg_preserves_row(&rows, idx, node_kind::BINARY);
        assert_pg_links_match_vast(&rows, idx);
    }
}
