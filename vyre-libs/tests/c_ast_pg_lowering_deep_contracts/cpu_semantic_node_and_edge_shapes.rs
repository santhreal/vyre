// Deep C AST-to-ProgramGraph semantic lowering contracts.

use super::pg_lowering_deep_constructs::*;
use crate::c_frontend::rows::{row_indices, word_at, VAST_STRIDE_U32};
use crate::c_frontend::semantic_graph::{
    assert_parent_edge, assert_semantic_node, semantic_edge_word,
};
use crate::c_frontend::token_fixture::classify;
use vyre_libs::parsing::c::lower::{
    reference_ast_to_pg_semantic_graph, C_AST_PG_CATEGORY_CONTROL, C_AST_PG_CATEGORY_EXPRESSION,
    C_AST_PG_EDGE_FIRST_CHILD, C_AST_PG_ROLE_ARRAY_DESIGNATOR_OR_SUBSCRIPT,
    C_AST_PG_ROLE_ASSIGNMENT, C_AST_PG_ROLE_CASE, C_AST_PG_ROLE_DEFAULT,
    C_AST_PG_ROLE_FIELD_DESIGNATOR_OR_MEMBER_ACCESS, C_AST_PG_ROLE_INITIALIZER_LIST,
    C_AST_PG_ROLE_LABEL, C_AST_PG_ROLE_STATEMENT_EXPR,
};
use vyre_libs::parsing::c::parse::vast::{
    C_AST_KIND_ARRAY_SUBSCRIPT_EXPR, C_AST_KIND_ASSIGN_EXPR, C_AST_KIND_CASE_STMT,
    C_AST_KIND_DEFAULT_STMT, C_AST_KIND_GNU_STATEMENT_EXPR, C_AST_KIND_INITIALIZER_LIST,
    C_AST_KIND_LABEL_STMT, C_AST_KIND_MEMBER_ACCESS_EXPR,
};

#[test]
pub(crate) fn labels_case_and_default_have_semantic_node_and_edge_witnesses() {
    let typed = classify(&fixture_label_case_default());
    let semantic = reference_ast_to_pg_semantic_graph(&typed);

    let case_idx = row_indices(&typed, C_AST_KIND_CASE_STMT)[0];
    let label_idx = row_indices(&typed, C_AST_KIND_LABEL_STMT)[0];
    let default_idx = row_indices(&typed, C_AST_KIND_DEFAULT_STMT)[0];

    assert_semantic_node(
        &semantic.nodes,
        case_idx,
        C_AST_KIND_CASE_STMT,
        C_AST_PG_CATEGORY_CONTROL,
        C_AST_PG_ROLE_CASE,
    );
    assert_semantic_node(
        &semantic.nodes,
        label_idx,
        C_AST_KIND_LABEL_STMT,
        C_AST_PG_CATEGORY_CONTROL,
        C_AST_PG_ROLE_LABEL,
    );
    assert_semantic_node(
        &semantic.nodes,
        default_idx,
        C_AST_KIND_DEFAULT_STMT,
        C_AST_PG_CATEGORY_CONTROL,
        C_AST_PG_ROLE_DEFAULT,
    );

    assert_parent_edge(
        &semantic.edges,
        case_idx,
        word_at(&typed, case_idx * VAST_STRIDE_U32 + 1),
        C_AST_PG_ROLE_CASE,
        C_AST_PG_CATEGORY_CONTROL,
    );
    assert_parent_edge(
        &semantic.edges,
        label_idx,
        word_at(&typed, label_idx * VAST_STRIDE_U32 + 1),
        C_AST_PG_ROLE_LABEL,
        C_AST_PG_CATEGORY_CONTROL,
    );
}

#[test]
pub(crate) fn statement_expression_has_expression_category_and_first_child_edge() {
    let typed = classify(&fixture_statement_expr());
    let semantic = reference_ast_to_pg_semantic_graph(&typed);
    let stmt_idx = row_indices(&typed, C_AST_KIND_GNU_STATEMENT_EXPR)[0];

    assert_semantic_node(
        &semantic.nodes,
        stmt_idx,
        C_AST_KIND_GNU_STATEMENT_EXPR,
        C_AST_PG_CATEGORY_EXPRESSION,
        C_AST_PG_ROLE_STATEMENT_EXPR,
    );
    assert_eq!(
        semantic_edge_word(&semantic.edges, stmt_idx, 1, 0),
        C_AST_PG_EDGE_FIRST_CHILD,
        "statement expression must retain first-child graph edge"
    );
    assert_eq!(
        semantic_edge_word(&semantic.edges, stmt_idx, 1, 2),
        word_at(&typed, stmt_idx * VAST_STRIDE_U32 + 2),
        "statement expression child edge must point at the VAST child"
    );
}

#[test]
pub(crate) fn initializer_designators_have_stable_roles() {
    let typed = classify(&fixture_initializer_designator());
    let semantic = reference_ast_to_pg_semantic_graph(&typed);

    for idx in row_indices(&typed, C_AST_KIND_INITIALIZER_LIST) {
        assert_semantic_node(
            &semantic.nodes,
            idx,
            C_AST_KIND_INITIALIZER_LIST,
            C_AST_PG_CATEGORY_EXPRESSION,
            C_AST_PG_ROLE_INITIALIZER_LIST,
        );
    }
    for idx in row_indices(&typed, C_AST_KIND_MEMBER_ACCESS_EXPR) {
        assert_semantic_node(
            &semantic.nodes,
            idx,
            C_AST_KIND_MEMBER_ACCESS_EXPR,
            C_AST_PG_CATEGORY_EXPRESSION,
            C_AST_PG_ROLE_FIELD_DESIGNATOR_OR_MEMBER_ACCESS,
        );
    }
    for idx in row_indices(&typed, C_AST_KIND_ARRAY_SUBSCRIPT_EXPR) {
        assert_semantic_node(
            &semantic.nodes,
            idx,
            C_AST_KIND_ARRAY_SUBSCRIPT_EXPR,
            C_AST_PG_CATEGORY_EXPRESSION,
            C_AST_PG_ROLE_ARRAY_DESIGNATOR_OR_SUBSCRIPT,
        );
    }
    for idx in row_indices(&typed, C_AST_KIND_ASSIGN_EXPR) {
        assert_semantic_node(
            &semantic.nodes,
            idx,
            C_AST_KIND_ASSIGN_EXPR,
            C_AST_PG_CATEGORY_EXPRESSION,
            C_AST_PG_ROLE_ASSIGNMENT,
        );
    }
}
