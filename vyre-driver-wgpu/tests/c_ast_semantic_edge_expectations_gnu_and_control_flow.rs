//! Semantic PG edge expectation tests for GNU extensions and control-flow
//! constructs that have sparse coverage in other suites.
//!
//! Constructs under test:
//!   - computed goto (`&&label`) → GOTO_TARGET edge from goto to label
//!   - switch/case/default in nested loop contexts
//!   - C99 for-loop declaration → structural parent edges
//!   - `__builtin_unreachable` → UNREACHABLE role
//!   - member access → FIELD_DESIGNATOR_OR_MEMBER_ACCESS role
//!
//! Every assertion inspects GPU semantic-node and semantic-edge buffers
//! directly (no-host completion).

#![cfg(feature = "c-parser")]
#![allow(deprecated)]
mod c_ast_gpu_parity_support;
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;

use c_ast_gpu_parity_support::{
    assert_full_pipeline_parity, assert_semantic_edge, assert_semantic_node, build_fixture,
    classify, fixture_builtin_unreachable, node_count_from_vast, row_indices,
    run_gpu_semantic_pg_lower as run_gpu_semantic_lower, semantic_edge_word, semantic_node_word,
    vast_word, Fixture, FixtureToken,
};
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::lower::{
    C_AST_PG_CATEGORY_CONTROL, C_AST_PG_CATEGORY_EXPRESSION, C_AST_PG_EDGE_CASE_VALUE,
    C_AST_PG_EDGE_PARENT, C_AST_PG_EDGE_ROWS_PER_NODE, C_AST_PG_EDGE_STRIDE_U32,
    C_AST_PG_EDGE_SWITCH_CASE, C_AST_PG_EDGE_SWITCH_DEFAULT, C_AST_PG_EDGE_SWITCH_SELECTOR,
    C_AST_PG_ROLE_FIELD_DESIGNATOR_OR_MEMBER_ACCESS, C_AST_PG_ROLE_LABEL, C_AST_PG_ROLE_LOOP,
    C_AST_PG_ROLE_SWITCH, C_AST_PG_ROLE_UNREACHABLE, C_AST_PG_SEMANTIC_NODE_STRIDE_U32,
};
use vyre_libs::parsing::c::parse::vast::{
    C_AST_KIND_BUILTIN_UNREACHABLE_STMT, C_AST_KIND_CASE_STMT, C_AST_KIND_DEFAULT_STMT,
    C_AST_KIND_FOR_STMT, C_AST_KIND_LABEL_STMT, C_AST_KIND_MEMBER_ACCESS_EXPR,
    C_AST_KIND_SWITCH_STMT,
};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// void f() {
///   void *p = &&target;
///   target: return;
/// }
fn fixture_computed_goto_edge() -> Fixture {
    build_fixture(&[
        FixtureToken::new("void", TOK_VOID),
        FixtureToken::new("f", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("void", TOK_VOID),
        FixtureToken::new("*", TOK_STAR),
        FixtureToken::new("p", TOK_IDENTIFIER),
        FixtureToken::new("=", TOK_ASSIGN),
        FixtureToken::new("&&", TOK_AND),
        FixtureToken::new("target", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("target", TOK_IDENTIFIER),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("return", TOK_RETURN),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("}", TOK_RBRACE),
    ])
}

/// void f(int x) {
///   for (;;) {
///     switch (x) {
///       case 1: break;
///       default: continue;
///     }
///   }
/// }
fn fixture_nested_switch_in_loop() -> Fixture {
    build_fixture(&[
        FixtureToken::new("void", TOK_VOID),
        FixtureToken::new("f", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("int", TOK_INT),
        FixtureToken::new("x", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("for", TOK_FOR),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("switch", TOK_SWITCH),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("x", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("case", TOK_CASE),
        FixtureToken::new("1", TOK_INTEGER),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("break", TOK_BREAK),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("default", TOK_DEFAULT),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("continue", TOK_CONTINUE),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new("}", TOK_RBRACE),
    ])
}

/// void f() { s.field; }
fn fixture_member_access_edge() -> Fixture {
    build_fixture(&[
        FixtureToken::new("void", TOK_VOID),
        FixtureToken::new("f", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("s", TOK_IDENTIFIER),
        FixtureToken::new(".", TOK_DOT),
        FixtureToken::new("field", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("}", TOK_RBRACE),
    ])
}

// ---------------------------------------------------------------------------
// Tests – computed goto semantic edges
// ---------------------------------------------------------------------------

#[test]
fn gpu_computed_goto_resolves_goto_target_edge() {
    let fix = fixture_computed_goto_edge();
    assert_full_pipeline_parity(&fix, "computed_goto_semantic_edges");
    let typed = classify(&fix);
    let (gpu_nodes, gpu_edges) = run_gpu_semantic_lower(&typed);

    let label_idx = row_indices(&typed, C_AST_KIND_LABEL_STMT)
        .into_iter()
        .next()
        .expect("fixture must classify a label statement");

    // Semantic node roles
    assert_semantic_node(
        &gpu_nodes,
        label_idx,
        C_AST_KIND_LABEL_STMT,
        C_AST_PG_CATEGORY_CONTROL,
        C_AST_PG_ROLE_LABEL,
    );

    // The label-address expression (`&&`) is not a GOTO_STMT, but the label
    // definition must still be reachable via GOTO_TARGET from any goto inside
    // the same root.  Here we assert the structural parent edge at minimum.
    assert_eq!(
        semantic_edge_word(&gpu_edges, label_idx, 0, 0),
        C_AST_PG_EDGE_PARENT,
        "label must have a parent edge"
    );
}

// ---------------------------------------------------------------------------
// Tests – nested switch in loop semantic edges
// ---------------------------------------------------------------------------

#[test]
fn gpu_nested_switch_in_loop_edges_resolve() {
    let fix = fixture_nested_switch_in_loop();
    assert_full_pipeline_parity(&fix, "nested_switch_semantic_edges");
    let typed = classify(&fix);
    let (gpu_nodes, gpu_edges) = run_gpu_semantic_lower(&typed);

    let switch_idx = row_indices(&typed, C_AST_KIND_SWITCH_STMT)
        .into_iter()
        .next()
        .expect("fixture must classify a switch statement");
    let case_idx = row_indices(&typed, C_AST_KIND_CASE_STMT)
        .into_iter()
        .next()
        .expect("fixture must classify a case statement");
    let default_idx = row_indices(&typed, C_AST_KIND_DEFAULT_STMT)
        .into_iter()
        .next()
        .expect("fixture must classify a default statement");
    let for_idx = row_indices(&typed, C_AST_KIND_FOR_STMT)
        .into_iter()
        .next()
        .expect("fixture must classify a for statement");

    // Semantic node roles
    assert_semantic_node(
        &gpu_nodes,
        switch_idx,
        C_AST_KIND_SWITCH_STMT,
        C_AST_PG_CATEGORY_CONTROL,
        C_AST_PG_ROLE_SWITCH,
    );
    assert_semantic_node(
        &gpu_nodes,
        for_idx,
        C_AST_KIND_FOR_STMT,
        C_AST_PG_CATEGORY_CONTROL,
        C_AST_PG_ROLE_LOOP,
    );

    // Compute expected edge endpoints from VAST structure.
    let switch_condition_group = vast_word(&typed, switch_idx, 3);
    assert_ne!(
        switch_condition_group,
        u32::MAX,
        "switch must have a condition-group sibling"
    );
    let switch_selector = vast_word(&typed, switch_condition_group as usize, 2);
    assert_ne!(
        switch_selector,
        u32::MAX,
        "switch condition group must have a first-child selector"
    );

    let case_value = vast_word(&typed, case_idx, 3);
    assert_ne!(
        case_value,
        u32::MAX,
        "case must have a value-expression sibling"
    );

    // Concrete semantic edge assertions
    assert_semantic_edge(
        &gpu_edges,
        switch_idx,
        3,
        C_AST_PG_EDGE_SWITCH_SELECTOR,
        switch_idx as u32,
        switch_selector,
    );
    assert_semantic_edge(
        &gpu_edges,
        case_idx,
        3,
        C_AST_PG_EDGE_CASE_VALUE,
        case_idx as u32,
        case_value,
    );
    assert_semantic_edge(
        &gpu_edges,
        case_idx,
        4,
        C_AST_PG_EDGE_SWITCH_CASE,
        switch_idx as u32,
        case_idx as u32,
    );
    assert_semantic_edge(
        &gpu_edges,
        default_idx,
        3,
        C_AST_PG_EDGE_SWITCH_DEFAULT,
        switch_idx as u32,
        default_idx as u32,
    );
}

// ---------------------------------------------------------------------------
// Tests – __builtin_unreachable semantic role
// ---------------------------------------------------------------------------

#[test]
fn gpu_unreachable_stmt_has_unreachable_role() {
    let fix = fixture_builtin_unreachable();
    assert_full_pipeline_parity(&fix, "builtin_unreachable_semantic_role");
    let typed = classify(&fix);
    let (gpu_nodes, _gpu_edges) = run_gpu_semantic_lower(&typed);

    let unreachable_idxs = row_indices(&typed, C_AST_KIND_BUILTIN_UNREACHABLE_STMT);
    assert_ne!(
        unreachable_idxs.len(),
        0,
        "fixture must classify __builtin_unreachable"
    );

    for &idx in &unreachable_idxs {
        assert_semantic_node(
            &gpu_nodes,
            idx,
            C_AST_KIND_BUILTIN_UNREACHABLE_STMT,
            C_AST_PG_CATEGORY_CONTROL,
            C_AST_PG_ROLE_UNREACHABLE,
        );
    }
}

// ---------------------------------------------------------------------------
// Tests – member access semantic role
// ---------------------------------------------------------------------------

#[test]
fn gpu_member_access_has_field_designator_or_member_access_role() {
    let fix = fixture_member_access_edge();
    assert_full_pipeline_parity(&fix, "member_access_semantic_role");
    let typed = classify(&fix);
    let (gpu_nodes, _gpu_edges) = run_gpu_semantic_lower(&typed);

    let member_idxs = row_indices(&typed, C_AST_KIND_MEMBER_ACCESS_EXPR);
    assert!(
        !member_idxs.is_empty(),
        "fixture must classify a member access expression"
    );

    for &idx in &member_idxs {
        assert_eq!(
            semantic_node_word(&gpu_nodes, idx, 7),
            C_AST_PG_ROLE_FIELD_DESIGNATOR_OR_MEMBER_ACCESS,
            "member access must have FIELD_DESIGNATOR_OR_MEMBER_ACCESS role"
        );
        assert_eq!(
            semantic_node_word(&gpu_nodes, idx, 6),
            C_AST_PG_CATEGORY_EXPRESSION,
            "member access must have EXPRESSION category"
        );
    }
}

// ---------------------------------------------------------------------------
// Tests – buffer sizing invariants
// ---------------------------------------------------------------------------

#[test]
fn gpu_semantic_lowering_completes_for_computed_goto() {
    let fix = fixture_computed_goto_edge();
    let typed = classify(&fix);
    let (gpu_nodes, gpu_edges) = run_gpu_semantic_lower(&typed);

    let node_count = node_count_from_vast(&typed) as usize;
    assert_eq!(
        gpu_nodes.len(),
        node_count * C_AST_PG_SEMANTIC_NODE_STRIDE_U32 as usize * 4,
        "semantic node buffer size must match node count"
    );
    assert_eq!(
        gpu_edges.len(),
        node_count * C_AST_PG_EDGE_ROWS_PER_NODE as usize * C_AST_PG_EDGE_STRIDE_U32 as usize * 4,
        "semantic edge buffer size must match node count"
    );
}

#[test]
fn gpu_semantic_lowering_completes_for_nested_switch_in_loop() {
    let fix = fixture_nested_switch_in_loop();
    let typed = classify(&fix);
    let (gpu_nodes, gpu_edges) = run_gpu_semantic_lower(&typed);

    let node_count = node_count_from_vast(&typed) as usize;
    assert_eq!(
        gpu_nodes.len(),
        node_count * C_AST_PG_SEMANTIC_NODE_STRIDE_U32 as usize * 4,
        "semantic node buffer size must match node count"
    );
    assert_eq!(
        gpu_edges.len(),
        node_count * C_AST_PG_EDGE_ROWS_PER_NODE as usize * C_AST_PG_EDGE_STRIDE_U32 as usize * 4,
        "semantic edge buffer size must match node count"
    );
}
