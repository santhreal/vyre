//! Semantic PG edge contracts for no-host parser completion.
//!
//! Tests assert concrete edge kinds and semantic roles directly on GPU output
//! without relying on the CPU semantic-lowering oracle (`reference_ast_to_pg_semantic_graph`).
//! VAST build / annotation / classify stages are used only as fixture setup.
//!
//! Constructs under test:
//!   - scope (structural parent-edge nesting)
//!   - type (pointer-declarator roles)
//!   - label / goto (`GOTO_TARGET` edge)
//!   - switch / case / default (`SWITCH_SELECTOR`, `SWITCH_CASE`, `SWITCH_DEFAULT`, `CASE_VALUE` edges)
//!   - function-pointer (`FUNCTION_POINTER_DECL` role)
//!   - typedef (`TYPEDEF_DECL` role)
//!   - tag / enum (`AGGREGATE_DECL` and `ENUMERATOR_DECL` roles)

#![cfg(feature = "c-parser")]
#![allow(deprecated)]
#[allow(dead_code)]
mod c_ast_gpu_parity_support;
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;

use c_ast_gpu_parity_support::{
    assert_parent_edge, assert_semantic_edge, assert_semantic_node, assert_switch_dispatch_edges,
    build_fixture, classify, first_row, node_count_from_vast, row_indices,
    run_gpu_semantic_pg_lower as run_gpu_semantic_lower, semantic_edge_word, semantic_node_word,
    vast_word, void_fn_fixture, Fixture, FixtureToken,
};
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::lower::ast_to_pg_nodes::C_AST_PG_ROLE_AGGREGATE_DECL;
use vyre_libs::parsing::c::lower::{
    C_AST_PG_CATEGORY_CONTROL, C_AST_PG_CATEGORY_DECLARATION, C_AST_PG_EDGE_GOTO_TARGET,
    C_AST_PG_EDGE_PARENT, C_AST_PG_EDGE_ROWS_PER_NODE, C_AST_PG_EDGE_STRIDE_U32,
    C_AST_PG_ROLE_ENUMERATOR_DECL, C_AST_PG_ROLE_FUNCTION_DEFINITION,
    C_AST_PG_ROLE_FUNCTION_POINTER_DECL, C_AST_PG_ROLE_GOTO, C_AST_PG_ROLE_LABEL,
    C_AST_PG_ROLE_POINTER_DECL, C_AST_PG_ROLE_TYPEDEF_DECL, C_AST_PG_SEMANTIC_NODE_STRIDE_U32,
};
use vyre_libs::parsing::c::parse::vast::{
    C_AST_KIND_ENUMERATOR_DECL, C_AST_KIND_ENUM_DECL, C_AST_KIND_FUNCTION_DECLARATOR,
    C_AST_KIND_FUNCTION_DEFINITION, C_AST_KIND_GOTO_STMT, C_AST_KIND_LABEL_STMT,
    C_AST_KIND_POINTER_DECL, C_AST_KIND_STRUCT_DECL, C_AST_KIND_TYPEDEF_DECL,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// ```c
/// typedef int T;
/// struct S { int x; };
/// enum E { A, B };
/// void (*fp)(struct S *);
/// ```
fn fixture_typedef_struct_enum_fnptr() -> Fixture {
    build_fixture(&[
        FixtureToken::new("typedef", TOK_TYPEDEF),
        FixtureToken::new("int", TOK_INT),
        FixtureToken::new("T", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("struct", TOK_STRUCT),
        FixtureToken::new("S", TOK_IDENTIFIER),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("int", TOK_INT),
        FixtureToken::new("x", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("enum", TOK_ENUM),
        FixtureToken::new("E", TOK_IDENTIFIER),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("A", TOK_IDENTIFIER),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("B", TOK_IDENTIFIER),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("void", TOK_VOID),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("*", TOK_STAR),
        FixtureToken::new("fp", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("struct", TOK_STRUCT),
        FixtureToken::new("S", TOK_IDENTIFIER),
        FixtureToken::new("*", TOK_STAR),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

/// ```c
/// void f(int x) {
///   switch (x) {
///     case 1: break;
///     default: goto end;
///   }
///   end: return;
/// }
/// ```
fn fixture_switch_case_default_goto_label() -> Fixture {
    build_fixture(&[
        FixtureToken::new("void", TOK_VOID),
        FixtureToken::new("f", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("int", TOK_INT),
        FixtureToken::new("x", TOK_IDENTIFIER),
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
        FixtureToken::new("goto", TOK_GOTO),
        FixtureToken::new("end", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new("end", TOK_IDENTIFIER),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("return", TOK_RETURN),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("}", TOK_RBRACE),
    ])
}

/// ```c
/// void f() {
///   { int a; }
///   { int b; }
/// }
/// ```
fn fixture_scope_nesting() -> Fixture {
    void_fn_fixture(&[
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("int", TOK_INT),
        FixtureToken::new("a", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("int", TOK_INT),
        FixtureToken::new("b", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("}", TOK_RBRACE),
    ])
}

// ---------------------------------------------------------------------------
// Typedef / tag / enum / function-pointer role contracts
// ---------------------------------------------------------------------------

#[path = "c_ast_semantic_pg_no_host_edge_contracts/node_roles_and_edges.rs"]
mod node_roles_and_edges;
#[path = "c_ast_semantic_pg_no_host_edge_contracts/scope_nesting.rs"]
mod scope_nesting;
