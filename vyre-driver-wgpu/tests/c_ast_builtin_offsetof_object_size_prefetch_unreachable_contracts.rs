//! Integration tests for GNU/C11 builtin forms not covered by other suites:
//!   - `__builtin_offsetof(type, member)`
//!   - `__builtin_object_size(ptr, type)`
//!   - `__builtin_prefetch(addr, rw, locality)`
//!   - `__builtin_unreachable()`
//!
//! Every test asserts distinct VAST kinds, no collapse into CALL/BINARY,
//! PG lowering preservation, and GPU/CPU parity.

#![cfg(feature = "c-parser")]
#![allow(deprecated)]
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
mod c_ast_gpu_parity_support;

use c_ast_gpu_parity_support::{
    assert_full_pipeline_parity, assert_pg_preserves_fixture_row, build_fixture, classify,
    fixture_builtin_unreachable, row_indices, run_gpu_pg_lower, word_at, Fixture, FixtureToken,
    VAST_STRIDE_U32,
};
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::lower::reference_ast_to_pg_nodes;
use vyre_libs::parsing::c::parse::vast::{
    reference_c11_annotate_typedef_names, reference_c11_build_vast_nodes,
    reference_c11_classify_vast_node_kinds, C_AST_KIND_BUILTIN_OBJECT_SIZE_EXPR,
    C_AST_KIND_BUILTIN_OFFSETOF_EXPR, C_AST_KIND_BUILTIN_PREFETCH_EXPR,
    C_AST_KIND_BUILTIN_UNREACHABLE_STMT,
};
use vyre_primitives::predicate::node_kind;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// void f() { __builtin_offsetof(struct S, field); }
fn fixture_builtin_offsetof() -> Fixture {
    build_fixture(&[
        FixtureToken::new("void", TOK_VOID),
        FixtureToken::new("f", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("__builtin_offsetof", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("struct", TOK_STRUCT),
        FixtureToken::new("S", TOK_IDENTIFIER),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("field", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("}", TOK_RBRACE),
    ])
}

/// void f() { __builtin_object_size(ptr, 0); }
fn fixture_builtin_object_size() -> Fixture {
    build_fixture(&[
        FixtureToken::new("void", TOK_VOID),
        FixtureToken::new("f", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("__builtin_object_size", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("ptr", TOK_IDENTIFIER),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("0", TOK_INTEGER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("}", TOK_RBRACE),
    ])
}

/// void f() { __builtin_prefetch(addr, 0, 3); }
fn fixture_builtin_prefetch() -> Fixture {
    build_fixture(&[
        FixtureToken::new("void", TOK_VOID),
        FixtureToken::new("f", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("__builtin_prefetch", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("addr", TOK_IDENTIFIER),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("0", TOK_INTEGER),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("3", TOK_INTEGER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("}", TOK_RBRACE),
    ])
}

// ---------------------------------------------------------------------------
// Tests – classification
// ---------------------------------------------------------------------------

#[test]
fn builtin_offsetof_classifies_as_distinct_expr() {
    let fix = fixture_builtin_offsetof();
    assert_full_pipeline_parity(&fix, "builtin_offsetof");

    let typed = classify(&fix);
    assert_eq!(
        row_indices(&typed, C_AST_KIND_BUILTIN_OFFSETOF_EXPR),
        vec![5],
        "__builtin_offsetof must classify as BUILTIN_OFFSETOF_EXPR"
    );
    assert_ne!(
        word_at(&typed, 5 * VAST_STRIDE_U32),
        node_kind::CALL,
        "__builtin_offsetof must not collapse into CALL"
    );
}

#[test]
fn builtin_object_size_classifies_as_distinct_expr() {
    let fix = fixture_builtin_object_size();
    assert_full_pipeline_parity(&fix, "builtin_object_size");

    let typed = classify(&fix);
    assert_eq!(
        row_indices(&typed, C_AST_KIND_BUILTIN_OBJECT_SIZE_EXPR),
        vec![5],
        "__builtin_object_size must classify as BUILTIN_OBJECT_SIZE_EXPR"
    );
    assert_ne!(
        word_at(&typed, 5 * VAST_STRIDE_U32),
        node_kind::CALL,
        "__builtin_object_size must not collapse into CALL"
    );
}

#[test]
fn builtin_prefetch_classifies_as_distinct_expr() {
    let fix = fixture_builtin_prefetch();
    assert_full_pipeline_parity(&fix, "builtin_prefetch");

    let typed = classify(&fix);
    assert_eq!(
        row_indices(&typed, C_AST_KIND_BUILTIN_PREFETCH_EXPR),
        vec![5],
        "__builtin_prefetch must classify as BUILTIN_PREFETCH_EXPR"
    );
    assert_ne!(
        word_at(&typed, 5 * VAST_STRIDE_U32),
        node_kind::CALL,
        "__builtin_prefetch must not collapse into CALL"
    );
}

#[test]
fn builtin_unreachable_classifies_as_distinct_stmt() {
    let fix = fixture_builtin_unreachable();
    assert_full_pipeline_parity(&fix, "builtin_unreachable");

    let typed = classify(&fix);
    assert_eq!(
        row_indices(&typed, C_AST_KIND_BUILTIN_UNREACHABLE_STMT),
        vec![5],
        "__builtin_unreachable must classify as BUILTIN_UNREACHABLE_STMT"
    );
    assert_ne!(
        word_at(&typed, 5 * VAST_STRIDE_U32),
        node_kind::CALL,
        "__builtin_unreachable must not collapse into CALL"
    );
}

// ---------------------------------------------------------------------------
// Tests – PG lowering preservation
// ---------------------------------------------------------------------------

#[test]
fn pg_lower_preserves_builtin_offsetof() {
    let fix = fixture_builtin_offsetof();
    let raw = reference_c11_build_vast_nodes(&fix.tok_types, &fix.tok_starts, &fix.tok_lens);
    let annotated = reference_c11_annotate_typedef_names(&raw, fix.source.as_bytes());
    let typed = reference_c11_classify_vast_node_kinds(&annotated);
    let pg = reference_ast_to_pg_nodes(&typed);

    assert_pg_preserves_fixture_row(&typed, &pg, &fix, 5, C_AST_KIND_BUILTIN_OFFSETOF_EXPR);
}

#[test]
fn pg_lower_preserves_builtin_object_size() {
    let fix = fixture_builtin_object_size();
    let raw = reference_c11_build_vast_nodes(&fix.tok_types, &fix.tok_starts, &fix.tok_lens);
    let annotated = reference_c11_annotate_typedef_names(&raw, fix.source.as_bytes());
    let typed = reference_c11_classify_vast_node_kinds(&annotated);
    let pg = reference_ast_to_pg_nodes(&typed);

    assert_pg_preserves_fixture_row(&typed, &pg, &fix, 5, C_AST_KIND_BUILTIN_OBJECT_SIZE_EXPR);
}

#[test]
fn pg_lower_preserves_builtin_prefetch() {
    let fix = fixture_builtin_prefetch();
    let raw = reference_c11_build_vast_nodes(&fix.tok_types, &fix.tok_starts, &fix.tok_lens);
    let annotated = reference_c11_annotate_typedef_names(&raw, fix.source.as_bytes());
    let typed = reference_c11_classify_vast_node_kinds(&annotated);
    let pg = reference_ast_to_pg_nodes(&typed);

    assert_pg_preserves_fixture_row(&typed, &pg, &fix, 5, C_AST_KIND_BUILTIN_PREFETCH_EXPR);
}

#[test]
fn pg_lower_preserves_builtin_unreachable() {
    let fix = fixture_builtin_unreachable();
    let raw = reference_c11_build_vast_nodes(&fix.tok_types, &fix.tok_starts, &fix.tok_lens);
    let annotated = reference_c11_annotate_typedef_names(&raw, fix.source.as_bytes());
    let typed = reference_c11_classify_vast_node_kinds(&annotated);
    let pg = reference_ast_to_pg_nodes(&typed);

    assert_pg_preserves_fixture_row(&typed, &pg, &fix, 5, C_AST_KIND_BUILTIN_UNREACHABLE_STMT);
}

// ---------------------------------------------------------------------------
// Tests – GPU PG lowering parity
// ---------------------------------------------------------------------------

#[test]
fn gpu_pg_lower_matches_cpu_for_remaining_builtin_fixtures() {
    let fixtures: Vec<(&str, Fixture)> = vec![
        ("builtin_offsetof", fixture_builtin_offsetof()),
        ("builtin_object_size", fixture_builtin_object_size()),
        ("builtin_prefetch", fixture_builtin_prefetch()),
        ("builtin_unreachable", fixture_builtin_unreachable()),
    ];

    for (label, fix) in fixtures {
        let raw = reference_c11_build_vast_nodes(&fix.tok_types, &fix.tok_starts, &fix.tok_lens);
        let annotated = reference_c11_annotate_typedef_names(&raw, fix.source.as_bytes());
        let typed = reference_c11_classify_vast_node_kinds(&annotated);
        let expected = reference_ast_to_pg_nodes(&typed);
        let gpu = run_gpu_pg_lower(&typed);
        assert_eq!(
            gpu, expected,
            "GPU PG lowerer must match CPU for fixture `{label}`"
        );
    }
}
