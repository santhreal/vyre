// Integration tests for C expression ambiguity contracts:
//   - member access (`a.b`) and pointer-member access (`a->b`)
//   - cast vs parenthesized expression (`(int)*p` vs `(x)*y`)
//   - nested conditional and comma expressions
//   - compound literals in array contexts
//   - sizeof / _Alignof type-name followed by `*` ambiguity
//
// Every fixture asserts semantic VAST/AST invariants: kind classification,
// parent/child tree links, span preservation, and PG lowering preservation.
// GPU/CPU parity is asserted for the full pipeline.

pub(crate) use crate::c_ast_gpu_parity_support::classify;
use crate::c_ast_gpu_parity_support::{
    assert_full_pipeline_parity, row_indices, void_fn_fixture, Fixture, FixtureToken,
};
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::parse::vast::{C_AST_KIND_CAST_EXPR, C_AST_KIND_MEMBER_ACCESS_EXPR};
use vyre_primitives::predicate::node_kind;
// ---------------------------------------------------------------------------
// Fixtures – member / pointer-member access
// ---------------------------------------------------------------------------

/// void f() { s.field; }
pub(crate) fn fixture_member_access_simple() -> Fixture {
    void_fn_fixture(&[
        FixtureToken::new("s", TOK_IDENTIFIER),
        FixtureToken::new(".", TOK_DOT),
        FixtureToken::new("field", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

/// void f() { p->field; }
pub(crate) fn fixture_ptr_member_access_simple() -> Fixture {
    void_fn_fixture(&[
        FixtureToken::new("p", TOK_IDENTIFIER),
        FixtureToken::new("->", TOK_ARROW),
        FixtureToken::new("field", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

/// void f() { a.b->c.d; }
pub(crate) fn fixture_chained_member_access() -> Fixture {
    void_fn_fixture(&[
        FixtureToken::new("a", TOK_IDENTIFIER),
        FixtureToken::new(".", TOK_DOT),
        FixtureToken::new("b", TOK_IDENTIFIER),
        FixtureToken::new("->", TOK_ARROW),
        FixtureToken::new("c", TOK_IDENTIFIER),
        FixtureToken::new(".", TOK_DOT),
        FixtureToken::new("d", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

// ---------------------------------------------------------------------------
// Fixtures – cast vs parenthesized expression
// ---------------------------------------------------------------------------

/// void f() { (int)*p; }
pub(crate) fn fixture_cast_then_deref() -> Fixture {
    void_fn_fixture(&[
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("int", TOK_INT),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("*", TOK_STAR),
        FixtureToken::new("p", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

/// void f() { (x)*y; }
pub(crate) fn fixture_paren_expr_then_mul() -> Fixture {
    void_fn_fixture(&[
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("x", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("*", TOK_STAR),
        FixtureToken::new("y", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

/// void f() { (int)(1); }
pub(crate) fn fixture_cast_not_compound_literal() -> Fixture {
    void_fn_fixture(&[
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("int", TOK_INT),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("1", TOK_INTEGER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

/// void f() { (x)(1); }
pub(crate) fn fixture_paren_expr_then_call_like() -> Fixture {
    void_fn_fixture(&[
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("x", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("1", TOK_INTEGER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

// ---------------------------------------------------------------------------
// Fixtures – nested conditional / comma
// ---------------------------------------------------------------------------

/// void f() { a ? b ? c : d : e, f; }
pub(crate) fn fixture_nested_conditional_comma() -> Fixture {
    void_fn_fixture(&[
        FixtureToken::new("a", TOK_IDENTIFIER),
        FixtureToken::new("?", TOK_QUESTION),
        FixtureToken::new("b", TOK_IDENTIFIER),
        FixtureToken::new("?", TOK_QUESTION),
        FixtureToken::new("c", TOK_IDENTIFIER),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("d", TOK_IDENTIFIER),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("e", TOK_IDENTIFIER),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("f", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

// ---------------------------------------------------------------------------
// Fixtures – compound literal (array)
// ---------------------------------------------------------------------------

/// void f() { int *p = (int[]){1, 2, 3}; }
pub(crate) fn fixture_array_compound_literal() -> Fixture {
    void_fn_fixture(&[
        FixtureToken::new("int", TOK_INT),
        FixtureToken::new("*", TOK_STAR),
        FixtureToken::new("p", TOK_IDENTIFIER),
        FixtureToken::new("=", TOK_ASSIGN),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("int", TOK_INT),
        FixtureToken::new("[", TOK_LBRACKET),
        FixtureToken::new("]", TOK_RBRACKET),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("1", TOK_INTEGER),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("2", TOK_INTEGER),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("3", TOK_INTEGER),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

// ---------------------------------------------------------------------------
// Fixtures – sizeof / _Alignof ambiguity
// ---------------------------------------------------------------------------

/// void f() { sizeof(int) * p; }
pub(crate) fn fixture_sizeof_typename_then_star() -> Fixture {
    void_fn_fixture(&[
        FixtureToken::new("sizeof", TOK_SIZEOF),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("int", TOK_INT),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("*", TOK_STAR),
        FixtureToken::new("p", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

/// void f() { _Alignof(int) * p; }
pub(crate) fn fixture_alignof_typename_then_star() -> Fixture {
    void_fn_fixture(&[
        FixtureToken::new("_Alignof", TOK_ALIGNOF),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("int", TOK_INT),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("*", TOK_STAR),
        FixtureToken::new("p", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

// ---------------------------------------------------------------------------
// Tests – member / pointer-member access
// ---------------------------------------------------------------------------

#[test]
pub(crate) fn member_access_simple_classifies() {
    let fix = fixture_member_access_simple();
    assert_full_pipeline_parity(&fix, "member_access_simple");

    let typed = classify(&fix);
    assert_eq!(
        row_indices(&typed, C_AST_KIND_MEMBER_ACCESS_EXPR),
        vec![6],
        "dot must classify as MEMBER_ACCESS_EXPR"
    );
}

#[test]
pub(crate) fn ptr_member_access_simple_classifies() {
    let fix = fixture_ptr_member_access_simple();
    assert_full_pipeline_parity(&fix, "ptr_member_access_simple");

    let typed = classify(&fix);
    assert_eq!(
        row_indices(&typed, C_AST_KIND_MEMBER_ACCESS_EXPR),
        vec![6],
        "arrow must classify as MEMBER_ACCESS_EXPR"
    );
}

#[test]
pub(crate) fn chained_member_access_classifies_all_operators() {
    let fix = fixture_chained_member_access();
    assert_full_pipeline_parity(&fix, "chained_member_access");

    let typed = classify(&fix);
    let members = row_indices(&typed, C_AST_KIND_MEMBER_ACCESS_EXPR);
    assert_eq!(
        members,
        vec![6, 8, 10],
        "all three `.` and `->` must classify as MEMBER_ACCESS_EXPR"
    );
}

// ---------------------------------------------------------------------------
// Tests – cast vs parenthesized expression
// ---------------------------------------------------------------------------

#[test]
pub(crate) fn cast_then_deref_is_cast_not_binary() {
    let fix = fixture_cast_then_deref();
    assert_full_pipeline_parity(&fix, "cast_then_deref");

    let typed = classify(&fix);
    assert_eq!(
        row_indices(&typed, C_AST_KIND_CAST_EXPR),
        vec![5],
        "`(int)` must classify as CAST_EXPR"
    );
    // The `*` after a cast is a dereference (unary), not a binary multiply.
    // It should NOT be classified as BINARY.
    assert!(
        !row_indices(&typed, node_kind::BINARY).contains(&8),
        "`*` after cast must not be BINARY"
    );
}
