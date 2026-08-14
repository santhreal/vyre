//! Token fixtures for Linux-grade C semantic gaps: nested typedef shadowing, GNU attributes on
//! enums and parameters, asm aliases, and mixed initializers.
//!
//! The CPU contracts in `vyre-libs/tests` and the backend parity arm in the
//! driver crate build the same token streams, so the fixtures have one owner
//! here rather than a copy per crate.

use crate::c_frontend::token_fixture::{build_fixture, Fixture, FixtureToken};
use vyre_libs::parsing::c::lex::tokens::*;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// ```c
/// typedef int T;
/// void f(void) {
///     typedef long T;
///     T x;
/// }
/// T y;
/// ```
pub(crate) fn fixture_inner_typedef_shadows_outer() -> Fixture {
    build_fixture(&[
        FixtureToken::new("typedef", TOK_IDENTIFIER),
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("T", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("void", TOK_IDENTIFIER),
        FixtureToken::new("f", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("void", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("typedef", TOK_IDENTIFIER),
        FixtureToken::new("long", TOK_IDENTIFIER),
        FixtureToken::new("T", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("T", TOK_IDENTIFIER),
        FixtureToken::new("x", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new("T", TOK_IDENTIFIER),
        FixtureToken::new("y", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

/// ```c
/// enum __attribute__((packed)) E { A, B };
/// ```
pub(crate) fn fixture_enum_with_attribute() -> Fixture {
    build_fixture(&[
        FixtureToken::new("enum", TOK_IDENTIFIER),
        FixtureToken::new("__attribute__", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("packed", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("E", TOK_IDENTIFIER),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("A", TOK_IDENTIFIER),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("B", TOK_IDENTIFIER),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

/// ```c
/// void f(int __attribute__((unused)) x);
/// ```
pub(crate) fn fixture_parameter_with_attribute() -> Fixture {
    build_fixture(&[
        FixtureToken::new("void", TOK_IDENTIFIER),
        FixtureToken::new("f", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("__attribute__", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("unused", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("x", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

/// ```c
/// void foo(void) __asm__("real_foo");
/// ```
pub(crate) fn fixture_asm_alias() -> Fixture {
    build_fixture(&[
        FixtureToken::new("void", TOK_IDENTIFIER),
        FixtureToken::new("foo", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("void", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("__asm__", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("\"real_foo\"", TOK_STRING),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

/// ```c
/// struct S s = { 1, .b = 2, 3 };
/// ```
pub(crate) fn fixture_mixed_designated_and_plain_init() -> Fixture {
    build_fixture(&[
        FixtureToken::new("struct", TOK_IDENTIFIER),
        FixtureToken::new("S", TOK_IDENTIFIER),
        FixtureToken::new("s", TOK_IDENTIFIER),
        FixtureToken::new("=", TOK_ASSIGN),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("1", TOK_INTEGER),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new(".", TOK_DOT),
        FixtureToken::new("b", TOK_IDENTIFIER),
        FixtureToken::new("=", TOK_ASSIGN),
        FixtureToken::new("2", TOK_INTEGER),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("3", TOK_INTEGER),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

/// ```c
/// int arr[4] = { 1, 2 };
/// ```
pub(crate) fn fixture_incomplete_array_init() -> Fixture {
    build_fixture(&[
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("arr", TOK_IDENTIFIER),
        FixtureToken::new("[", TOK_LBRACKET),
        FixtureToken::new("4", TOK_INTEGER),
        FixtureToken::new("]", TOK_RBRACKET),
        FixtureToken::new("=", TOK_ASSIGN),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("1", TOK_INTEGER),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("2", TOK_INTEGER),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

/// ```c
/// typedef int (*fn_t)(int);
/// fn_t f;
/// ```
pub(crate) fn fixture_function_pointer_typedef_usage() -> Fixture {
    build_fixture(&[
        FixtureToken::new("typedef", TOK_IDENTIFIER),
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("*", TOK_STAR),
        FixtureToken::new("fn_t", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("fn_t", TOK_IDENTIFIER),
        FixtureToken::new("f", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}
