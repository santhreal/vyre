//! Compound-literal and designated-initializer token fixtures, including nested forms.
//!
//! The CPU contracts in `vyre-libs/tests` and the backend parity arm in the
//! driver crate build the same token streams, so the fixtures have one owner
//! here rather than a copy per crate.

use crate::c_frontend::token_fixture::{build_fixture, Fixture, FixtureToken};
use vyre_libs::parsing::c::lex::tokens::*;

/// ```c
/// struct S { int a; struct { int b; int c; } inner; };
/// struct S s = (struct S){ .a = 1, .inner = { .b = 2, .c = 3 } };
/// ```
pub(crate) fn fixture_compound_literal_nested_designated() -> Fixture {
    build_fixture(&[
        FixtureToken::new("struct", TOK_STRUCT),
        FixtureToken::new("S", TOK_IDENTIFIER),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("a", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("struct", TOK_STRUCT),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("b", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("c", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new("inner", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("struct", TOK_STRUCT),
        FixtureToken::new("S", TOK_IDENTIFIER),
        FixtureToken::new("s", TOK_IDENTIFIER),
        FixtureToken::new("=", TOK_ASSIGN),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("struct", TOK_STRUCT),
        FixtureToken::new("S", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new(".", TOK_DOT),
        FixtureToken::new("a", TOK_IDENTIFIER),
        FixtureToken::new("=", TOK_ASSIGN),
        FixtureToken::new("1", TOK_INTEGER),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new(".", TOK_DOT),
        FixtureToken::new("inner", TOK_IDENTIFIER),
        FixtureToken::new("=", TOK_ASSIGN),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new(".", TOK_DOT),
        FixtureToken::new("b", TOK_IDENTIFIER),
        FixtureToken::new("=", TOK_ASSIGN),
        FixtureToken::new("2", TOK_INTEGER),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new(".", TOK_DOT),
        FixtureToken::new("c", TOK_IDENTIFIER),
        FixtureToken::new("=", TOK_ASSIGN),
        FixtureToken::new("3", TOK_INTEGER),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

/// ```c
/// int x = ({ (struct S){ .v = 1 }; });
/// ```
pub(crate) fn fixture_compound_literal_inside_statement_expr() -> Fixture {
    build_fixture(&[
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("x", TOK_IDENTIFIER),
        FixtureToken::new("=", TOK_ASSIGN),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("struct", TOK_STRUCT),
        FixtureToken::new("S", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new(".", TOK_DOT),
        FixtureToken::new("v", TOK_IDENTIFIER),
        FixtureToken::new("=", TOK_ASSIGN),
        FixtureToken::new("1", TOK_INTEGER),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

/// ```c
/// struct S s = { .a = __builtin_choose_expr(1, 10, 20) };
/// ```
pub(crate) fn fixture_designated_init_with_builtin_choose_expr() -> Fixture {
    build_fixture(&[
        FixtureToken::new("struct", TOK_STRUCT),
        FixtureToken::new("S", TOK_IDENTIFIER),
        FixtureToken::new("s", TOK_IDENTIFIER),
        FixtureToken::new("=", TOK_ASSIGN),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new(".", TOK_DOT),
        FixtureToken::new("a", TOK_IDENTIFIER),
        FixtureToken::new("=", TOK_ASSIGN),
        FixtureToken::new("__builtin_choose_expr", TOK_BUILTIN_CHOOSE_EXPR),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("1", TOK_INTEGER),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("10", TOK_INTEGER),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("20", TOK_INTEGER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

/// ```c
/// struct S arr[2] = { (struct S){ .x = 1 }, (struct S){ .x = 2 } };
/// ```
pub(crate) fn fixture_array_of_compound_literals() -> Fixture {
    build_fixture(&[
        FixtureToken::new("struct", TOK_STRUCT),
        FixtureToken::new("S", TOK_IDENTIFIER),
        FixtureToken::new("arr", TOK_IDENTIFIER),
        FixtureToken::new("[", TOK_LBRACKET),
        FixtureToken::new("2", TOK_INTEGER),
        FixtureToken::new("]", TOK_RBRACKET),
        FixtureToken::new("=", TOK_ASSIGN),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("struct", TOK_STRUCT),
        FixtureToken::new("S", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new(".", TOK_DOT),
        FixtureToken::new("x", TOK_IDENTIFIER),
        FixtureToken::new("=", TOK_ASSIGN),
        FixtureToken::new("1", TOK_INTEGER),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("struct", TOK_STRUCT),
        FixtureToken::new("S", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new(".", TOK_DOT),
        FixtureToken::new("x", TOK_IDENTIFIER),
        FixtureToken::new("=", TOK_ASSIGN),
        FixtureToken::new("2", TOK_INTEGER),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

/// ```c
/// struct S *p = cond ? (struct S){ .x = 1 } : (struct S){ .x = 2 };
/// ```
pub(crate) fn fixture_compound_literal_in_ternary() -> Fixture {
    build_fixture(&[
        FixtureToken::new("struct", TOK_STRUCT),
        FixtureToken::new("S", TOK_IDENTIFIER),
        FixtureToken::new("*", TOK_STAR),
        FixtureToken::new("p", TOK_IDENTIFIER),
        FixtureToken::new("=", TOK_ASSIGN),
        FixtureToken::new("cond", TOK_IDENTIFIER),
        FixtureToken::new("?", TOK_QUESTION),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("struct", TOK_STRUCT),
        FixtureToken::new("S", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new(".", TOK_DOT),
        FixtureToken::new("x", TOK_IDENTIFIER),
        FixtureToken::new("=", TOK_ASSIGN),
        FixtureToken::new("1", TOK_INTEGER),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("struct", TOK_STRUCT),
        FixtureToken::new("S", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new(".", TOK_DOT),
        FixtureToken::new("x", TOK_IDENTIFIER),
        FixtureToken::new("=", TOK_ASSIGN),
        FixtureToken::new("2", TOK_INTEGER),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}
