//! GNU attribute-on-statement token fixtures: fallthrough, unused, aligned labels.
//!
//! The CPU contracts in `vyre-libs/tests` and the backend parity arm in the
//! driver crate build the same token streams, so the fixtures have one owner
//! here rather than a copy per crate.

use crate::c_frontend::token_fixture::{build_fixture, Fixture, FixtureToken};
use vyre_libs::parsing::c::lex::tokens::*;

/// ```c
/// void f(int x) {
///   switch (x) {
///     case 1:
///       __attribute__((fallthrough));
///     case 2:
///       break;
///   }
/// }
/// ```
pub(crate) fn fixture_attribute_fallthrough_statement() -> Fixture {
    build_fixture(&[
        FixtureToken::new("void", TOK_IDENTIFIER),
        FixtureToken::new("f", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("int", TOK_IDENTIFIER),
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
        FixtureToken::new("__attribute__", TOK_GNU_ATTRIBUTE),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("fallthrough", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("case", TOK_CASE),
        FixtureToken::new("2", TOK_INTEGER),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("break", TOK_BREAK),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new("}", TOK_RBRACE),
    ])
}

/// ```c
/// int v = ({ __attribute__((unused)) int tmp = 1; tmp; });
/// ```
pub(crate) fn fixture_attribute_unused_in_statement_expr() -> Fixture {
    build_fixture(&[
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("v", TOK_IDENTIFIER),
        FixtureToken::new("=", TOK_ASSIGN),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("__attribute__", TOK_GNU_ATTRIBUTE),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("unused", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("tmp", TOK_IDENTIFIER),
        FixtureToken::new("=", TOK_ASSIGN),
        FixtureToken::new("1", TOK_INTEGER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("tmp", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

/// ```c
/// void g() {
///   __attribute__((aligned(16))) label:
///     return;
/// }
/// ```
pub(crate) fn fixture_attribute_aligned_on_label() -> Fixture {
    build_fixture(&[
        FixtureToken::new("void", TOK_IDENTIFIER),
        FixtureToken::new("g", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("__attribute__", TOK_GNU_ATTRIBUTE),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("aligned", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("16", TOK_INTEGER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("label", TOK_IDENTIFIER),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("return", TOK_RETURN),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("}", TOK_RBRACE),
    ])
}

/// ```c
/// void h() {
///   __attribute__((section(".data"))) __attribute__((used)) int sym = 0;
/// }
/// ```
pub(crate) fn fixture_multiple_attributes_in_compound() -> Fixture {
    build_fixture(&[
        FixtureToken::new("void", TOK_IDENTIFIER),
        FixtureToken::new("h", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("__attribute__", TOK_GNU_ATTRIBUTE),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("section", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("\".data\"", TOK_STRING),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("__attribute__", TOK_GNU_ATTRIBUTE),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("used", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("sym", TOK_IDENTIFIER),
        FixtureToken::new("=", TOK_ASSIGN),
        FixtureToken::new("0", TOK_INTEGER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("}", TOK_RBRACE),
    ])
}

/// ```c
/// void k() {
///   if (1)
///     __attribute__((cold)) return;
/// }
/// ```
pub(crate) fn fixture_attribute_on_if_arm_statement() -> Fixture {
    build_fixture(&[
        FixtureToken::new("void", TOK_IDENTIFIER),
        FixtureToken::new("k", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("if", TOK_IF),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("1", TOK_INTEGER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("__attribute__", TOK_GNU_ATTRIBUTE),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("cold", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("return", TOK_RETURN),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("}", TOK_RBRACE),
    ])
}
