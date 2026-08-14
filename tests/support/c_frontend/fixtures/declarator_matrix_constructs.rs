//! Token fixtures for the C declarator matrix contracts.
//!
//! The CPU contracts in `vyre-libs/tests` and the backend parity arm in the
//! driver crate build the same token streams, so the fixtures have one owner
//! here rather than a copy per crate.

use crate::c_frontend::token_fixture::{c_fixture, Fixture};
use vyre_libs::parsing::c::lex::tokens::*;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// `int (*p)[4];`
pub(crate) fn fixture_pointer_to_array() -> Fixture {
    c_fixture![
        ("int", TOK_IDENTIFIER),
        ("(", TOK_LPAREN),
        ("*", TOK_STAR),
        ("p", TOK_IDENTIFIER),
        (")", TOK_RPAREN),
        ("[", TOK_LBRACKET),
        ("4", TOK_INTEGER),
        ("]", TOK_RBRACKET),
        (";", TOK_SEMICOLON),
    ]
}

/// `static const int *p, arr[4];`
pub(crate) fn fixture_storage_class_multi_declarator() -> Fixture {
    c_fixture![
        ("static", TOK_IDENTIFIER),
        ("const", TOK_IDENTIFIER),
        ("int", TOK_IDENTIFIER),
        ("*", TOK_STAR),
        ("p", TOK_IDENTIFIER),
        (",", TOK_COMMA),
        ("arr", TOK_IDENTIFIER),
        ("[", TOK_LBRACKET),
        ("4", TOK_INTEGER),
        ("]", TOK_RBRACKET),
        (";", TOK_SEMICOLON),
    ]
}

/// `void f(int arr[static restrict 10]);`
pub(crate) fn fixture_parameter_array_static_restrict() -> Fixture {
    c_fixture![
        ("void", TOK_IDENTIFIER),
        ("f", TOK_IDENTIFIER),
        ("(", TOK_LPAREN),
        ("int", TOK_IDENTIFIER),
        ("arr", TOK_IDENTIFIER),
        ("[", TOK_LBRACKET),
        ("static", TOK_IDENTIFIER),
        ("restrict", TOK_IDENTIFIER),
        ("10", TOK_INTEGER),
        ("]", TOK_RBRACKET),
        (")", TOK_RPAREN),
        (";", TOK_SEMICOLON),
    ]
}

/// ```c
/// typedef int (*fn_t)(int);
/// fn_t f;
/// ```
pub(crate) fn fixture_nested_typedef_complex_declarator() -> Fixture {
    c_fixture![
        ("typedef", TOK_IDENTIFIER),
        ("int", TOK_IDENTIFIER),
        ("(", TOK_LPAREN),
        ("*", TOK_STAR),
        ("fn_t", TOK_IDENTIFIER),
        (")", TOK_RPAREN),
        ("(", TOK_LPAREN),
        ("int", TOK_IDENTIFIER),
        (")", TOK_RPAREN),
        (";", TOK_SEMICOLON),
        ("fn_t", TOK_IDENTIFIER),
        ("f", TOK_IDENTIFIER),
        (";", TOK_SEMICOLON),
    ]
}

/// ```c
/// struct foo { int x; } *p, arr[2];
/// ```
pub(crate) fn fixture_struct_tag_with_mixed_declarators() -> Fixture {
    c_fixture![
        ("struct", TOK_IDENTIFIER),
        ("foo", TOK_IDENTIFIER),
        ("{", TOK_LBRACE),
        ("int", TOK_IDENTIFIER),
        ("x", TOK_IDENTIFIER),
        (";", TOK_SEMICOLON),
        ("}", TOK_RBRACE),
        ("*", TOK_STAR),
        ("p", TOK_IDENTIFIER),
        (",", TOK_COMMA),
        ("arr", TOK_IDENTIFIER),
        ("[", TOK_LBRACKET),
        ("2", TOK_INTEGER),
        ("]", TOK_RBRACKET),
        (";", TOK_SEMICOLON),
    ]
}

/// ```c
/// union cell { char c; int i; } u, *up;
/// ```
pub(crate) fn fixture_union_tag_with_mixed_declarators() -> Fixture {
    c_fixture![
        ("union", TOK_IDENTIFIER),
        ("cell", TOK_IDENTIFIER),
        ("{", TOK_LBRACE),
        ("char", TOK_IDENTIFIER),
        ("c", TOK_IDENTIFIER),
        (";", TOK_SEMICOLON),
        ("int", TOK_IDENTIFIER),
        ("i", TOK_IDENTIFIER),
        (";", TOK_SEMICOLON),
        ("}", TOK_RBRACE),
        ("u", TOK_IDENTIFIER),
        (",", TOK_COMMA),
        ("*", TOK_STAR),
        ("up", TOK_IDENTIFIER),
        (";", TOK_SEMICOLON),
    ]
}

/// ```c
/// enum mode { ON, OFF } ev, *ep;
/// ```
pub(crate) fn fixture_enum_tag_with_mixed_declarators() -> Fixture {
    c_fixture![
        ("enum", TOK_IDENTIFIER),
        ("mode", TOK_IDENTIFIER),
        ("{", TOK_LBRACE),
        ("ON", TOK_IDENTIFIER),
        (",", TOK_COMMA),
        ("OFF", TOK_IDENTIFIER),
        ("}", TOK_RBRACE),
        ("ev", TOK_IDENTIFIER),
        (",", TOK_COMMA),
        ("*", TOK_STAR),
        ("ep", TOK_IDENTIFIER),
        (";", TOK_SEMICOLON),
    ]
}

/// `extern volatile char * const * restrict x, y[8];`
pub(crate) fn fixture_heavy_qualifiers_and_storage_multi_decl() -> Fixture {
    c_fixture![
        ("extern", TOK_IDENTIFIER),
        ("volatile", TOK_IDENTIFIER),
        ("char", TOK_IDENTIFIER),
        ("*", TOK_STAR),
        ("const", TOK_IDENTIFIER),
        ("*", TOK_STAR),
        ("restrict", TOK_IDENTIFIER),
        ("x", TOK_IDENTIFIER),
        (",", TOK_COMMA),
        ("y", TOK_IDENTIFIER),
        ("[", TOK_LBRACKET),
        ("8", TOK_INTEGER),
        ("]", TOK_RBRACKET),
        (";", TOK_SEMICOLON),
    ]
}

/// `(const int (*)(void))p;`
pub(crate) fn fixture_abstract_declarator_with_qualifiers() -> Fixture {
    c_fixture![
        ("(", TOK_LPAREN),
        ("const", TOK_IDENTIFIER),
        ("int", TOK_IDENTIFIER),
        ("(", TOK_LPAREN),
        ("*", TOK_STAR),
        (")", TOK_RPAREN),
        ("(", TOK_LPAREN),
        ("void", TOK_IDENTIFIER),
        (")", TOK_RPAREN),
        (")", TOK_RPAREN),
        ("p", TOK_IDENTIFIER),
        (";", TOK_SEMICOLON),
    ]
}

/// `char * __restrict z;`
pub(crate) fn fixture_gnu_restrict_qualifier() -> Fixture {
    c_fixture![
        ("char", TOK_IDENTIFIER),
        ("*", TOK_STAR),
        ("__restrict", TOK_IDENTIFIER),
        ("z", TOK_IDENTIFIER),
        (";", TOK_SEMICOLON),
    ]
}
