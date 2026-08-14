//! Token fixtures for the advanced C declaration contracts.
//!
//! The CPU contracts in `vyre-libs/tests` and the backend parity arm in the
//! driver crate build the same token streams, so the fixtures have one owner
//! here rather than a copy per crate.

use crate::c_frontend::token_fixture::{c_fixture, Fixture};
use vyre_libs::parsing::c::lex::tokens::*;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// ```c
/// struct outer {
///     union {
///         struct { int x; } s;
///         int y;
///     } u;
///     enum { A = 1, B = 2 } e;
/// };
/// ```
pub(crate) fn fixture_nested_struct_union_enum() -> Fixture {
    c_fixture![
        ("struct", TOK_IDENTIFIER),
        ("outer", TOK_IDENTIFIER),
        ("{", TOK_LBRACE),
        ("union", TOK_IDENTIFIER),
        ("{", TOK_LBRACE),
        ("struct", TOK_IDENTIFIER),
        ("{", TOK_LBRACE),
        ("int", TOK_IDENTIFIER),
        ("x", TOK_IDENTIFIER),
        (";", TOK_SEMICOLON),
        ("}", TOK_RBRACE),
        ("s", TOK_IDENTIFIER),
        (";", TOK_SEMICOLON),
        ("int", TOK_IDENTIFIER),
        ("y", TOK_IDENTIFIER),
        (";", TOK_SEMICOLON),
        ("}", TOK_RBRACE),
        ("u", TOK_IDENTIFIER),
        (";", TOK_SEMICOLON),
        ("enum", TOK_IDENTIFIER),
        ("{", TOK_LBRACE),
        ("A", TOK_IDENTIFIER),
        ("=", TOK_ASSIGN),
        ("1", TOK_INTEGER),
        (",", TOK_COMMA),
        ("B", TOK_IDENTIFIER),
        ("=", TOK_ASSIGN),
        ("2", TOK_INTEGER),
        ("}", TOK_RBRACE),
        ("e", TOK_IDENTIFIER),
        (";", TOK_SEMICOLON),
        ("}", TOK_RBRACE),
        (";", TOK_SEMICOLON),
    ]
}

/// ```c
/// struct {
///     union {
///         int i;
///         float f;
///     };
///     int tag;
/// };
/// ```
pub(crate) fn fixture_anonymous_struct_union() -> Fixture {
    c_fixture![
        ("struct", TOK_IDENTIFIER),
        ("{", TOK_LBRACE),
        ("union", TOK_IDENTIFIER),
        ("{", TOK_LBRACE),
        ("int", TOK_IDENTIFIER),
        ("i", TOK_IDENTIFIER),
        (";", TOK_SEMICOLON),
        ("float", TOK_IDENTIFIER),
        ("f", TOK_IDENTIFIER),
        (";", TOK_SEMICOLON),
        ("}", TOK_RBRACE),
        (";", TOK_SEMICOLON),
        ("int", TOK_IDENTIFIER),
        ("tag", TOK_IDENTIFIER),
        (";", TOK_SEMICOLON),
        ("}", TOK_RBRACE),
        (";", TOK_SEMICOLON),
    ]
}

/// ```c
/// typedef struct Node { int v; } Node, *NodePtr;
/// ```
pub(crate) fn fixture_typedef_multiple_declarators() -> Fixture {
    c_fixture![
        ("typedef", TOK_IDENTIFIER),
        ("struct", TOK_IDENTIFIER),
        ("Node", TOK_IDENTIFIER),
        ("{", TOK_LBRACE),
        ("int", TOK_IDENTIFIER),
        ("v", TOK_IDENTIFIER),
        (";", TOK_SEMICOLON),
        ("}", TOK_RBRACE),
        ("Node", TOK_IDENTIFIER),
        (",", TOK_COMMA),
        ("*", TOK_STAR),
        ("NodePtr", TOK_IDENTIFIER),
        (";", TOK_SEMICOLON),
    ]
}

/// ```c
/// const int * const * volatile * restrict p;
/// ```
pub(crate) fn fixture_deeply_nested_pointer() -> Fixture {
    c_fixture![
        ("const", TOK_IDENTIFIER),
        ("int", TOK_IDENTIFIER),
        ("*", TOK_STAR),
        ("const", TOK_IDENTIFIER),
        ("*", TOK_STAR),
        ("volatile", TOK_IDENTIFIER),
        ("*", TOK_STAR),
        ("restrict", TOK_IDENTIFIER),
        ("p", TOK_IDENTIFIER),
        (";", TOK_SEMICOLON),
    ]
}

/// ```c
/// static inline int f(void);
/// extern register int x;
/// _Thread_local _Atomic int y;
/// ```
pub(crate) fn fixture_storage_class_combinations() -> Fixture {
    c_fixture![
        ("static", TOK_IDENTIFIER),
        ("inline", TOK_IDENTIFIER),
        ("int", TOK_IDENTIFIER),
        ("f", TOK_IDENTIFIER),
        ("(", TOK_LPAREN),
        ("void", TOK_IDENTIFIER),
        (")", TOK_RPAREN),
        (";", TOK_SEMICOLON),
        ("extern", TOK_IDENTIFIER),
        ("register", TOK_IDENTIFIER),
        ("int", TOK_IDENTIFIER),
        ("x", TOK_IDENTIFIER),
        (";", TOK_SEMICOLON),
        ("_Thread_local", TOK_IDENTIFIER),
        ("_Atomic", TOK_IDENTIFIER),
        ("int", TOK_IDENTIFIER),
        ("y", TOK_IDENTIFIER),
        (";", TOK_SEMICOLON),
    ]
}

/// ```c
/// struct {
///     unsigned int a : 4;
///     struct {
///         int b : 8;
///         unsigned int : 0;
///     } inner;
/// };
/// ```
pub(crate) fn fixture_bitfield_nested_struct() -> Fixture {
    c_fixture![
        ("struct", TOK_IDENTIFIER),
        ("{", TOK_LBRACE),
        ("unsigned", TOK_IDENTIFIER),
        ("int", TOK_IDENTIFIER),
        ("a", TOK_IDENTIFIER),
        (":", TOK_COLON),
        ("4", TOK_INTEGER),
        (";", TOK_SEMICOLON),
        ("struct", TOK_IDENTIFIER),
        ("{", TOK_LBRACE),
        ("int", TOK_IDENTIFIER),
        ("b", TOK_IDENTIFIER),
        (":", TOK_COLON),
        ("8", TOK_INTEGER),
        (";", TOK_SEMICOLON),
        ("unsigned", TOK_IDENTIFIER),
        ("int", TOK_IDENTIFIER),
        (":", TOK_COLON),
        ("0", TOK_INTEGER),
        (";", TOK_SEMICOLON),
        ("}", TOK_RBRACE),
        ("inner", TOK_IDENTIFIER),
        (";", TOK_SEMICOLON),
        ("}", TOK_RBRACE),
        (";", TOK_SEMICOLON),
    ]
}

/// ```c
/// struct {
///     __attribute__((aligned(8))) int x;
/// };
/// typedef int __attribute__((packed)) packed_int;
/// ```
pub(crate) fn fixture_gnu_attribute_field_and_typedef() -> Fixture {
    c_fixture![
        ("struct", TOK_IDENTIFIER),
        ("{", TOK_LBRACE),
        ("__attribute__", TOK_IDENTIFIER),
        ("(", TOK_LPAREN),
        ("(", TOK_LPAREN),
        ("aligned", TOK_IDENTIFIER),
        ("(", TOK_LPAREN),
        ("8", TOK_INTEGER),
        (")", TOK_RPAREN),
        (")", TOK_RPAREN),
        (")", TOK_RPAREN),
        ("int", TOK_IDENTIFIER),
        ("x", TOK_IDENTIFIER),
        (";", TOK_SEMICOLON),
        ("}", TOK_RBRACE),
        (";", TOK_SEMICOLON),
        ("typedef", TOK_IDENTIFIER),
        ("int", TOK_IDENTIFIER),
        ("__attribute__", TOK_IDENTIFIER),
        ("(", TOK_LPAREN),
        ("(", TOK_LPAREN),
        ("packed", TOK_IDENTIFIER),
        (")", TOK_RPAREN),
        (")", TOK_RPAREN),
        ("packed_int", TOK_IDENTIFIER),
        (";", TOK_SEMICOLON),
    ]
}

/// ```c
/// int (**fp)(void);
/// ```
pub(crate) fn fixture_function_pointer_to_pointer() -> Fixture {
    c_fixture![
        ("int", TOK_IDENTIFIER),
        ("(", TOK_LPAREN),
        ("*", TOK_STAR),
        ("*", TOK_STAR),
        ("fp", TOK_IDENTIFIER),
        (")", TOK_RPAREN),
        ("(", TOK_LPAREN),
        ("void", TOK_IDENTIFIER),
        (")", TOK_RPAREN),
        (";", TOK_SEMICOLON),
    ]
}

/// ```c
/// int (*handlers[4])(int, const char * restrict);
/// ```
pub(crate) fn fixture_array_of_function_pointers_qualified() -> Fixture {
    c_fixture![
        ("int", TOK_IDENTIFIER),
        ("(", TOK_LPAREN),
        ("*", TOK_STAR),
        ("handlers", TOK_IDENTIFIER),
        ("[", TOK_LBRACKET),
        ("4", TOK_INTEGER),
        ("]", TOK_RBRACKET),
        (")", TOK_RPAREN),
        ("(", TOK_LPAREN),
        ("int", TOK_IDENTIFIER),
        (",", TOK_COMMA),
        ("const", TOK_IDENTIFIER),
        ("char", TOK_IDENTIFIER),
        ("*", TOK_STAR),
        ("restrict", TOK_IDENTIFIER),
        (")", TOK_RPAREN),
        (";", TOK_SEMICOLON),
    ]
}
