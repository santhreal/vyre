//! Token fixtures for GNU builtin expressions and C11 `_Generic` selections.
//!
//! The CPU contracts in `vyre-libs/tests` and the backend parity arm in the
//! driver crate build the same token streams, so the fixtures have one owner
//! here rather than a copy per crate.

use vyre_libs::parsing::c::lex::tokens::*;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------
pub(crate) fn builtin_constant_p_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_BUILTIN_CONSTANT_P,
        TOK_LPAREN,
        TOK_IDENTIFIER,
        TOK_RPAREN,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

pub(crate) fn builtin_choose_expr_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_BUILTIN_CHOOSE_EXPR,
        TOK_LPAREN,
        TOK_INTEGER,
        TOK_COMMA,
        TOK_INTEGER,
        TOK_COMMA,
        TOK_INTEGER,
        TOK_RPAREN,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

pub(crate) fn builtin_types_compatible_p_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_BUILTIN_TYPES_COMPATIBLE_P,
        TOK_LPAREN,
        TOK_INT,
        TOK_COMMA,
        TOK_LONG,
        TOK_RPAREN,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

pub(crate) fn generic_selection_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_GENERIC,
        TOK_LPAREN,
        TOK_IDENTIFIER,
        TOK_COMMA,
        TOK_INT,
        TOK_COLON,
        TOK_INTEGER,
        TOK_COMMA,
        TOK_DEFAULT,
        TOK_COLON,
        TOK_INTEGER,
        TOK_RPAREN,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

pub(crate) fn nested_builtin_fixture() -> (Vec<u32>, Vec<u32>) {
    // __builtin_choose_expr(1, __builtin_constant_p(2), 0);
    let tok_types = vec![
        TOK_BUILTIN_CHOOSE_EXPR,
        TOK_LPAREN,
        TOK_INTEGER,
        TOK_COMMA,
        TOK_BUILTIN_CONSTANT_P,
        TOK_LPAREN,
        TOK_INTEGER,
        TOK_RPAREN,
        TOK_COMMA,
        TOK_INTEGER,
        TOK_RPAREN,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}
