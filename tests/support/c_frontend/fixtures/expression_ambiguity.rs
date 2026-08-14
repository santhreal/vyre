//! Token fixtures for C expression-operator ambiguity: unary vs binary `*`/`&`/`+`/`-`, casts vs
//! parenthesised expressions, and `sizeof`/`typeof` operands.
//!
//! The CPU contracts in `vyre-libs/tests` and the backend parity arm in the
//! driver crate build the same token streams, so the fixtures have one owner
//! here rather than a copy per crate.

use vyre_libs::parsing::c::lex::tokens::*;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------
pub(crate) fn star_binary_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = vec![TOK_IDENTIFIER, TOK_STAR, TOK_IDENTIFIER, TOK_SEMICOLON];
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

pub(crate) fn star_unary_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = vec![TOK_STAR, TOK_IDENTIFIER, TOK_SEMICOLON];
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

pub(crate) fn amp_binary_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = vec![TOK_IDENTIFIER, TOK_AMP, TOK_IDENTIFIER, TOK_SEMICOLON];
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

pub(crate) fn amp_unary_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = vec![TOK_AMP, TOK_IDENTIFIER, TOK_SEMICOLON];
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

pub(crate) fn plus_binary_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = vec![TOK_IDENTIFIER, TOK_PLUS, TOK_IDENTIFIER, TOK_SEMICOLON];
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

pub(crate) fn plus_unary_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = vec![TOK_PLUS, TOK_IDENTIFIER, TOK_SEMICOLON];
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

pub(crate) fn minus_binary_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = vec![TOK_IDENTIFIER, TOK_MINUS, TOK_IDENTIFIER, TOK_SEMICOLON];
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

pub(crate) fn minus_unary_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = vec![TOK_MINUS, TOK_IDENTIFIER, TOK_SEMICOLON];
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

pub(crate) fn cast_simple_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_LPAREN,
        TOK_INT,
        TOK_RPAREN,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

pub(crate) fn paren_expr_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_LPAREN,
        TOK_IDENTIFIER,
        TOK_PLUS,
        TOK_IDENTIFIER,
        TOK_RPAREN,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

pub(crate) fn cast_complex_fixture() -> (Vec<u32>, Vec<u32>) {
    // (const int *)p;
    let tok_types = vec![
        TOK_LPAREN,
        TOK_CONST,
        TOK_INT,
        TOK_STAR,
        TOK_RPAREN,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

pub(crate) fn paren_nested_fixture() -> (Vec<u32>, Vec<u32>) {
    // ((a));
    let tok_types = vec![
        TOK_LPAREN,
        TOK_LPAREN,
        TOK_IDENTIFIER,
        TOK_RPAREN,
        TOK_RPAREN,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

pub(crate) fn sizeof_typename_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = vec![TOK_SIZEOF, TOK_LPAREN, TOK_INT, TOK_RPAREN, TOK_SEMICOLON];
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

pub(crate) fn sizeof_expr_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = vec![TOK_SIZEOF, TOK_IDENTIFIER, TOK_SEMICOLON];
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

pub(crate) fn typeof_typename_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_GNU_TYPEOF,
        TOK_LPAREN,
        TOK_INT,
        TOK_RPAREN,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

pub(crate) fn typeof_expr_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = vec![TOK_GNU_TYPEOF, TOK_IDENTIFIER, TOK_SEMICOLON];
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}
