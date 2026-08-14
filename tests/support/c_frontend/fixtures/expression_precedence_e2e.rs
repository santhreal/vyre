//! Token fixtures for the end-to-end C expression precedence lowering contracts.
//!
//! The CPU contracts in `vyre-libs/tests` and the backend parity arm in the
//! driver crate build the same token streams, so the fixtures have one owner
//! here rather than a copy per crate.

use crate::c_frontend::expression_pipeline::unit_lens_fixture;
use vyre_libs::parsing::c::lex::tokens::*;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------
pub(crate) fn comma_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_IDENTIFIER,
        TOK_COMMA,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_IDENTIFIER,
        TOK_COMMA,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
    ];
    unit_lens_fixture(tok_types)
}

pub(crate) fn assignment_chain_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
    ];
    unit_lens_fixture(tok_types)
}

pub(crate) fn ternary_nesting_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_IDENTIFIER,
        TOK_QUESTION,
        TOK_IDENTIFIER,
        TOK_QUESTION,
        TOK_IDENTIFIER,
        TOK_COLON,
        TOK_IDENTIFIER,
        TOK_COLON,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
    ];
    unit_lens_fixture(tok_types)
}

pub(crate) fn logical_bitwise_fixture() -> (Vec<u32>, Vec<u32>) {
    // a || b && c | d ^ e & f == g < h + i * j;
    let tok_types = vec![
        TOK_IDENTIFIER,
        TOK_OR,
        TOK_IDENTIFIER,
        TOK_AND,
        TOK_IDENTIFIER,
        TOK_PIPE,
        TOK_IDENTIFIER,
        TOK_CARET,
        TOK_IDENTIFIER,
        TOK_AMP,
        TOK_IDENTIFIER,
        TOK_EQ,
        TOK_IDENTIFIER,
        TOK_LT,
        TOK_IDENTIFIER,
        TOK_PLUS,
        TOK_IDENTIFIER,
        TOK_STAR,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
    ];
    unit_lens_fixture(tok_types)
}

pub(crate) fn cast_vs_paren_fixture() -> (Vec<u32>, Vec<u32>) {
    // (int)a; (b + c);
    let tok_types = vec![
        TOK_LPAREN,
        TOK_INT,
        TOK_RPAREN,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
        TOK_LPAREN,
        TOK_IDENTIFIER,
        TOK_PLUS,
        TOK_IDENTIFIER,
        TOK_RPAREN,
        TOK_SEMICOLON,
    ];
    unit_lens_fixture(tok_types)
}

pub(crate) fn postfix_fixture() -> (Vec<u32>, Vec<u32>) {
    // a(b); a[b]; a.c; a->d;
    let tok_types = vec![
        TOK_IDENTIFIER,
        TOK_LPAREN,
        TOK_IDENTIFIER,
        TOK_RPAREN,
        TOK_SEMICOLON,
        TOK_IDENTIFIER,
        TOK_LBRACKET,
        TOK_IDENTIFIER,
        TOK_RBRACKET,
        TOK_SEMICOLON,
        TOK_IDENTIFIER,
        TOK_DOT,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
        TOK_IDENTIFIER,
        TOK_ARROW,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
    ];
    unit_lens_fixture(tok_types)
}

pub(crate) fn unary_chain_fixture() -> (Vec<u32>, Vec<u32>) {
    // !~-*&++a;
    let tok_types = vec![
        TOK_BANG,
        TOK_TILDE,
        TOK_MINUS,
        TOK_STAR,
        TOK_AMP,
        TOK_INC,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
    ];
    unit_lens_fixture(tok_types)
}
