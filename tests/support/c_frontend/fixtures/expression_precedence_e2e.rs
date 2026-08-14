//! Token fixtures for the end-to-end C expression precedence lowering contracts.
//!
//! The CPU contracts in `vyre-libs/tests` and the backend parity arm in the
//! driver crate build the same token streams, so the fixtures have one owner
//! here rather than a copy per crate.

use crate::c_frontend::expression_pipeline::unit_lens_fixture;
use crate::c_frontend::spelling::c_kinds;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------
pub(crate) fn comma_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = c_kinds(
        "IDENTIFIER ASSIGN IDENTIFIER COMMA IDENTIFIER ASSIGN IDENTIFIER COMMA IDENTIFIER \
         ASSIGN IDENTIFIER SEMICOLON",
    );
    unit_lens_fixture(tok_types)
}

pub(crate) fn assignment_chain_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types =
        c_kinds("IDENTIFIER ASSIGN IDENTIFIER ASSIGN IDENTIFIER ASSIGN IDENTIFIER SEMICOLON");
    unit_lens_fixture(tok_types)
}

pub(crate) fn ternary_nesting_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = c_kinds(
        "IDENTIFIER QUESTION IDENTIFIER QUESTION IDENTIFIER COLON IDENTIFIER COLON \
         IDENTIFIER SEMICOLON",
    );
    unit_lens_fixture(tok_types)
}

pub(crate) fn logical_bitwise_fixture() -> (Vec<u32>, Vec<u32>) {
    // a || b && c | d ^ e & f == g < h + i * j;
    let tok_types = c_kinds(
        "IDENTIFIER OR IDENTIFIER AND IDENTIFIER PIPE IDENTIFIER CARET IDENTIFIER AMP \
         IDENTIFIER EQ IDENTIFIER LT IDENTIFIER PLUS IDENTIFIER STAR IDENTIFIER SEMICOLON",
    );
    unit_lens_fixture(tok_types)
}

pub(crate) fn cast_vs_paren_fixture() -> (Vec<u32>, Vec<u32>) {
    // (int)a; (b + c);
    let tok_types = c_kinds(
        "LPAREN INT RPAREN IDENTIFIER SEMICOLON LPAREN IDENTIFIER PLUS IDENTIFIER RPAREN \
         SEMICOLON",
    );
    unit_lens_fixture(tok_types)
}

pub(crate) fn postfix_fixture() -> (Vec<u32>, Vec<u32>) {
    // a(b); a[b]; a.c; a->d;
    let tok_types = c_kinds(
        "IDENTIFIER LPAREN IDENTIFIER RPAREN SEMICOLON IDENTIFIER LBRACKET IDENTIFIER \
         RBRACKET SEMICOLON IDENTIFIER DOT IDENTIFIER SEMICOLON IDENTIFIER ARROW \
         IDENTIFIER SEMICOLON",
    );
    unit_lens_fixture(tok_types)
}

pub(crate) fn unary_chain_fixture() -> (Vec<u32>, Vec<u32>) {
    // !~-*&++a;
    let tok_types = c_kinds("BANG TILDE MINUS STAR AMP INC IDENTIFIER SEMICOLON");
    unit_lens_fixture(tok_types)
}
