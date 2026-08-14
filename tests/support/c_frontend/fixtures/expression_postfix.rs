//! Token fixtures for postfix and unary operator chains: member access, subscripts, increments, and
//! GNU `__real__`/`__imag__`.
//!
//! The CPU contracts in `vyre-libs/tests` and the backend parity arm in the
//! driver crate build the same token streams, so the fixtures have one owner
//! here rather than a copy per crate.

use crate::c_frontend::spelling::c_kinds;
use vyre_libs::parsing::c::lex::tokens::*;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------
pub(crate) fn chained_member_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = c_kinds("IDENTIFIER DOT IDENTIFIER DOT IDENTIFIER SEMICOLON");
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

pub(crate) fn chained_arrow_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = c_kinds("IDENTIFIER ARROW IDENTIFIER ARROW IDENTIFIER SEMICOLON");
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

pub(crate) fn mixed_postfix_fixture() -> (Vec<u32>, Vec<u32>) {
    // a[0].b->c;
    let tok_types = c_kinds(
         IDENTIFIER LBRACKET INTEGER RBRACKET DOT IDENTIFIER ARROW IDENTIFIER SEMICOLON",
    );
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

pub(crate) fn unary_deref_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = vec![TOK_STAR, TOK_IDENTIFIER, TOK_SEMICOLON];
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

pub(crate) fn unary_addressof_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = vec![TOK_AMP, TOK_IDENTIFIER, TOK_SEMICOLON];
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

pub(crate) fn gnu_real_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = vec![TOK_GNU_REAL, TOK_IDENTIFIER, TOK_SEMICOLON];
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

pub(crate) fn gnu_imag_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = vec![TOK_GNU_IMAG, TOK_IDENTIFIER, TOK_SEMICOLON];
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

pub(crate) fn label_address_fixture() -> (Vec<u32>, Vec<u32>) {
    // &&label;  -- && is a single TOK_AND token in this pipeline
    let tok_types = vec![TOK_AND, TOK_IDENTIFIER, TOK_SEMICOLON];
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

pub(crate) fn postfix_inc_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = vec![TOK_IDENTIFIER, TOK_INC, TOK_SEMICOLON];
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

pub(crate) fn postfix_dec_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = vec![TOK_IDENTIFIER, TOK_DEC, TOK_SEMICOLON];
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}
