//! Token fixtures for the C expression precedence and associativity contracts.
//!
//! The CPU contracts in `vyre-libs/tests` and the backend parity arm in the
//! driver crate build the same token streams, so the fixtures have one owner
//! here rather than a copy per crate.

use crate::c_frontend::expression_pipeline::unit_lens_fixture;
use vyre_libs::parsing::c::lex::tokens::*;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------
pub(crate) fn shift_precedence_fixture() -> (Vec<u32>, Vec<u32>) {
    // a << b + c;
    let tok_types = vec![
        TOK_IDENTIFIER,
        TOK_LSHIFT,
        TOK_IDENTIFIER,
        TOK_PLUS,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
    ];
    unit_lens_fixture(tok_types)
}

pub(crate) fn relational_precedence_fixture() -> (Vec<u32>, Vec<u32>) {
    // a < b << c;
    let tok_types = vec![
        TOK_IDENTIFIER,
        TOK_LT,
        TOK_IDENTIFIER,
        TOK_LSHIFT,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
    ];
    unit_lens_fixture(tok_types)
}

pub(crate) fn equality_precedence_fixture() -> (Vec<u32>, Vec<u32>) {
    // a == b < c;
    let tok_types = vec![
        TOK_IDENTIFIER,
        TOK_EQ,
        TOK_IDENTIFIER,
        TOK_LT,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
    ];
    unit_lens_fixture(tok_types)
}

pub(crate) fn equality_left_assoc_fixture() -> (Vec<u32>, Vec<u32>) {
    // a == b != c;
    let tok_types = vec![
        TOK_IDENTIFIER,
        TOK_EQ,
        TOK_IDENTIFIER,
        TOK_NE,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
    ];
    unit_lens_fixture(tok_types)
}

pub(crate) fn compound_assignment_fixture() -> (Vec<u32>, Vec<u32>) {
    // a += b -= c;
    let tok_types = vec![
        TOK_IDENTIFIER,
        TOK_PLUS_EQ,
        TOK_IDENTIFIER,
        TOK_MINUS_EQ,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
    ];
    unit_lens_fixture(tok_types)
}

pub(crate) fn ternary_looser_than_assignment_fixture() -> (Vec<u32>, Vec<u32>) {
    // a = b ? c : d;
    let tok_types = vec![
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_IDENTIFIER,
        TOK_QUESTION,
        TOK_IDENTIFIER,
        TOK_COLON,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
    ];
    unit_lens_fixture(tok_types)
}

pub(crate) fn ternary_right_assoc_fixture() -> (Vec<u32>, Vec<u32>) {
    // a ? b : c ? d : e;
    let tok_types = vec![
        TOK_IDENTIFIER,
        TOK_QUESTION,
        TOK_IDENTIFIER,
        TOK_COLON,
        TOK_IDENTIFIER,
        TOK_QUESTION,
        TOK_IDENTIFIER,
        TOK_COLON,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
    ];
    unit_lens_fixture(tok_types)
}

pub(crate) fn comma_boundary_fixture() -> (Vec<u32>, Vec<u32>) {
    // a = b, c = d;
    let tok_types = vec![
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

pub(crate) fn full_precedence_ladder_fixture() -> (Vec<u32>, Vec<u32>) {
    // a || b && c | d ^ e & f == g < h + i << j * k;
    let tok_types = vec![
        TOK_IDENTIFIER, // 0  a
        TOK_OR,         // 1  ||
        TOK_IDENTIFIER, // 2  b
        TOK_AND,        // 3  &&
        TOK_IDENTIFIER, // 4  c
        TOK_PIPE,       // 5  |
        TOK_IDENTIFIER, // 6  d
        TOK_CARET,      // 7  ^
        TOK_IDENTIFIER, // 8  e
        TOK_AMP,        // 9  &
        TOK_IDENTIFIER, // 10 f
        TOK_EQ,         // 11 ==
        TOK_IDENTIFIER, // 12 g
        TOK_LT,         // 13 <
        TOK_IDENTIFIER, // 14 h
        TOK_PLUS,       // 15 +
        TOK_IDENTIFIER, // 16 i
        TOK_LSHIFT,     // 17 <<
        TOK_IDENTIFIER, // 18 j
        TOK_STAR,       // 19 *
        TOK_IDENTIFIER, // 20 k
        TOK_SEMICOLON,  // 21 ;
    ];
    unit_lens_fixture(tok_types)
}
