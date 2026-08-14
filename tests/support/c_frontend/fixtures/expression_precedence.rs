//! Token fixtures for the C expression precedence and associativity contracts.
//!
//! The CPU contracts in `vyre-libs/tests` and the backend parity arm in the
//! driver crate build the same token streams, so the fixtures have one owner
//! here rather than a copy per crate.

use crate::c_frontend::expression_pipeline::unit_lens_fixture;
use crate::c_frontend::spelling::c_kinds;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------
pub(crate) fn shift_precedence_fixture() -> (Vec<u32>, Vec<u32>) {
    // a << b + c;
    let tok_types = c_kinds("IDENTIFIER LSHIFT IDENTIFIER PLUS IDENTIFIER SEMICOLON");
    unit_lens_fixture(tok_types)
}

pub(crate) fn relational_precedence_fixture() -> (Vec<u32>, Vec<u32>) {
    // a < b << c;
    let tok_types = c_kinds("IDENTIFIER LT IDENTIFIER LSHIFT IDENTIFIER SEMICOLON");
    unit_lens_fixture(tok_types)
}

pub(crate) fn equality_precedence_fixture() -> (Vec<u32>, Vec<u32>) {
    // a == b < c;
    let tok_types = c_kinds("IDENTIFIER EQ IDENTIFIER LT IDENTIFIER SEMICOLON");
    unit_lens_fixture(tok_types)
}

pub(crate) fn equality_left_assoc_fixture() -> (Vec<u32>, Vec<u32>) {
    // a == b != c;
    let tok_types = c_kinds("IDENTIFIER EQ IDENTIFIER NE IDENTIFIER SEMICOLON");
    unit_lens_fixture(tok_types)
}

pub(crate) fn compound_assignment_fixture() -> (Vec<u32>, Vec<u32>) {
    // a += b -= c;
    let tok_types = c_kinds("IDENTIFIER PLUS_EQ IDENTIFIER MINUS_EQ IDENTIFIER SEMICOLON");
    unit_lens_fixture(tok_types)
}

pub(crate) fn ternary_looser_than_assignment_fixture() -> (Vec<u32>, Vec<u32>) {
    // a = b ? c : d;
    let tok_types =
        c_kinds("IDENTIFIER ASSIGN IDENTIFIER QUESTION IDENTIFIER COLON IDENTIFIER SEMICOLON");
    unit_lens_fixture(tok_types)
}

pub(crate) fn ternary_right_assoc_fixture() -> (Vec<u32>, Vec<u32>) {
    // a ? b : c ? d : e;
    let tok_types = c_kinds(
        "IDENTIFIER QUESTION IDENTIFIER COLON IDENTIFIER QUESTION IDENTIFIER COLON \
         IDENTIFIER SEMICOLON",
    );
    unit_lens_fixture(tok_types)
}

pub(crate) fn comma_boundary_fixture() -> (Vec<u32>, Vec<u32>) {
    // a = b, c = d;
    let tok_types =
        c_kinds("IDENTIFIER ASSIGN IDENTIFIER COMMA IDENTIFIER ASSIGN IDENTIFIER SEMICOLON");
    unit_lens_fixture(tok_types)
}

pub(crate) fn full_precedence_ladder_fixture() -> (Vec<u32>, Vec<u32>) {
    // a || b && c | d ^ e & f == g < h + i << j * k;
    let tok_types = c_kinds(
        "IDENTIFIER OR IDENTIFIER AND IDENTIFIER PIPE IDENTIFIER CARET IDENTIFIER AMP \
         IDENTIFIER EQ IDENTIFIER LT IDENTIFIER PLUS IDENTIFIER LSHIFT IDENTIFIER STAR \
         IDENTIFIER SEMICOLON",
    );
    unit_lens_fixture(tok_types)
}
