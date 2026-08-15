//! Token fixtures for Linux-grade C semantic gaps: nested typedef shadowing, GNU attributes on
//! enums and parameters, asm aliases, and mixed initializers.
//!
//! The CPU contracts in `vyre-libs/tests` and the backend parity arm in the
//! driver crate build the same token streams, so the fixtures have one owner
//! here rather than a copy per crate.
//!
//! [`CASES`] is that one owner's case list. Both arms iterate it, so a fixture
//! added below and named there is proven on the CPU oracle and on every
//! backend arm at once, and `c_ast_parity_case_matrix_gate` fails when a
//! builder here is missing from it.
//!
//! There is deliberately no function-pointer-typedef fixture here.
//! `typedef int (*fn_t)(int); fn_t f;` is
//! `declarator_matrix_constructs::fixture_nested_typedef_complex_declarator`,
//! which already asserts the same four rows on both arms; a second builder for
//! the identical token stream ran one construct twice under two names.

use crate::c_frontend::parity_matrix::ParityCase;
use crate::c_frontend::spelling::c_tokens;
use crate::c_frontend::token_fixture::Fixture;

/// Every semantic-gap construct both arms evaluate.
pub(crate) const CASES: &[ParityCase] = &[
    ParityCase::new(
        "inner_typedef_shadows_outer",
        fixture_inner_typedef_shadows_outer,
    ),
    ParityCase::new("enum_with_attribute", fixture_enum_with_attribute),
    ParityCase::new("parameter_with_attribute", fixture_parameter_with_attribute),
    ParityCase::new("asm_alias", fixture_asm_alias),
    ParityCase::new(
        "mixed_designated_and_plain_init",
        fixture_mixed_designated_and_plain_init,
    ),
    ParityCase::new("incomplete_array_init", fixture_incomplete_array_init),
];

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// ```c
/// typedef int T;
/// void f(void) {
///     typedef long T;
///     T x;
/// }
/// T y;
/// ```
pub(crate) fn fixture_inner_typedef_shadows_outer() -> Fixture {
    c_tokens("typedef int T ; void f ( void ) { typedef long T ; T x ; } T y ;")
}

/// ```c
/// enum __attribute__((packed)) E { A, B };
/// ```
pub(crate) fn fixture_enum_with_attribute() -> Fixture {
    c_tokens("enum __attribute__ ( ( packed ) ) E { A , B } ;")
}

/// ```c
/// void f(int __attribute__((unused)) x);
/// ```
pub(crate) fn fixture_parameter_with_attribute() -> Fixture {
    c_tokens("void f ( int __attribute__ ( ( unused ) ) x ) ;")
}

/// ```c
/// void foo(void) __asm__("real_foo");
/// ```
pub(crate) fn fixture_asm_alias() -> Fixture {
    c_tokens("void foo ( void ) __asm__ ( \"real_foo\" ) ;")
}

/// ```c
/// struct S s = { 1, .b = 2, 3 };
/// ```
pub(crate) fn fixture_mixed_designated_and_plain_init() -> Fixture {
    c_tokens("struct S s = { 1 , . b = 2 , 3 } ;")
}

/// ```c
/// int arr[4] = { 1, 2 };
/// ```
pub(crate) fn fixture_incomplete_array_init() -> Fixture {
    c_tokens("int arr [ 4 ] = { 1 , 2 } ;")
}
