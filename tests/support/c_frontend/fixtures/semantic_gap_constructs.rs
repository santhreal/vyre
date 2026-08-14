//! Token fixtures for Linux-grade C semantic gaps: nested typedef shadowing, GNU attributes on
//! enums and parameters, asm aliases, and mixed initializers.
//!
//! The CPU contracts in `vyre-libs/tests` and the backend parity arm in the
//! driver crate build the same token streams, so the fixtures have one owner
//! here rather than a copy per crate.

use crate::c_frontend::spelling::c_tokens;
use crate::c_frontend::token_fixture::Fixture;

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

/// ```c
/// typedef int (*fn_t)(int);
/// fn_t f;
/// ```
pub(crate) fn fixture_function_pointer_typedef_usage() -> Fixture {
    c_tokens("typedef int ( * fn_t ) ( int ) ; fn_t f ;")
}
