//! Compound-literal and designated-initializer token fixtures, including nested forms.
//!
//! The CPU contracts in `vyre-libs/tests` and the backend parity arm in the
//! driver crate build the same token streams, so the fixtures have one owner
//! here rather than a copy per crate.

use crate::c_frontend::spelling::c_tokens;
use crate::c_frontend::token_fixture::Fixture;

/// ```c
/// struct S { int a; struct { int b; int c; } inner; };
/// struct S s = (struct S){ .a = 1, .inner = { .b = 2, .c = 3 } };
/// ```
pub(crate) fn fixture_compound_literal_nested_designated() -> Fixture {
    c_tokens(
        "struct S { int a ; struct { int b ; int c ; } inner ; } ; struct S s = ( struct \
         S ) { . a = 1 , . inner = { . b = 2 , . c = 3 } } ;",
    )
}

/// ```c
/// int x = ({ (struct S){ .v = 1 }; });
/// ```
pub(crate) fn fixture_compound_literal_inside_statement_expr() -> Fixture {
    c_tokens("int x = ( { ( struct S ) { . v = 1 } ; } ) ;")
}

/// ```c
/// struct S s = { .a = __builtin_choose_expr(1, 10, 20) };
/// ```
pub(crate) fn fixture_designated_init_with_builtin_choose_expr() -> Fixture {
    c_tokens("struct S s = { . a = __builtin_choose_expr ( 1 , 10 , 20 ) } ;")
}

/// ```c
/// struct S arr[2] = { (struct S){ .x = 1 }, (struct S){ .x = 2 } };
/// ```
pub(crate) fn fixture_array_of_compound_literals() -> Fixture {
    c_tokens("struct S arr [ 2 ] = { ( struct S ) { . x = 1 } , ( struct S ) { . x = 2 } } ;")
}

/// ```c
/// struct S *p = cond ? (struct S){ .x = 1 } : (struct S){ .x = 2 };
/// ```
pub(crate) fn fixture_compound_literal_in_ternary() -> Fixture {
    c_tokens("struct S * p = cond ? ( struct S ) { . x = 1 } : ( struct S ) { . x = 2 } ;")
}
