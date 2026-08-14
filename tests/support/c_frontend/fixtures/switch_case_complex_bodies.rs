//! Switch/case token fixtures with statement expressions, compound literals, and Duff's device.
//!
//! The CPU contracts in `vyre-libs/tests` and the backend parity arm in the
//! driver crate build the same token streams, so the fixtures have one owner
//! here rather than a copy per crate.

use crate::c_frontend::spelling::c_tokens;
use crate::c_frontend::token_fixture::Fixture;

/// ```c
/// void f(int x) {
///   switch (x) {
///     case 1:
///       ({ int t = 1; t; });
///       break;
///   }
/// }
/// ```
pub(crate) fn fixture_switch_case_with_statement_expr() -> Fixture {
    c_tokens("void f ( int x ) { switch ( x ) { case 1 : ( { int t = 1 ; t ; } ) ; break ; } }")
}

/// ```c
/// void g(int x) {
///   switch (x) {
///     case 2:
///       (struct S){ .a = 1 };
///       break;
///   }
/// }
/// ```
pub(crate) fn fixture_switch_case_with_compound_literal() -> Fixture {
    c_tokens("void g ( int x ) { switch ( x ) { case 2 : ( struct S ) { . a = 1 } ; break ; } }")
}

/// ```c
/// void h(int x) {
///   switch (x) {
///     case 3:
///       int arr[2] = { [0] = 1, [1] = 2 };
///       break;
///   }
/// }
/// ```
pub(crate) fn fixture_switch_case_with_designated_init() -> Fixture {
    c_tokens(
        "void h ( int x ) { switch ( x ) { case 3 : int arr [ 2 ] = { [ 0 ] = 1 , [ 1 ] = \
         2 } ; break ; } }",
    )
}

/// ```c
/// void k() {
///   int n = 4;
///   switch (n) {
///     case 0:
///       do { *p++ = *q++;
///     case 1:
///       *p++ = *q++;
///       } while (--n > 0);
///   }
/// }
/// ```
/// (Simplified Duff's device pattern  -  interleaved switch/loop/labels.)
pub(crate) fn fixture_duffs_device_interleaved() -> Fixture {
    c_tokens(
        "void k ( ) { int n = 4 ; switch ( n ) { case 0 : do { p ++ ; } while ( -- n > 0 \
         ) ; case 1 : q ++ ; } }",
    )
}

/// ```c
/// int m() {
///   return ({ switch (1) { case 1: return 1; default: return 0; } });
/// }
/// ```
pub(crate) fn fixture_nested_switch_inside_statement_expr() -> Fixture {
    c_tokens(
        "int m ( ) { return ( { switch ( 1 ) { case 1 : return 1 ; default : return 0 ; } \
         } ) ; }",
    )
}

/// ```c
/// void n(int x) {
///   switch (x) {
///     default:
///     shared:
///       break;
///   }
/// }
/// ```
pub(crate) fn fixture_default_with_user_label() -> Fixture {
    c_tokens("void n ( int x ) { switch ( x ) { default : shared : break ; } }")
}
