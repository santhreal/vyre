//! `__builtin_expect` and `__builtin_choose_expr` control-flow token fixtures.
//!
//! The CPU contracts in `vyre-libs/tests` and the backend parity arm in the
//! driver crate build the same token streams, so the fixtures have one owner
//! here rather than a copy per crate.

use crate::c_frontend::spelling::c_tokens;
use crate::c_frontend::token_fixture::Fixture;

/// ```c
/// if (__builtin_expect(x, 1)) { }
/// ```
pub(crate) fn fixture_builtin_expect_if_condition() -> Fixture {
    c_tokens("if ( __builtin_expect ( x , 1 ) ) { }")
}

/// ```c
/// switch (__builtin_expect(x, 0)) { case 1: break; }
/// ```
pub(crate) fn fixture_builtin_expect_switch_selector() -> Fixture {
    c_tokens("switch ( __builtin_expect ( x , 0 ) ) { case 1 : break ; }")
}

/// ```c
/// int y = ({ __builtin_choose_expr(1, 2, 3); });
/// ```
pub(crate) fn fixture_builtin_choose_expr_in_statement_expr() -> Fixture {
    c_tokens("int y = ( { __builtin_choose_expr ( 1 , 2 , 3 ) ; } ) ;")
}

/// ```c
/// struct S { int a; };
/// struct S s = { .a = __builtin_choose_expr(1, 10, 20) };
/// ```
pub(crate) fn fixture_builtin_choose_expr_in_designated_init() -> Fixture {
    c_tokens(
        "struct S { int a ; } ; struct S s = { . a = __builtin_choose_expr ( 1 , 10 , 20 \
         ) } ;",
    )
}

/// ```c
/// int z = __builtin_expect(__builtin_choose_expr(1, 2, 3), 1);
/// ```
pub(crate) fn fixture_nested_builtin_expect_choose_expr() -> Fixture {
    c_tokens("int z = __builtin_expect ( __builtin_choose_expr ( 1 , 2 , 3 ) , 1 ) ;")
}

/// ```c
/// int w = __builtin_expect(!!(x), 1) ? 1 : 0;
/// ```
pub(crate) fn fixture_builtin_expect_in_ternary() -> Fixture {
    c_tokens("int w = __builtin_expect ( ! ! ( x ) , 1 ) ? 1 : 0 ;")
}
