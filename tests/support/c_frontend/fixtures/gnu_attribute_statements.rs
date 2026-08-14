//! GNU attribute-on-statement token fixtures: fallthrough, unused, aligned labels.
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
///       __attribute__((fallthrough));
///     case 2:
///       break;
///   }
/// }
/// ```
pub(crate) fn fixture_attribute_fallthrough_statement() -> Fixture {
    c_tokens(
        "void f ( int x ) { switch ( x ) { case 1 : __attribute__ ( ( fallthrough ) ) ; \
         case 2 : break ; } }",
    )
}

/// ```c
/// int v = ({ __attribute__((unused)) int tmp = 1; tmp; });
/// ```
pub(crate) fn fixture_attribute_unused_in_statement_expr() -> Fixture {
    c_tokens("int v = ( { __attribute__ ( ( unused ) ) int tmp = 1 ; tmp ; } ) ;")
}

/// ```c
/// void g() {
///   __attribute__((aligned(16))) label:
///     return;
/// }
/// ```
pub(crate) fn fixture_attribute_aligned_on_label() -> Fixture {
    c_tokens("void g ( ) { __attribute__ ( ( aligned ( 16 ) ) ) label : return ; }")
}

/// ```c
/// void h() {
///   __attribute__((section(".data"))) __attribute__((used)) int sym = 0;
/// }
/// ```
pub(crate) fn fixture_multiple_attributes_in_compound() -> Fixture {
    c_tokens(
        "void h ( ) { __attribute__ ( ( section ( ".data" ) ) ) __attribute__ ( ( used ) \
         ) int sym = 0 ; }",
    )
}

/// ```c
/// void k() {
///   if (1)
///     __attribute__((cold)) return;
/// }
/// ```
pub(crate) fn fixture_attribute_on_if_arm_statement() -> Fixture {
    c_tokens("void k ( ) { if ( 1 ) __attribute__ ( ( cold ) ) return ; }")
}
