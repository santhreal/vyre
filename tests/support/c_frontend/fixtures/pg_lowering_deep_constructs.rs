//! Token fixtures for deep property-graph semantic lowering: labels, statement expressions,
//! designators, and function definitions.
//!
//! The CPU contracts in `vyre-libs/tests` and the backend parity arm in the
//! driver crate build the same token streams, so the fixtures have one owner
//! here rather than a copy per crate.

use crate::c_frontend::spelling::c_tokens;
use crate::c_frontend::token_fixture::Fixture;

pub(crate) fn fixture_label_case_default() -> Fixture {
    c_tokens("void f ( int x ) { switch ( x ) { case 1 : target : return ; default : break ; } }")
}

pub(crate) fn fixture_statement_expr() -> Fixture {
    c_tokens("int x = ( { int y = 1 ; y ; } ) ;")
}

pub(crate) fn fixture_initializer_designator() -> Fixture {
    c_tokens("struct S { int a ; int b [ 2 ] ; } s = { . a = 1 , . b [ 0 ] = 2 } ;")
}

pub(crate) fn fixture_function_definition() -> Fixture {
    c_tokens("typedef int T ; int f ( T x ) { return x ; }")
}

pub(crate) fn fixture_control_flow_roles() -> Fixture {
    c_tokens(
        "void f ( ) { if ( cond ) return ; for ( ; ; ) while ( cond ) do { break ; \
         continue ; } while ( 0 ) ; switch ( x ) { case 1 : goto end ; default : break ; \
         } end : return ; }",
    )
}

pub(crate) fn fixture_alignof_expression() -> Fixture {
    c_tokens("int x = _Alignof ( int ) ;")
}

pub(crate) fn fixture_function_pointer_declarator() -> Fixture {
    c_tokens("static void ( * const ops [ ] ) ( struct device * ) = { probe , remove } ;")
}
