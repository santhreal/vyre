//! Token fixtures for the C declarator matrix contracts.
//!
//! The CPU contracts in `vyre-libs/tests` and the backend parity arm in the
//! driver crate build the same token streams, so the fixtures have one owner
//! here rather than a copy per crate.
//!
//! [`CASES`] is that one owner's case list. Both arms iterate it, so a fixture
//! added below and named there is proven on the CPU oracle and on every
//! backend arm at once. `c_ast_parity_case_matrix_gate` fails when a builder
//! here is missing from it, which is how `fixture_gnu_restrict_qualifier`
//! came to be CPU-proven and backend-unproven.

use crate::c_frontend::parity_matrix::ParityCase;
use crate::c_frontend::spelling::c_tokens;
use crate::c_frontend::token_fixture::Fixture;

/// Every declarator-matrix construct both arms evaluate.
pub(crate) const CASES: &[ParityCase] = &[
    ParityCase::new("pointer_to_array", fixture_pointer_to_array),
    ParityCase::new(
        "storage_class_multi_declarator",
        fixture_storage_class_multi_declarator,
    ),
    ParityCase::new(
        "parameter_array_static_restrict",
        fixture_parameter_array_static_restrict,
    ),
    ParityCase::new(
        "nested_typedef_complex_declarator",
        fixture_nested_typedef_complex_declarator,
    ),
    ParityCase::new(
        "struct_tag_with_mixed_declarators",
        fixture_struct_tag_with_mixed_declarators,
    ),
    ParityCase::new(
        "union_tag_with_mixed_declarators",
        fixture_union_tag_with_mixed_declarators,
    ),
    ParityCase::new(
        "enum_tag_with_mixed_declarators",
        fixture_enum_tag_with_mixed_declarators,
    ),
    ParityCase::new(
        "heavy_qualifiers_and_storage_multi_decl",
        fixture_heavy_qualifiers_and_storage_multi_decl,
    ),
    ParityCase::new(
        "abstract_declarator_with_qualifiers",
        fixture_abstract_declarator_with_qualifiers,
    ),
    ParityCase::new("gnu_restrict_qualifier", fixture_gnu_restrict_qualifier),
];

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// `int (*p)[4];`
pub(crate) fn fixture_pointer_to_array() -> Fixture {
    c_tokens("int ( * p ) [ 4 ] ;")
}

/// `static const int *p, arr[4];`
pub(crate) fn fixture_storage_class_multi_declarator() -> Fixture {
    c_tokens("static const int * p , arr [ 4 ] ;")
}

/// `void f(int arr[static restrict 10]);`
pub(crate) fn fixture_parameter_array_static_restrict() -> Fixture {
    c_tokens("void f ( int arr [ static restrict 10 ] ) ;")
}

/// ```c
/// typedef int (*fn_t)(int);
/// fn_t f;
/// ```
pub(crate) fn fixture_nested_typedef_complex_declarator() -> Fixture {
    c_tokens("typedef int ( * fn_t ) ( int ) ; fn_t f ;")
}

/// ```c
/// struct foo { int x; } *p, arr[2];
/// ```
pub(crate) fn fixture_struct_tag_with_mixed_declarators() -> Fixture {
    c_tokens("struct foo { int x ; } * p , arr [ 2 ] ;")
}

/// ```c
/// union cell { char c; int i; } u, *up;
/// ```
pub(crate) fn fixture_union_tag_with_mixed_declarators() -> Fixture {
    c_tokens("union cell { char c ; int i ; } u , * up ;")
}

/// ```c
/// enum mode { ON, OFF } ev, *ep;
/// ```
pub(crate) fn fixture_enum_tag_with_mixed_declarators() -> Fixture {
    c_tokens("enum mode { ON , OFF } ev , * ep ;")
}

/// `extern volatile char * const * restrict x, y[8];`
pub(crate) fn fixture_heavy_qualifiers_and_storage_multi_decl() -> Fixture {
    c_tokens("extern volatile char * const * restrict x , y [ 8 ] ;")
}

/// `(const int (*)(void))p;`
pub(crate) fn fixture_abstract_declarator_with_qualifiers() -> Fixture {
    c_tokens("( const int ( * ) ( void ) ) p ;")
}

/// `char * __restrict z;`
pub(crate) fn fixture_gnu_restrict_qualifier() -> Fixture {
    c_tokens("char * __restrict z ;")
}
