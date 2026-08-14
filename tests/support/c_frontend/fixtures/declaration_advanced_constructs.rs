//! Token fixtures for the advanced C declaration contracts.
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
/// struct outer {
///     union {
///         struct { int x; } s;
///         int y;
///     } u;
///     enum { A = 1, B = 2 } e;
/// };
/// ```
pub(crate) fn fixture_nested_struct_union_enum() -> Fixture {
    c_tokens(
        "struct outer { union { struct { int x ; } s ; int y ; } u ; enum { A = 1 , B = 2 \
         } e ; } ;",
    )
}

/// ```c
/// struct {
///     union {
///         int i;
///         float f;
///     };
///     int tag;
/// };
/// ```
pub(crate) fn fixture_anonymous_struct_union() -> Fixture {
    c_tokens("struct { union { int i ; float f ; } ; int tag ; } ;")
}

/// ```c
/// typedef struct Node { int v; } Node, *NodePtr;
/// ```
pub(crate) fn fixture_typedef_multiple_declarators() -> Fixture {
    c_tokens("typedef struct Node { int v ; } Node , * NodePtr ;")
}

/// ```c
/// const int * const * volatile * restrict p;
/// ```
pub(crate) fn fixture_deeply_nested_pointer() -> Fixture {
    c_tokens("const int * const * volatile * restrict p ;")
}

/// ```c
/// static inline int f(void);
/// extern register int x;
/// _Thread_local _Atomic int y;
/// ```
pub(crate) fn fixture_storage_class_combinations() -> Fixture {
    c_tokens(
        "static inline int f ( void ) ; extern register int x ; _Thread_local _Atomic int \
         y ;",
    )
}

/// ```c
/// struct {
///     unsigned int a : 4;
///     struct {
///         int b : 8;
///         unsigned int : 0;
///     } inner;
/// };
/// ```
pub(crate) fn fixture_bitfield_nested_struct() -> Fixture {
    c_tokens(
        "struct { unsigned int a : 4 ; struct { int b : 8 ; unsigned int : 0 ; } inner ; \
         } ;",
    )
}

/// ```c
/// struct {
///     __attribute__((aligned(8))) int x;
/// };
/// typedef int __attribute__((packed)) packed_int;
/// ```
pub(crate) fn fixture_gnu_attribute_field_and_typedef() -> Fixture {
    c_tokens(
        "struct { __attribute__ ( ( aligned ( 8 ) ) ) int x ; } ; typedef int \
         __attribute__ ( ( packed ) ) packed_int ;",
    )
}

/// ```c
/// int (**fp)(void);
/// ```
pub(crate) fn fixture_function_pointer_to_pointer() -> Fixture {
    c_tokens("int ( * * fp ) ( void ) ;")
}

/// ```c
/// int (*handlers[4])(int, const char * restrict);
/// ```
pub(crate) fn fixture_array_of_function_pointers_qualified() -> Fixture {
    c_tokens("int ( * handlers [ 4 ] ) ( int , const char * restrict ) ;")
}
