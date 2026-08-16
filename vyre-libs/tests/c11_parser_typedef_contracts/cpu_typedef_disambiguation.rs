// Contract tests for C typedef/name disambiguation.
//
// These tests assert the *correct* C semantics for typedef/name
// disambiguation.  Where the current reference implementation deviates
// from the standard, the tests fail and document the gap.
//
// Coverage:
//   * typedef T vs variable x in `(T)*p` (cast+deref) vs `(x)*p` (multiply)
//   * typedef shadowing in nested block scopes
//   * struct/enum tag names versus typedef names in declaration contexts
//   * pointer / array / function declarator context preservation

use super::typedef_disambiguation::*;
use crate::c_frontend::rows::{word_at, VAST_STRIDE_U32};
use vyre_libs::parsing::c::parse::vast::{
    reference_c11_build_vast_nodes, reference_c11_classify_vast_node_kinds, C_AST_KIND_ARRAY_DECL,
    C_AST_KIND_CAST_EXPR, C_AST_KIND_FUNCTION_DECLARATOR, C_AST_KIND_POINTER_DECL,
    C_AST_KIND_UNARY_EXPR,
};
use vyre_libs::predicate::node_kind;

// ---------------------------------------------------------------------------
// CPU reference contract tests
// ---------------------------------------------------------------------------

#[test]
pub(crate) fn cpu_reference_typedef_cast_vs_expr_multiply() {
    let (tok_types, tok_starts, tok_lens) = fixture_typedef_cast_vs_expr_multiply();
    let raw = reference_c11_build_vast_nodes(&tok_types, &tok_starts, &tok_lens);
    let typed = reference_c11_classify_vast_node_kinds(&raw);

    // (T)*p  -  T is a typedef name, so (T) introduces a cast.
    assert_eq!(
        word_at(&typed, 10 * VAST_STRIDE_U32),
        C_AST_KIND_CAST_EXPR,
        "(T)*p where T is typedef must classify (T) as cast expression"
    );
    // The * is unary dereference in a cast context.
    assert_eq!(
        word_at(&typed, 13 * VAST_STRIDE_U32),
        C_AST_KIND_UNARY_EXPR,
        "(T)*p star must be unary dereference"
    );

    // (x)*p  -  x is a variable, so (x) is a parenthesised expression and * is multiply.
    assert_ne!(
        word_at(&typed, 16 * VAST_STRIDE_U32),
        C_AST_KIND_CAST_EXPR,
        "(x)*p where x is a variable must NOT classify (x) as cast expression"
    );
    assert_eq!(
        word_at(&typed, 19 * VAST_STRIDE_U32),
        node_kind::BINARY,
        "(x)*p star must be binary multiplication"
    );
}

#[test]
pub(crate) fn cpu_reference_typedef_shadowing_nested() {
    let (tok_types, tok_starts, tok_lens) = fixture_typedef_shadowing_nested();
    let raw = reference_c11_build_vast_nodes(&tok_types, &tok_starts, &tok_lens);
    let typed = reference_c11_classify_vast_node_kinds(&raw);

    // Inside the inner block, T is a variable, so T * b is multiplication.
    assert_eq!(
        word_at(&typed, 15 * VAST_STRIDE_U32),
        node_kind::BINARY,
        "shadowed typedef: T * b must be binary multiplication, not pointer declarator"
    );

    // T itself should be VARIABLE in the inner block (not a type).
    assert_eq!(
        word_at(&typed, 14 * VAST_STRIDE_U32),
        node_kind::VARIABLE,
        "shadowed typedef name used as value must classify as VARIABLE"
    );
}

#[test]
pub(crate) fn cpu_reference_struct_tag_vs_typedef() {
    let (tok_types, tok_starts, tok_lens) = fixture_struct_tag_vs_typedef();
    let raw = reference_c11_build_vast_nodes(&tok_types, &tok_starts, &tok_lens);
    let typed = reference_c11_classify_vast_node_kinds(&raw);

    // struct S *a;  -  * must be a pointer declarator because we are in declaration context.
    assert_eq!(
        word_at(&typed, 21 * VAST_STRIDE_U32),
        C_AST_KIND_POINTER_DECL,
        "struct S *a star must be POINTER_DECL"
    );

    // S *b;  -  typedef name in declaration position, star must also be POINTER_DECL.
    assert_eq!(
        word_at(&typed, 25 * VAST_STRIDE_U32),
        C_AST_KIND_POINTER_DECL,
        "typedef S *b star must be POINTER_DECL"
    );

    // Both a and b must classify as variables (identifiers in declaration).
    assert_eq!(
        word_at(&typed, 22 * VAST_STRIDE_U32),
        node_kind::VARIABLE,
        "identifier a in struct S *a must be VARIABLE"
    );
    assert_eq!(
        word_at(&typed, 26 * VAST_STRIDE_U32),
        node_kind::VARIABLE,
        "identifier b in S *b must be VARIABLE"
    );
}

#[test]
pub(crate) fn cpu_reference_declarator_contexts() {
    let (tok_types, tok_starts, tok_lens) = fixture_declarator_contexts();
    let raw = reference_c11_build_vast_nodes(&tok_types, &tok_starts, &tok_lens);
    let typed = reference_c11_classify_vast_node_kinds(&raw);

    // int *a[10];   -  star is POINTER_DECL, bracket is ARRAY_DECL
    assert_eq!(
        word_at(&typed, 7 * VAST_STRIDE_U32),
        C_AST_KIND_POINTER_DECL,
        "int *a[10] star must be POINTER_DECL"
    );
    assert_eq!(
        word_at(&typed, 9 * VAST_STRIDE_U32),
        C_AST_KIND_ARRAY_DECL,
        "int *a[10] bracket must be ARRAY_DECL"
    );

    // int (*a)[10];  -  star is POINTER_DECL, bracket is ARRAY_DECL
    assert_eq!(
        word_at(&typed, 15 * VAST_STRIDE_U32),
        C_AST_KIND_POINTER_DECL,
        "int (*a)[10] inner star must be POINTER_DECL"
    );
    assert_eq!(
        word_at(&typed, 18 * VAST_STRIDE_U32),
        C_AST_KIND_ARRAY_DECL,
        "int (*a)[10] bracket must be ARRAY_DECL"
    );

    // int *f(int);  -  star is POINTER_DECL, f is FUNCTION_DECL, ( is FUNCTION_DECLARATOR
    assert_eq!(
        word_at(&typed, 23 * VAST_STRIDE_U32),
        C_AST_KIND_POINTER_DECL,
        "int *f(int) star must be POINTER_DECL"
    );
    assert_eq!(
        word_at(&typed, 24 * VAST_STRIDE_U32),
        node_kind::FUNCTION_DECL,
        "int *f(int) identifier f must be FUNCTION_DECL"
    );
    assert_eq!(
        word_at(&typed, 25 * VAST_STRIDE_U32),
        C_AST_KIND_FUNCTION_DECLARATOR,
        "int *f(int) parameter paren must be FUNCTION_DECLARATOR"
    );

    // int (*f)(int);  -  star is POINTER_DECL, parameter ( is FUNCTION_DECLARATOR
    assert_eq!(
        word_at(&typed, 31 * VAST_STRIDE_U32),
        C_AST_KIND_POINTER_DECL,
        "int (*f)(int) inner star must be POINTER_DECL"
    );
    assert_eq!(
        word_at(&typed, 34 * VAST_STRIDE_U32),
        C_AST_KIND_FUNCTION_DECLARATOR,
        "int (*f)(int) parameter paren must be FUNCTION_DECLARATOR"
    );
}
