// Semantic-gap contracts for Linux-grade C AST / compiler front-end.
//
// High-signal tests that encode desired behavior for constructs not fully
// exercised by the existing hostile-parser or corpus suites:
//   * inner-typedef shadowing an outer typedef in nested block scopes
//   * enum definitions carrying GNU attributes
//   * GNU attributes on function parameters
//   * asm aliases on function declarations
//   * mixed designated / non-designated initializers
//   * incomplete initializer lists
//   * typedef-of-function-pointer used as a type specifier
//   * AST-to-PG lowering preservation for the above

use super::semantic_gap_constructs::*;
use crate::c_frontend::rows::{
    flags_at, kind_at, row_indices, TYPEDEF_FLAG_DECL, TYPEDEF_FLAG_VISIBLE,
};
use crate::c_frontend::token_fixture::{annotate_and_classify, classify};
use vyre_libs::parsing::c::parse::vast::{
    C_AST_KIND_ATTRIBUTE_PACKED, C_AST_KIND_ATTRIBUTE_UNUSED, C_AST_KIND_ENUMERATOR_DECL,
    C_AST_KIND_ENUM_DECL, C_AST_KIND_FUNCTION_DECLARATOR, C_AST_KIND_GNU_ATTRIBUTE,
};
use vyre_primitives::predicate::node_kind;

// ---------------------------------------------------------------------------
// CPU reference contracts
// ---------------------------------------------------------------------------

#[test]
pub(crate) fn cpu_inner_typedef_shadows_outer_typedef() {
    let fix = fixture_inner_typedef_shadows_outer();
    let (annotated, typed) = annotate_and_classify(&fix);

    // Outer typedef declaration
    assert_ne!(
        flags_at(&annotated, 2) & TYPEDEF_FLAG_DECL,
        0,
        "outer typedef `T` must carry TYPEDEF_FLAG_DECL"
    );
    // Inner typedef declaration
    assert_ne!(
        flags_at(&annotated, 12) & TYPEDEF_FLAG_DECL,
        0,
        "inner typedef `T` must carry TYPEDEF_FLAG_DECL"
    );
    // Use of inner typedef
    assert_ne!(
        flags_at(&annotated, 14) & TYPEDEF_FLAG_VISIBLE,
        0,
        "`T` inside `f` must be visible as the inner typedef"
    );
    assert_eq!(
        kind_at(&typed, 15),
        node_kind::VARIABLE,
        "`x` declared with inner typedef must classify as VARIABLE"
    );
    // Use of outer typedef after block
    assert_ne!(
        flags_at(&annotated, 18) & TYPEDEF_FLAG_VISIBLE,
        0,
        "`T` after `f` must be visible as the outer typedef"
    );
    assert_eq!(
        kind_at(&typed, 19),
        node_kind::VARIABLE,
        "`y` declared with restored outer typedef must classify as VARIABLE"
    );
}

#[test]
pub(crate) fn cpu_enum_with_attribute_classifies() {
    let fix = fixture_enum_with_attribute();
    let typed = classify(&fix);

    assert_eq!(
        kind_at(&typed, 0),
        C_AST_KIND_ENUM_DECL,
        "enum keyword must classify as ENUM_DECL"
    );
    assert_eq!(
        row_indices(&typed, C_AST_KIND_GNU_ATTRIBUTE),
        vec![1],
        "__attribute__ must classify as GNU_ATTRIBUTE"
    );
    assert_eq!(
        row_indices(&typed, C_AST_KIND_ATTRIBUTE_PACKED),
        vec![4],
        "packed must classify as ATTRIBUTE_PACKED"
    );
    assert_eq!(
        row_indices(&typed, C_AST_KIND_ENUMERATOR_DECL),
        vec![9, 11],
        "enumerators A and B must classify as ENUMERATOR_DECL"
    );
}

#[test]
pub(crate) fn cpu_parameter_with_attribute_classifies() {
    let fix = fixture_parameter_with_attribute();
    let typed = classify(&fix);

    assert_eq!(
        kind_at(&typed, 1),
        node_kind::FUNCTION_DECL,
        "function name `f` must classify as FUNCTION_DECL"
    );
    assert_eq!(
        row_indices(&typed, C_AST_KIND_FUNCTION_DECLARATOR),
        vec![2],
        "parameter-list paren must classify as FUNCTION_DECLARATOR"
    );
    assert_eq!(
        row_indices(&typed, C_AST_KIND_GNU_ATTRIBUTE),
        vec![4],
        "parameter attribute must classify as GNU_ATTRIBUTE"
    );
    assert_eq!(
        row_indices(&typed, C_AST_KIND_ATTRIBUTE_UNUSED),
        vec![7],
        "unused must classify as ATTRIBUTE_UNUSED"
    );
    assert_eq!(
        kind_at(&typed, 10),
        node_kind::VARIABLE,
        "parameter name `x` must classify as VARIABLE"
    );
}
