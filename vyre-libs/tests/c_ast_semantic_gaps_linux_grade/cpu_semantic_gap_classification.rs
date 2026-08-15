// CPU classification contracts for Linux-grade C semantic gaps.
//
// Constructs not fully exercised by the hostile-parser or corpus suites:
//   * inner-typedef shadowing an outer typedef in nested block scopes
//   * enum definitions carrying GNU attributes
//   * GNU attributes on function parameters
//   * asm aliases on function declarations
//   * mixed designated / non-designated initializers
//   * incomplete initializer lists
//
// Backend parity for the same cases, and the AST-to-PG row contract, come from
// `semantic_gap_constructs::CASES` in the sibling arm and in
// `c_ast_parity_matrix_cpu_reference`, so nothing here dispatches a kernel.
//
// The asm-alias, mixed-initializer and incomplete-initializer contracts used to
// live in `vyre-driver-wgpu/tests`. Nothing in them dispatches anything, so
// putting them there made the CPU classification of three constructs depend on
// a GPU driver crate compiling, and left `cargo test -p vyre-libs` unable to
// check them at all.

use super::semantic_gap_constructs::*;
use crate::c_frontend::rows::{
    flags_at, kind_at, row_indices, TYPEDEF_FLAG_DECL, TYPEDEF_FLAG_VISIBLE,
};
use crate::c_frontend::token_fixture::{annotate_and_classify, classify};
use vyre_libs::parsing::c::parse::vast::{
    C_AST_KIND_ASM_TEMPLATE, C_AST_KIND_ASSIGN_EXPR, C_AST_KIND_ATTRIBUTE_PACKED,
    C_AST_KIND_ATTRIBUTE_UNUSED, C_AST_KIND_ENUMERATOR_DECL, C_AST_KIND_ENUM_DECL,
    C_AST_KIND_FUNCTION_DECLARATOR, C_AST_KIND_GNU_ATTRIBUTE, C_AST_KIND_INITIALIZER_LIST,
    C_AST_KIND_INLINE_ASM, C_AST_KIND_MEMBER_ACCESS_EXPR,
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

#[test]
pub(crate) fn cpu_asm_alias_classifies() {
    let fix = fixture_asm_alias();
    let typed = classify(&fix);

    assert_eq!(
        kind_at(&typed, 1),
        node_kind::FUNCTION_DECL,
        "function name `foo` must classify as FUNCTION_DECL"
    );
    assert_eq!(
        row_indices(&typed, C_AST_KIND_INLINE_ASM),
        vec![5],
        "asm alias must classify as INLINE_ASM (or a dedicated alias kind)"
    );
    assert_eq!(
        kind_at(&typed, 7),
        C_AST_KIND_ASM_TEMPLATE,
        "asm alias string must classify as ASM_TEMPLATE"
    );
}

#[test]
pub(crate) fn cpu_mixed_designated_and_plain_initializer() {
    let fix = fixture_mixed_designated_and_plain_init();
    let typed = classify(&fix);

    assert_eq!(
        row_indices(&typed, C_AST_KIND_INITIALIZER_LIST),
        vec![4],
        "brace must be an initializer list"
    );
    assert_eq!(
        row_indices(&typed, C_AST_KIND_MEMBER_ACCESS_EXPR),
        vec![7],
        "dot designator must classify as MEMBER_ACCESS_EXPR"
    );
    assert_eq!(
        row_indices(&typed, C_AST_KIND_ASSIGN_EXPR),
        vec![9],
        "declaration initializer `=` is not an assignment expression; only the designator `=` is"
    );
    let literals = row_indices(&typed, node_kind::LITERAL);
    assert!(literals.contains(&5), "plain literal `1` must be a LITERAL");
    assert!(
        literals.contains(&10),
        "designated value `2` must be a LITERAL"
    );
    assert!(
        literals.contains(&12),
        "trailing plain literal `3` must be a LITERAL"
    );
}

#[test]
pub(crate) fn cpu_incomplete_array_initializer() {
    let fix = fixture_incomplete_array_init();
    let typed = classify(&fix);

    assert_eq!(
        row_indices(&typed, C_AST_KIND_INITIALIZER_LIST),
        vec![6],
        "incomplete initializer brace must still be an initializer list"
    );
    let literals = row_indices(&typed, node_kind::LITERAL);
    assert!(literals.contains(&7), "first element `1` must be a LITERAL");
    assert!(
        literals.contains(&9),
        "second element `2` must be a LITERAL"
    );
}
