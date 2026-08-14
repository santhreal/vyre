//! Typedef-name identity tests for the C VAST semantic annotation pass.

#![cfg(feature = "c-parser")]
#![allow(deprecated)]
mod c_ast_gpu_parity_support;
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;

use c_ast_gpu_parity_support::scope_gpu_support::{run_gpu_annotate, run_gpu_classify};
use c_frontend::rows::assert_words_eq;
use c_frontend::scope_fixture::{
    annotate_cpu, c_atoms, classify_cpu_annotated, fixture, ScopeFixture,
};
use c_frontend::scope_fixture::{
    flags_at, kind_at, ORDINARY_FLAG_DECL, TYPEDEF_FLAG_DECL, TYPEDEF_FLAG_VISIBLE,
};
use vyre_libs::parsing::c::parse::vast::{
    reference_c11_classify_vast_node_kinds, C_AST_KIND_CAST_EXPR, C_AST_KIND_POINTER_DECL,
};

/// `typedef int T; void f(void) { { int T; (T)*p; } (T)*p; }`, the stream whose
/// inner ordinary declaration hides the typedef until the block closes. Both the
/// CPU contract and its GPU parity arm read it.
fn fixture_inner_shadow() -> ScopeFixture {
    fixture(
        "inner_shadow",
        &c_atoms("typedef int T ; void f ( void ) { { int T ; ( T ) * p ; } ( T ) * p ; }"),
    )
}

/// `typedef int S; struct S { int field; }; enum S { A }; void f(void) { (S)*p; }`,
/// where the struct and enum tags must not displace the typedef name.
fn fixture_tag_namespaces() -> ScopeFixture {
    fixture(
        "tag_namespaces",
        &c_atoms(
            "typedef int S ; struct S { int field ; } ; enum S { A } ; void f ( void ) { ( S ) \
             * p ; }",
        ),
    )
}

/// Run annotation and classification on both arms and assert the GPU reproduces
/// the CPU oracle word for word. Returns the GPU buffers so a caller can assert
/// the construct-specific rows it cares about.
fn assert_annotation_and_classifier_parity(fix: &ScopeFixture, label: &str) -> (Vec<u8>, Vec<u8>) {
    let expected_annotations = annotate_cpu(fix);
    let gpu_annotations = run_gpu_annotate(fix);
    assert_words_eq(&gpu_annotations, &expected_annotations, label);

    let expected_typed = reference_c11_classify_vast_node_kinds(&expected_annotations);
    let gpu_typed = run_gpu_classify(&gpu_annotations, fix.tok_types.len());
    assert_words_eq(&gpu_typed, &expected_typed, label);
    (gpu_annotations, gpu_typed)
}

#[test]
fn cpu_typedef_name_and_expression_identifier_are_distinct() {
    let fix = fixture(
        "typedef_vs_expression_identifier",
        &c_atoms("typedef int T ; void f ( void ) { T * p ; ( T ) * p ; int x ; ( x ) * p ; }"),
    );
    let annotated = annotate_cpu(&fix);
    let typed = classify_cpu_annotated(&fix);

    assert_ne!(flags_at(&annotated, 2) & TYPEDEF_FLAG_DECL, 0);
    assert_ne!(flags_at(&annotated, 10) & TYPEDEF_FLAG_VISIBLE, 0);
    assert_eq!(kind_at(&typed, 11), C_AST_KIND_POINTER_DECL);
    assert_eq!(kind_at(&typed, 14), C_AST_KIND_CAST_EXPR);
    assert_eq!(
        kind_at(&typed, 23),
        0,
        "ordinary expression identifier `(x)` must not become a cast"
    );
    assert_ne!(flags_at(&annotated, 21) & ORDINARY_FLAG_DECL, 0);
    assert_eq!(flags_at(&annotated, 24) & TYPEDEF_FLAG_VISIBLE, 0);
}

#[test]
fn cpu_inner_ordinary_declaration_shadows_typedef_until_scope_exit() {
    let fix = fixture_inner_shadow();
    let annotated = annotate_cpu(&fix);
    let typed = classify_cpu_annotated(&fix);

    assert_ne!(flags_at(&annotated, 12) & ORDINARY_FLAG_DECL, 0);
    assert_eq!(flags_at(&annotated, 15) & TYPEDEF_FLAG_VISIBLE, 0);
    assert_eq!(kind_at(&typed, 14), 0);
    assert_ne!(flags_at(&annotated, 22) & TYPEDEF_FLAG_VISIBLE, 0);
    assert_eq!(kind_at(&typed, 21), C_AST_KIND_CAST_EXPR);
}

#[test]
fn cpu_struct_and_enum_tags_do_not_shadow_typedef_namespace() {
    let fix = fixture_tag_namespaces();
    let annotated = annotate_cpu(&fix);
    let typed = classify_cpu_annotated(&fix);

    assert_eq!(flags_at(&annotated, 5) & ORDINARY_FLAG_DECL, 0);
    assert_eq!(flags_at(&annotated, 13) & ORDINARY_FLAG_DECL, 0);
    assert_ne!(flags_at(&annotated, 25) & TYPEDEF_FLAG_VISIBLE, 0);
    assert_eq!(kind_at(&typed, 24), C_AST_KIND_CAST_EXPR);
}

#[test]
fn gpu_annotation_and_classifier_match_cpu_for_shadowing() {
    let fix = fixture_inner_shadow();
    let (_, gpu_typed) = assert_annotation_and_classifier_parity(&fix, "inner shadow");
    assert_eq!(kind_at(&gpu_typed, 14), 0);
    assert_eq!(kind_at(&gpu_typed, 21), C_AST_KIND_CAST_EXPR);
}

#[test]
fn gpu_typedef_visibility_walks_deeper_than_four_block_scopes() {
    let fix = fixture(
        "deep_block_visibility",
        &c_atoms("void f ( void ) { typedef int T ; { { { { { T * p ; } } } } } }"),
    );
    let (gpu_annotations, gpu_typed) =
        assert_annotation_and_classifier_parity(&fix, "deep block visibility");
    assert_ne!(
        flags_at(&gpu_annotations, 15) & TYPEDEF_FLAG_VISIBLE,
        0,
        "typedef declared in the outer function block must remain visible five block scopes down"
    );
    assert_eq!(kind_at(&gpu_typed, 16), C_AST_KIND_POINTER_DECL);
}

#[test]
fn gpu_annotation_and_classifier_match_cpu_for_tags() {
    let fix = fixture_tag_namespaces();
    let (_, gpu_typed) = assert_annotation_and_classifier_parity(&fix, "tag namespaces");
    assert_eq!(kind_at(&gpu_typed, 24), C_AST_KIND_CAST_EXPR);
}
