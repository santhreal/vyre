use super::*;

#[test]
fn annotation_label_does_not_affect_typedef_visibility() {
    let fix = fixture(
        "label_typedef",
        &c_atoms("typedef int T ; void f ( void ) { T : ( T ) * p ; }"),
    );
    let ann = annotate_cpu(&fix);
    let typed = classify_cpu_annotated(&fix);

    // Label T at token 10 should not shadow typedef T
    assert_ne!(
        flags_at(&ann, 10) & TYPEDEF_FLAG_VISIBLE,
        0,
        "typedef T must remain visible despite label with same name"
    );
    assert_eq!(
        kind_at(&typed, 12),
        C_AST_KIND_CAST_EXPR,
        "(T) must remain cast despite label"
    );
}

// ---------------------------------------------------------------------------
// Typedef + tag same name
// ---------------------------------------------------------------------------

#[test]
fn annotation_struct_tag_and_typedef_same_name_both_visible() {
    let fix = fixture(
        "tag_typedef_same",
        &c_atoms(
            "struct S { int x ; } ; typedef struct S S ; void f ( void ) { struct S * a ; S * b \
             ; }",
        ),
    );
    let ann = annotate_cpu(&fix);
    let typed = classify_cpu_annotated(&fix);

    // typedef S at token 11
    assert_ne!(
        flags_at(&ann, 11) & TYPEDEF_FLAG_DECL,
        0,
        "typedef S must be decl"
    );
    // struct S * a: star must be POINTER_DECL
    assert_eq!(
        kind_at(&typed, 21),
        C_AST_KIND_POINTER_DECL,
        "struct S * a must be pointer decl"
    );
    // S * b: star must be POINTER_DECL
    assert_eq!(
        kind_at(&typed, 25),
        C_AST_KIND_POINTER_DECL,
        "typedef S * b must be pointer decl"
    );
}

// ---------------------------------------------------------------------------
// GPU parity
// ---------------------------------------------------------------------------

#[test]
fn gpu_parity_scope_tree_struct_tag_vs_ordinary() {
    let fix = fixture(
        "gpu_struct_tag",
        &c_atoms("struct S { int x ; } ; int S ; void f ( void ) { }"),
    );
    let expected = scope_tree_for(&fix);
    let gpu = run_gpu_scope_tree(&fix);
    assert_eq!(
        gpu, expected,
        "GPU scope tree must match CPU for struct tag vs ordinary"
    );
}

#[test]
fn gpu_parity_scope_tree_label_namespace() {
    let fix = fixture("gpu_label", &c_atoms("void f ( void ) { L : goto L ; }"));
    let expected = scope_tree_for(&fix);
    let gpu = run_gpu_scope_tree(&fix);
    assert_eq!(
        gpu, expected,
        "GPU scope tree must match CPU for label namespace"
    );
}

#[test]
fn gpu_parity_annotation_tag_typedef_same_name() {
    let fix = fixture(
        "gpu_tag_td",
        &c_atoms("struct S { int x ; } ; typedef struct S S ; void f ( void ) { struct S * a ; }"),
    );
    let expected_ann = annotate_cpu(&fix);
    let gpu_ann = run_gpu_annotate(&fix);
    assert_eq!(
        gpu_ann, expected_ann,
        "GPU annotation must match CPU for tag+typedef same name"
    );
}

#[test]
fn gpu_parity_classifier_enum_constant_context() {
    let fix = fixture(
        "gpu_enum",
        &c_atoms("enum E { A , B } ; void f ( void ) { return A ; }"),
    );
    let expected_ann = annotate_cpu(&fix);
    let gpu_ann = run_gpu_annotate(&fix);
    assert_eq!(
        gpu_ann, expected_ann,
        "GPU annotation must match CPU for enum context"
    );

    let expected_typed = reference_c11_classify_vast_node_kinds(&expected_ann);
    let gpu_typed = run_gpu_classify(&gpu_ann, fix.tok_types.len());
    assert_eq!(
        gpu_typed, expected_typed,
        "GPU classifier must match CPU for enum context"
    );
}

#[test]
fn gpu_parity_scope_tree_enum_tag_vs_variable() {
    let fix = fixture(
        "gpu_enum_var",
        &c_atoms("enum E { A } ; int E ; void f ( void ) { }"),
    );
    let expected = scope_tree_for(&fix);
    let gpu = run_gpu_scope_tree(&fix);
    assert_eq!(
        gpu, expected,
        "GPU scope tree must match CPU for enum tag vs variable"
    );
}

#[test]
fn gpu_parity_annotation_label_does_not_shadow_typedef() {
    let fix = fixture(
        "gpu_label_td",
        &c_atoms("typedef int T ; void f ( void ) { T : ( T ) * p ; }"),
    );
    let expected_ann = annotate_cpu(&fix);
    let gpu_ann = run_gpu_annotate(&fix);
    assert_eq!(
        gpu_ann, expected_ann,
        "GPU annotation must match CPU for label+typedef"
    );
}
