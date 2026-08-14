// (use super::* removed  -  parts now share crate-root scope via flat include)

use super::*;
use crate::c_frontend::spelling::c_tokens;

#[test]
pub(crate) fn cpu_typedef_shadowed_by_auto_type_variable() {
    let fix = fixture_typedef_shadowed_by_auto_type();
    let raw = reference_c11_build_vast_nodes(&fix.tok_types, &fix.tok_starts, &fix.tok_lens);
    let annotated = reference_c11_annotate_typedef_names(&raw, fix.source.as_bytes());
    let typed = reference_c11_classify_vast_node_kinds(&annotated);

    assert_eq!(
        word_at(&typed, 0 * VAST_STRIDE_U32),
        C_AST_KIND_TYPEDEF_DECL,
        "typedef keyword must classify"
    );
    assert_eq!(
        fix.tok_types[10], TOK_GNU_AUTO_TYPE,
        "__auto_type must promote"
    );
    assert_ne!(
        word_at(&annotated, 11 * VAST_STRIDE_U32 + FLAGS_FIELD) & ORDINARY_FLAG_DECL,
        0,
        "T after __auto_type must be ordinary decl"
    );
    assert_eq!(
        word_at(&annotated, 15 * VAST_STRIDE_U32 + FLAGS_FIELD) & TYPEDEF_FLAG_VISIBLE,
        0,
        "typedef must be shadowed by __auto_type variable"
    );
    assert_eq!(
        word_at(&typed, 16 * VAST_STRIDE_U32),
        node_kind::BINARY,
        "* must be binary multiply after shadowed T"
    );
}

#[test]
pub(crate) fn gpu_parity_typedef_shadowed_by_auto_type() {
    let fix = fixture_typedef_shadowed_by_auto_type();
    assert_full_pipeline_parity(&fix, "typedef_shadowed_by_auto_type");
}

// ---------------------------------------------------------------------------
// 2. Enum / tag scopes
// ---------------------------------------------------------------------------

pub(crate) fn fixture_struct_tag_forward_declaration() -> Fixture {
    c_tokens("struct S ; typedef struct S S ; struct S * p ;")
}

pub(crate) fn fixture_enum_tag_forward_declaration() -> Fixture {
    c_tokens("enum E ; enum E e ;")
}

#[test]
pub(crate) fn cpu_struct_tag_forward_declaration_coexists_with_typedef() {
    let fix = fixture_struct_tag_forward_declaration();
    let raw = reference_c11_build_vast_nodes(&fix.tok_types, &fix.tok_starts, &fix.tok_lens);
    let annotated = reference_c11_annotate_typedef_names(&raw, fix.source.as_bytes());
    let typed = reference_c11_classify_vast_node_kinds(&annotated);

    assert_eq!(
        word_at(&typed, 0 * VAST_STRIDE_U32),
        C_AST_KIND_STRUCT_DECL,
        "struct keyword must classify"
    );
    assert_eq!(
        word_at(&typed, 1 * VAST_STRIDE_U32),
        node_kind::VARIABLE,
        "forward struct tag S must classify as VARIABLE"
    );
    assert_eq!(
        word_at(&typed, 3 * VAST_STRIDE_U32),
        C_AST_KIND_TYPEDEF_DECL,
        "typedef keyword must classify"
    );
    assert_ne!(
        word_at(&annotated, 6 * VAST_STRIDE_U32 + FLAGS_FIELD) & TYPEDEF_FLAG_DECL,
        0,
        "typedef S must carry DECL flag"
    );
    assert_eq!(
        word_at(&typed, 8 * VAST_STRIDE_U32),
        C_AST_KIND_STRUCT_DECL,
        "second struct keyword must classify"
    );
    assert_eq!(
        word_at(&typed, 10 * VAST_STRIDE_U32),
        C_AST_KIND_POINTER_DECL,
        "star must be POINTER_DECL"
    );
    assert_eq!(
        word_at(&typed, 11 * VAST_STRIDE_U32),
        node_kind::VARIABLE,
        "p must be VARIABLE"
    );
}
