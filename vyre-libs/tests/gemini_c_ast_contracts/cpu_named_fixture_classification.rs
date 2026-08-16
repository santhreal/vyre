// Gemini-mandated aggressive AST contract tests for VYRE C parser.
//
// Scope: Typedef shadowing, Cast/Pointer ambiguity, Nested FnPtrs,
// Compound Literals, GNU Attributes, Tag Separation, PG Parity.

use super::gemini_named_fixtures::*;
use crate::c_frontend::rows::{row_indices as typed_indices, word_at, VAST_STRIDE_U32};
use crate::c_frontend::scope_fixture::{annotate_cpu, c_atoms, fixture};
use vyre_libs::parsing::c::parse::vast::*;
use vyre_libs::predicate::node_kind;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
pub(crate) fn cpu_reference_typedef_shadowing() {
    let (tok_types, tok_starts, tok_lens) = fixture_typedef_shadowing();
    let raw = reference_c11_build_vast_nodes(&tok_types, &tok_starts, &tok_lens);
    let typed = reference_c11_classify_vast_node_kinds(&raw);

    assert_eq!(
        word_at(&typed, 11 * VAST_STRIDE_U32),
        node_kind::VARIABLE,
        "x must be a variable"
    );
    assert_eq!(
        word_at(&typed, 15 * VAST_STRIDE_U32),
        node_kind::VARIABLE,
        "shadowing T must be a variable"
    );
    assert_eq!(
        word_at(&typed, 17 * VAST_STRIDE_U32),
        node_kind::VARIABLE,
        "use of shadowed T must be a variable"
    );
    assert_eq!(
        word_at(&typed, 23 * VAST_STRIDE_U32),
        node_kind::VARIABLE,
        "y must be a variable after shadow block"
    );
}

#[test]
pub(crate) fn cpu_reference_cast_vs_multiply() {
    let fix = fixture(
        "cast_vs_multiply",
        &c_atoms("typedef int T ; void f ( void ) { ( T ) * x ; int T ; ( T ) * x ; }"),
    );
    let typed = reference_c11_classify_vast_node_kinds(&annotate_cpu(&fix));

    assert_eq!(
        word_at(&typed, 10 * VAST_STRIDE_U32),
        C_AST_KIND_CAST_EXPR,
        "(T)*x must be cast when T is typedef"
    );
    assert_ne!(
        word_at(&typed, 19 * VAST_STRIDE_U32),
        C_AST_KIND_CAST_EXPR,
        "(T)*x must NOT be cast when T is shadowed by variable"
    );
}

#[test]
pub(crate) fn cpu_reference_nested_fnptr() {
    let (tok_types, tok_starts, tok_lens) = fixture_nested_fnptr();
    let raw = reference_c11_build_vast_nodes(&tok_types, &tok_starts, &tok_lens);
    let typed = reference_c11_classify_vast_node_kinds(&raw);

    let ptr_decls = typed_indices(&typed, C_AST_KIND_POINTER_DECL);
    assert_eq!(
        ptr_decls.len(),
        2,
        "must find two pointer declarators in nested fnptr"
    );
    assert!(ptr_decls.contains(&2));
    assert!(ptr_decls.contains(&4));

    let fn_decls = typed_indices(&typed, C_AST_KIND_FUNCTION_DECLARATOR);
    assert_eq!(
        fn_decls.len(),
        2,
        "must find two function declarators in nested fnptr"
    );
}
