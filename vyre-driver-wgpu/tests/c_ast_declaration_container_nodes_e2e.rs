//! GPU/CPU parity tests for declaration container VAST kinds.
//!
//! Linux-grade C depends on aggregate tag declarations, typedefs, function
//! definitions, bitfields, and `_Static_assert` being semantic AST rows rather
//! than raw keyword noise.

#![cfg(feature = "c-parser")]
#![allow(deprecated)]

use c_frontend::spelling::c_tokens;
use vyre_libs::parsing::c::parse::vast::{
    reference_c11_annotate_typedef_names, reference_c11_build_vast_nodes,
    reference_c11_classify_vast_node_kinds, C_AST_KIND_BIT_FIELD_DECL, C_AST_KIND_ENUM_DECL,
    C_AST_KIND_FUNCTION_DEFINITION, C_AST_KIND_STATIC_ASSERT_DECL, C_AST_KIND_STRUCT_DECL,
    C_AST_KIND_TYPEDEF_DECL, C_AST_KIND_UNION_DECL,
};
use vyre_primitives::predicate::node_kind;

const VAST_STRIDE_U32: usize = 10;

mod c_ast_gpu_parity_support;
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
use c_ast_gpu_parity_support::{row_indices, run_gpu_classifier_with_count, Fixture};

fn classify(fix: &Fixture) -> Vec<u8> {
    let raw = reference_c11_build_vast_nodes(&fix.tok_types, &fix.tok_starts, &fix.tok_lens);
    let annotated = reference_c11_annotate_typedef_names(&raw, fix.source.as_bytes());
    let expected = reference_c11_classify_vast_node_kinds(&annotated);
    let actual =
        run_gpu_classifier_with_count(&annotated, (annotated.len() / (VAST_STRIDE_U32 * 4)) as u32);
    assert_eq!(
        actual, expected,
        "GPU C VAST classifier must match the CPU oracle"
    );
    expected
}

fn aggregate_fixture() -> Fixture {
    c_tokens(
        "struct foo { int a ; unsigned flags : 3 ; } ; union cell { long raw ; } ; enum mode { \
         MODE_A = 1 , MODE_B } ;",
    )
}

#[test]
fn aggregate_containers_and_bitfields_are_semantic_rows() {
    let typed = classify(&aggregate_fixture());
    assert_eq!(row_indices(&typed, C_AST_KIND_STRUCT_DECL), vec![0]);
    assert_eq!(row_indices(&typed, C_AST_KIND_BIT_FIELD_DECL), vec![7]);
    assert_eq!(row_indices(&typed, C_AST_KIND_UNION_DECL), vec![13]);
    assert_eq!(row_indices(&typed, C_AST_KIND_ENUM_DECL), vec![21]);
    assert_eq!(
        row_indices(&typed, node_kind::FUNCTION_DECL),
        Vec::<usize>::new()
    );
}

#[test]
fn forward_opaque_tags_are_not_flat_keyword_noise() {
    let fixture = c_tokens("struct opaque * p ; union payload ; enum state ;");
    let typed = classify(&fixture);
    assert_eq!(row_indices(&typed, C_AST_KIND_STRUCT_DECL), vec![0]);
    assert_eq!(row_indices(&typed, C_AST_KIND_UNION_DECL), vec![5]);
    assert_eq!(row_indices(&typed, C_AST_KIND_ENUM_DECL), vec![8]);
}

#[test]
fn typedef_and_function_definition_have_distinct_contract_rows() {
    let fixture = c_tokens(
        "typedef unsigned long size_t ; int decl ( int a ) ; int defn ( int a ) { return a ; }",
    );
    let typed = classify(&fixture);
    assert_eq!(row_indices(&typed, C_AST_KIND_TYPEDEF_DECL), vec![0]);
    assert_eq!(row_indices(&typed, node_kind::FUNCTION_DECL), vec![6]);
    assert_eq!(
        row_indices(&typed, C_AST_KIND_FUNCTION_DEFINITION),
        vec![13]
    );
}

#[test]
fn static_assert_is_a_declaration_node() {
    let fixture = c_tokens("_Static_assert ( 1 , \"ok\" ) ;");
    let typed = classify(&fixture);
    assert_eq!(row_indices(&typed, C_AST_KIND_STATIC_ASSERT_DECL), vec![0]);
}
