//! Integration contracts for Linux-grade C declarator matrices.
//!
//! Coverage:
//!   * pointer-to-array declarators (`int (*p)[4];`)
//!   * storage-class specifiers threaded through multi-declarator lists
//!   * parameter array declarators with `static` / `restrict` (C99)
//!   * nested typedef names inside declarators (function-pointer typedef reuse)
//!   * struct / union / enum tag definitions followed by mixed declarators
//!   * abstract declarators with qualifiers in cast contexts
//!   * GNU `__restrict` normalized to the C restrict qualifier
//!
//! Asserts:
//!   - specifier propagation: standard qualifiers and storage classes stay raw
//!     syntax while declarator identifiers, pointers, arrays and function parens
//!     get precise AST kinds.
//!   - AST classification: POINTER_DECL, ARRAY_DECL, FUNCTION_DECLARATOR,
//!     VARIABLE, FUNCTION_DECL, FIELD_DECL, STRUCT_DECL, UNION_DECL, ENUM_DECL,
//!     ENUMERATOR_DECL.
//!   - typedef annotations: typedef declarations carry TYPEDEF_FLAG_DECL;
//!     typedef uses inside declarator contexts carry TYPEDEF_FLAG_VISIBLE.
//!   - CPU/GPU parity for VAST builder, classifier and PG lowerer, including
//!     stage-specific parity for abstract-declarator casts without typedef names.
//!
//! A missing GPU adapter is a configuration failure, never a silent skip.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
mod c_ast_gpu_parity_support;
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
#[path = "../../tests/support/c_frontend/fixtures/declarator_matrix_constructs.rs"]
mod declarator_matrix_constructs;

use c_ast_gpu_parity_support::{
    assert_pg_preserves_row, assert_words_eq, kind_at, node_count_from_vast, run_gpu_classifier,
    run_gpu_fast_typedef_annotation, run_gpu_pg_lower_with_count as run_gpu_pg_lower,
    run_gpu_vast_builder_from_parts, Fixture,
};
use declarator_matrix_constructs::*;
use vyre_libs::parsing::c::lower::reference_ast_to_pg_nodes;
use vyre_libs::parsing::c::parse::vast::{
    reference_c11_annotate_typedef_names, reference_c11_build_vast_nodes,
    reference_c11_classify_vast_node_kinds,
};

fn run_gpu_annotate(fix: &Fixture, raw_vast: &[u8]) -> Vec<u8> {
    run_gpu_fast_typedef_annotation(fix.source.as_bytes(), raw_vast)
}

fn assert_full_pipeline_parity(fix: &Fixture, label: &str) {
    let raw_cpu = reference_c11_build_vast_nodes(&fix.tok_types, &fix.tok_starts, &fix.tok_lens);
    let raw_gpu = run_gpu_vast_builder_from_parts(&fix.tok_types, &fix.tok_starts, &fix.tok_lens);
    assert_words_eq(
        &raw_gpu,
        &raw_cpu,
        &format!("{label}: raw VAST GPU/CPU parity"),
    );

    let annotated_cpu = reference_c11_annotate_typedef_names(&raw_cpu, fix.source.as_bytes());
    let annotated_gpu = run_gpu_annotate(fix, &raw_gpu);
    assert_words_eq(
        &annotated_gpu,
        &annotated_cpu,
        &format!("{label}: annotated VAST GPU/CPU parity"),
    );

    let typed_cpu = reference_c11_classify_vast_node_kinds(&annotated_cpu);
    let typed_gpu = run_gpu_classifier(&annotated_gpu);
    assert_words_eq(
        &typed_gpu,
        &typed_cpu,
        &format!("{label}: typed VAST GPU/CPU parity"),
    );
}

#[path = "c_ast_declarator_matrix_contracts/pg_lowering_and_gpu_parity.rs"]
mod pg_lowering_and_gpu_parity;
