//! CPU reference C11 AST construction across a corpus covering every declaration, statement, and
//! expression construct.
//!
//! The VAST builder / classifier / PG lowerer dispatch is owned by
//! `c_ast_gpu_parity_support`. What stays here is the corpus and the node kinds
//! each construct must produce.
#![cfg(feature = "c-parser")]
#![allow(clippy::type_complexity)]
#![allow(deprecated)]
mod c_ast_gpu_parity_support;
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
#[path = "../../tests/support/c_frontend/fixtures/complete_construct_corpus.rs"]
mod complete_construct_corpus;

use c_ast_gpu_parity_support::{
    run_gpu_classifier_with_count as run_gpu_classifier,
    run_gpu_pg_lower_with_count as run_gpu_pg_lower,
    run_gpu_vast_builder_from_parts as run_gpu_vast_builder,
};
use c_frontend::rows::{
    assert_kind, pg_word_at, row_indices as typed_indices, word_at, PG_STRIDE_U32, VAST_STRIDE_U32,
};
use complete_construct_corpus::*;
use vyre_libs::parsing::c::lower::reference_ast_to_pg_nodes;
use vyre_libs::parsing::c::parse::vast::{
    reference_c11_build_vast_nodes, reference_c11_classify_vast_node_kinds, C_AST_KIND_ARRAY_DECL,
    C_AST_KIND_ARRAY_SUBSCRIPT_EXPR, C_AST_KIND_CONDITIONAL_EXPR, C_AST_KIND_ENUMERATOR_DECL,
    C_AST_KIND_FIELD_DECL, C_AST_KIND_FUNCTION_DECLARATOR, C_AST_KIND_FUNCTION_DEFINITION,
    C_AST_KIND_GNU_ATTRIBUTE, C_AST_KIND_IF_STMT, C_AST_KIND_INITIALIZER_LIST,
    C_AST_KIND_INLINE_ASM, C_AST_KIND_POINTER_DECL, C_AST_KIND_RETURN_STMT, C_AST_KIND_SIZEOF_EXPR,
};
use vyre_libs::predicate::node_kind;

#[path = "c11_ast_corpus_complete_constructs/gpu_classifier_aggregates_and_designators.rs"]
mod gpu_classifier_aggregates_and_designators;
#[path = "c11_ast_corpus_complete_constructs/gpu_classifier_asm_enums_and_sizeof.rs"]
mod gpu_classifier_asm_enums_and_sizeof;
#[path = "c11_ast_corpus_complete_constructs/gpu_pg_lowering_enums_sizeof_and_statement_expressions.rs"]
mod gpu_pg_lowering_enums_sizeof_and_statement_expressions;
#[path = "c11_ast_corpus_complete_constructs/gpu_pg_lowering_function_pointers_and_asm.rs"]
mod gpu_pg_lowering_function_pointers_and_asm;
#[path = "c11_ast_corpus_complete_constructs/gpu_statement_expressions_and_macro_declarations.rs"]
mod gpu_statement_expressions_and_macro_declarations;
#[path = "c11_ast_corpus_complete_constructs/gpu_vast_builder_designators_and_statement_expressions.rs"]
mod gpu_vast_builder_designators_and_statement_expressions;
#[path = "c11_ast_corpus_complete_constructs/pg_lowering_and_gpu_parity.rs"]
mod pg_lowering_and_gpu_parity;
