//! Linux/kernel-grade C AST contracts for macro shapes, builtins, and qualifiers.
//!
//! Constructs under test:
//!   * container_of macro definition as preprocessor token stream + call-shaped usage
//!   * list_entry / list_for_each macro patterns
//!   * __builtin_expect (likely/unlikely) direct usage and macro wrapper preservation
//!   * static inline __attribute__((always_inline)) function definitions
//!   * volatile / _Atomic qualifier promotions in declarations and parameters
//!   * Linux error-label cleanup patterns (goto err; ... err: return -errno;)
//!
//! Every fixture asserts full GPU/CPU parity for build, annotate, and classify.
//! PG preservation is asserted for rows that carry semantic payload.
//! A missing GPU adapter is a configuration failure  -  tests panic loudly.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
mod c_ast_gpu_parity_support;
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
#[path = "../../tests/support/c_frontend/fixtures/linux_macro_builtin_qualifier.rs"]
mod linux_macro_builtin_qualifier;

use c_ast_gpu_parity_support::{
    assert_full_pipeline_parity, assert_pg_preserves_row, kind_at, lexeme_indices,
    node_count_from_vast, row_indices, run_gpu_pg_lower_with_count as run_gpu_pg_lower,
    token_indices_containing,
};
use linux_macro_builtin_qualifier::*;
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::lower::reference_ast_to_pg_nodes;
use vyre_libs::parsing::c::parse::vast::{
    reference_c11_annotate_typedef_names, reference_c11_build_vast_nodes,
    reference_c11_classify_vast_node_kinds, C_AST_KIND_ALIGNOF_EXPR,
    C_AST_KIND_BUILTIN_EXPECT_EXPR, C_AST_KIND_FUNCTION_DEFINITION, C_AST_KIND_GNU_ATTRIBUTE,
    C_AST_KIND_GOTO_STMT, C_AST_KIND_IF_STMT, C_AST_KIND_LABEL_STMT, C_AST_KIND_MEMBER_ACCESS_EXPR,
    C_AST_KIND_POINTER_DECL, C_AST_KIND_RETURN_STMT,
};
use vyre_libs::predicate::node_kind;

#[path = "c_ast_linux_corpus_macro_builtin_and_qualifier_contracts/kernel_macros_builtins_and_qualifiers.rs"]
mod kernel_macros_builtins_and_qualifiers;
