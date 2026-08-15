//! GPU parity tests for token-stream to VAST row construction.
#![cfg(feature = "c-parser")]
#![allow(clippy::too_many_arguments)]
#![allow(deprecated)]
mod c_ast_gpu_parity_support;
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
#[path = "../../tests/support/c_frontend/fixtures/vast_builder_token_streams.rs"]
mod vast_builder_token_streams;

use c_ast_gpu_parity_support::{run_gpu_expr_shape, GpuArm};
use c_frontend::expression_pipeline::assert_shape_row as assert_expr_shape_row;
use c_frontend::parity_matrix::{arm_classified_from_tokens, arm_raw_vast_with_count};
use c_frontend::rows::{assert_kind, assert_vast_row, starts_for_lens, word_at, VAST_STRIDE_U32};
use vast_builder_token_streams::*;
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::parse::vast::{
    reference_c11_build_expression_shape_nodes, reference_c11_build_vast_nodes,
    reference_c11_classify_vast_node_kinds, C_AST_KIND_ARRAY_DECL, C_AST_KIND_ARRAY_SUBSCRIPT_EXPR,
    C_AST_KIND_ASM_CLOBBERS_LIST, C_AST_KIND_ASM_TEMPLATE, C_AST_KIND_ASSIGN_EXPR,
    C_AST_KIND_CAST_EXPR, C_AST_KIND_COMPOUND_LITERAL_EXPR, C_AST_KIND_CONDITIONAL_EXPR,
    C_AST_KIND_DO_STMT, C_AST_KIND_ENUMERATOR_DECL, C_AST_KIND_FIELD_DECL, C_AST_KIND_FOR_STMT,
    C_AST_KIND_FUNCTION_DECLARATOR, C_AST_KIND_FUNCTION_DEFINITION, C_AST_KIND_GNU_ATTRIBUTE,
    C_AST_KIND_GOTO_STMT, C_AST_KIND_IF_STMT, C_AST_KIND_INITIALIZER_LIST, C_AST_KIND_INLINE_ASM,
    C_AST_KIND_MEMBER_ACCESS_EXPR, C_AST_KIND_POINTER_DECL, C_AST_KIND_RETURN_STMT,
    C_AST_KIND_SIZEOF_EXPR, C_AST_KIND_SWITCH_STMT, C_AST_KIND_UNARY_EXPR, C_EXPR_ASSOC_RIGHT,
    C_EXPR_SHAPE_CONDITIONAL,
};
use vyre_primitives::predicate::node_kind;

/// Classified rows for a token stream, already compared against the CPU oracle.
fn classified_rows(tok_types: &[u32], tok_starts: &[u32], tok_lens: &[u32]) -> Vec<u8> {
    arm_classified_from_tokens(&GpuArm, tok_types, tok_starts, tok_lens)
}

#[path = "c11_build_vast_nodes/gpu_declarators_and_anonymous_aggregates.rs"]
mod gpu_declarators_and_anonymous_aggregates;
#[path = "c11_build_vast_nodes/gpu_gnu_attributes_statements_and_operators.rs"]
mod gpu_gnu_attributes_statements_and_operators;
