//! GPU parity tests for token-stream to VAST row construction.
#![cfg(feature = "c-parser")]
#![allow(clippy::too_many_arguments)]
#![allow(deprecated)]
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
#[path = "../../tests/support/c_frontend/fixtures/vast_builder_token_streams.rs"]
mod vast_builder_token_streams;

use crate::c_frontend::expression_pipeline::{
    assert_shape_rows as assert_expr_shape_rows, conditional_row,
};
use crate::c_frontend::rows::{
    assert_kind, assert_vast_row, row_indices as typed_indices, word_at, VAST_STRIDE_U32,
};
use crate::c_frontend::spelling::c_rows;
use vast_builder_token_streams::*;
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::parse::vast::{
    reference_c11_build_expression_shape_nodes, reference_c11_build_vast_nodes,
    reference_c11_classify_vast_node_kinds, C_AST_KIND_ARRAY_DECL, C_AST_KIND_ARRAY_SUBSCRIPT_EXPR,
    C_AST_KIND_ASM_CLOBBERS_LIST, C_AST_KIND_ASM_TEMPLATE, C_AST_KIND_ASSIGN_EXPR,
    C_AST_KIND_BREAK_STMT, C_AST_KIND_CASE_STMT, C_AST_KIND_CAST_EXPR,
    C_AST_KIND_COMPOUND_LITERAL_EXPR, C_AST_KIND_CONDITIONAL_EXPR, C_AST_KIND_CONTINUE_STMT,
    C_AST_KIND_DEFAULT_STMT, C_AST_KIND_DO_STMT, C_AST_KIND_ELSE_STMT, C_AST_KIND_ENUMERATOR_DECL,
    C_AST_KIND_FIELD_DECL, C_AST_KIND_FOR_STMT, C_AST_KIND_FUNCTION_DECLARATOR,
    C_AST_KIND_FUNCTION_DEFINITION, C_AST_KIND_GNU_ATTRIBUTE, C_AST_KIND_GOTO_STMT,
    C_AST_KIND_IF_STMT, C_AST_KIND_INITIALIZER_LIST, C_AST_KIND_INLINE_ASM,
    C_AST_KIND_MEMBER_ACCESS_EXPR, C_AST_KIND_POINTER_DECL, C_AST_KIND_RETURN_STMT,
    C_AST_KIND_SIZEOF_EXPR, C_AST_KIND_SWITCH_STMT, C_AST_KIND_UNARY_EXPR, C_AST_KIND_WHILE_STMT,
    C_EXPR_ASSOC_LEFT, C_EXPR_SHAPE_BINARY,
};
use vyre_libs::predicate::node_kind;

#[path = "c11_build_vast_nodes/cpu_delimiter_tree_and_statement_keywords.rs"]
mod cpu_delimiter_tree_and_statement_keywords;
#[path = "c11_build_vast_nodes/cpu_expression_operators_and_declarators.rs"]
mod cpu_expression_operators_and_declarators;
#[path = "c11_build_vast_nodes/cpu_hostile_casts_and_qualified_declarators.rs"]
mod cpu_hostile_casts_and_qualified_declarators;
