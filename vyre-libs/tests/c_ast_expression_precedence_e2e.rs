//! End-to-end GPU/CPU parity tests for C expression precedence and associativity.
//!
//! Covers: comma boundaries, assignment chains, nested ternary, logical/bitwise
//! precedence ladders, cast vs parenthesized expression typing, postfix
//! call/index/member, and unary chains.  Every fixture asserts both expression
//! shape rows and PG lowering (kind, span, and tree-link preservation).
#![cfg(feature = "c-parser")]
#![allow(clippy::too_many_arguments, clippy::erasing_op)]
#![allow(deprecated)]
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
#[path = "../../tests/support/c_frontend/fixtures/expression_precedence_e2e.rs"]
mod expression_precedence_e2e;

use crate::c_frontend::expression_pipeline::{
    assert_pg_links_match_vast, assert_pg_preserves_row, assert_shape_none, assert_shape_rows,
    binary_row, conditional_row, run_pipeline, shape_none_row,
};
use crate::c_frontend::rows::{row_indices_by_stride as row_indices, word_at, VAST_STRIDE_U32};
use expression_precedence_e2e::*;
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::parse::vast::{
    C_AST_KIND_ARRAY_SUBSCRIPT_EXPR, C_AST_KIND_ASSIGN_EXPR, C_AST_KIND_CAST_EXPR,
    C_AST_KIND_CONDITIONAL_EXPR, C_AST_KIND_MEMBER_ACCESS_EXPR, C_EXPR_ASSOC_LEFT,
    C_EXPR_ASSOC_RIGHT,
};
use vyre_primitives::predicate::node_kind;

#[path = "c_ast_expression_precedence_e2e/precedence_shapes_lowered_to_pg.rs"]
mod precedence_shapes_lowered_to_pg;
