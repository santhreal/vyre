//! End-to-end GPU/CPU parity tests for C expression precedence and associativity.
//!
//! Covers: comma boundaries, assignment chains, nested ternary, logical/bitwise
//! precedence ladders, cast vs parenthesized expression typing, postfix
//! call/index/member, and unary chains.  Every fixture asserts both expression
//! shape rows and PG lowering (kind, span, and tree-link preservation).
#![cfg(feature = "c-parser")]
#![allow(clippy::too_many_arguments, clippy::erasing_op)]
#![allow(deprecated)]
mod c_ast_gpu_parity_support;
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
#[path = "../../tests/support/c_frontend/fixtures/expression_precedence_e2e.rs"]
mod expression_precedence_e2e;

use c_ast_gpu_parity_support::expression_pipeline::{
    assert_pg_links_match_vast, assert_pg_preserves_row, assert_shape_row, run_pipeline,
};
use c_ast_gpu_parity_support::{
    assert_expression_shape_parity, row_indices_by_stride as row_indices, SENTINEL, VAST_STRIDE_U32,
};
use expression_precedence_e2e::*;
use vyre_libs::parsing::c::parse::vast::{
    C_AST_KIND_UNARY_EXPR, C_EXPR_ASSOC_NONE, C_EXPR_SHAPE_NONE,
};

#[path = "c_ast_expression_precedence_e2e/unary_chains_and_gpu_parity.rs"]
mod unary_chains_and_gpu_parity;
