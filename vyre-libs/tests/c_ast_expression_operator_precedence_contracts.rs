//! Contracts for C expression-operator precedence and associativity.
//!
//! Covers every precedence band that participates in expression-shape rows,
//! including shift, relational, equality, compound assignment, ternary
//! conditional, and comma boundaries.  Each fixture asserts the exact root
//! operator and operand links expected from a full precedence-climbing parser.
//! GPU/CPU parity and PG lowering preservation are required for all fixtures.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
#[path = "../../tests/support/c_frontend/fixtures/expression_precedence.rs"]
mod expression_precedence;

use crate::c_frontend::expression_pipeline::{
    assert_pg_links_match_vast, assert_pg_preserves_row, assert_shape_rows, run_pipeline,
    shape_none_row,
};
use crate::c_frontend::rows::{row_indices_by_stride as row_indices, SENTINEL, VAST_STRIDE_U32};
use expression_precedence::*;
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::parse::vast::{
    C_AST_KIND_ASSIGN_EXPR, C_AST_KIND_CONDITIONAL_EXPR, C_EXPR_ASSOC_LEFT, C_EXPR_ASSOC_RIGHT,
    C_EXPR_SHAPE_BINARY, C_EXPR_SHAPE_CONDITIONAL,
};
use vyre_primitives::predicate::node_kind;

#[path = "c_ast_expression_operator_precedence_contracts/precedence_ladder_and_associativity.rs"]
mod precedence_ladder_and_associativity;
