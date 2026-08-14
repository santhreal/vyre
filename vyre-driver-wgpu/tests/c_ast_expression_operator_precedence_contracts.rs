//! Contracts for C expression-operator precedence and associativity.
//!
//! Covers every precedence band that participates in expression-shape rows,
//! including shift, relational, equality, compound assignment, ternary
//! conditional, and comma boundaries.  Each fixture asserts the exact root
//! operator and operand links expected from a full precedence-climbing parser.
//! GPU/CPU parity and PG lowering preservation are required for all fixtures.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
mod c_ast_gpu_parity_support;
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
#[path = "../../tests/support/c_frontend/fixtures/expression_precedence.rs"]
mod expression_precedence;

use c_ast_gpu_parity_support::expression_pipeline::run_reference_pg_lower;
use c_ast_gpu_parity_support::{bytes, run_gpu_expr_shape, run_gpu_pg_lower, starts_for_lens};
use expression_precedence::*;
use vyre_libs::parsing::c::parse::vast::{
    reference_c11_build_expression_shape_nodes, reference_c11_build_vast_nodes,
    reference_c11_classify_vast_node_kinds,
};

#[path = "c_ast_expression_operator_precedence_contracts/gpu_parity.rs"]
mod gpu_parity;
