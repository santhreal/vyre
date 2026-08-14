//! End-to-end C parser coverage for expression-shape gaps.

#![cfg(feature = "c-parser")]
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
#[path = "../../tests/support/c_frontend/fixtures/expression_shape_gap_constructs.rs"]
mod expression_shape_gap_constructs;
#[allow(deprecated)]
#[path = "c_ast_expression_shape_gaps_e2e/gpu_parity.rs"]
mod gpu_parity;
#[allow(deprecated)]
#[path = "c_ast_expression_shape_gaps_e2e/support.rs"]
mod support;
