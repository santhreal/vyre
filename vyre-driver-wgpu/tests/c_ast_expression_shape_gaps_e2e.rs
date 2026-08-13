//! End-to-end C parser coverage for expression-shape gaps.

#![cfg(feature = "c-parser")]
#[path = "c_ast_expression_shape_gaps_e2e/fixtures.rs"]
mod fixtures;
#[allow(deprecated)]
#[path = "c_ast_expression_shape_gaps_e2e/gpu_parity.rs"]
mod gpu_parity;
#[path = "c_ast_expression_shape_gaps_e2e/kind_shape.rs"]
mod kind_shape;
#[allow(deprecated)]
#[path = "c_ast_expression_shape_gaps_e2e/support.rs"]
mod support;
