//! Postfix increment and decrement are neither unary nor binary operator shapes, on CPU and GPU
//! alike.
#![cfg(feature = "c-parser")]
#![allow(clippy::erasing_op)]
#![allow(deprecated)]
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
mod c_ast_gpu_parity_support;
#[path = "../../tests/support/c_frontend/fixtures/expression_postfix.rs"]
mod expression_postfix;
#[path = "c_ast_expression_operator_postfix_contracts/postfix_expression_shapes.rs"]
mod postfix_expression_shapes;
