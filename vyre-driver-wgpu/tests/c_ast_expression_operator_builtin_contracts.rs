//! GNU builtin expressions carry no binary or unary operator shape, on CPU and GPU alike.
#![cfg(feature = "c-parser")]
#![allow(clippy::erasing_op)]
#![allow(deprecated)]
#[path = "c_ast_expression_operator_builtin_contracts/builtin_expression_shapes.rs"]
mod builtin_expression_shapes;
mod c_ast_gpu_parity_support;
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
#[path = "../../tests/support/c_frontend/fixtures/expression_builtin.rs"]
mod expression_builtin;
