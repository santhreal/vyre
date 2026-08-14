//! GNU builtin expressions carry no binary or unary operator shape, on CPU and GPU alike.
#![cfg(feature = "c-parser")]
#![allow(clippy::erasing_op)]
#![allow(deprecated)]
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
#[path = "c_ast_expression_operator_builtin_contracts/cpu_builtin_expression_classification.rs"]
mod cpu_builtin_expression_classification;
#[path = "../../tests/support/c_frontend/fixtures/expression_builtin.rs"]
mod expression_builtin;
