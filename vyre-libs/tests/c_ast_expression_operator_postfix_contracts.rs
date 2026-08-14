//! Postfix increment and decrement are neither unary nor binary operator shapes, on CPU and GPU
//! alike.
#![cfg(feature = "c-parser")]
#![allow(clippy::erasing_op)]
#![allow(deprecated)]
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
#[path = "../../tests/support/c_frontend/fixtures/expression_postfix.rs"]
mod expression_postfix;
#[path = "c_ast_expression_operator_postfix_contracts/cpu_postfix_and_unary_classification.rs"]
mod cpu_postfix_and_unary_classification;
