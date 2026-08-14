//! Operators that are both unary and binary, and the cast versus parenthesized expression
//! classification that separates them.
#![cfg(feature = "c-parser")]
#![allow(clippy::erasing_op)]
#![allow(deprecated)]
mod c_ast_gpu_parity_support;
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
#[path = "../../tests/support/c_frontend/fixtures/expression_ambiguity.rs"]
mod expression_ambiguity;
#[path = "c_ast_expression_operator_ambiguity_contracts/unary_binary_and_cast_classification.rs"]
mod unary_binary_and_cast_classification;
