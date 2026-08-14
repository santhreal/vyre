//! Operators that are both unary and binary, and the cast versus parenthesized expression
//! classification that separates them.
#![cfg(feature = "c-parser")]
#![allow(clippy::erasing_op)]
#![allow(deprecated)]
#[path = "c_ast_expression_operator_ambiguity_contracts/bytes.rs"]
mod bytes;
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
mod c_ast_gpu_parity_support;
#[path = "c_ast_expression_operator_ambiguity_contracts/unary_binary_and_cast_classification.rs"]
mod unary_binary_and_cast_classification;
