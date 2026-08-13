//! Operators that are both unary and binary, and the cast versus parenthesized expression
//! classification that separates them.
#![cfg(feature = "c-parser")]
#![allow(clippy::erasing_op)]
#![allow(deprecated)]
#[path = "c_ast_expression_operator_ambiguity_contracts/bytes.rs"]
mod bytes;
mod c_ast_gpu_parity_support;
#[path = "c_ast_expression_operator_ambiguity_contracts/plus_binary_is_binary_and_unary_is_unary.rs"]
mod plus_binary_is_binary_and_unary_is_unary;
