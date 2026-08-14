//! Operators that are both unary and binary, and the cast versus parenthesized expression
//! classification that separates them.
#![cfg(feature = "c-parser")]
#![allow(clippy::erasing_op)]
#![allow(deprecated)]
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
#[path = "../../tests/support/c_frontend/fixtures/expression_ambiguity.rs"]
mod expression_ambiguity;
#[path = "c_ast_expression_operator_ambiguity_contracts/cpu_unary_binary_ambiguity.rs"]
mod cpu_unary_binary_ambiguity;
