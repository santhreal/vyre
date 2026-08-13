//! Contract tests for c ast expression operator postfix contracts.
#![cfg(feature = "c-parser")]
#![allow(clippy::erasing_op)]
#![allow(deprecated)]
#[path = "c_ast_expression_operator_postfix_contracts/bytes.rs"]
mod bytes;
mod c_ast_gpu_parity_support;
#[path = "c_ast_expression_operator_postfix_contracts/postfix_inc_dec_are_not_unary_and_not_binary.rs"]
mod postfix_inc_dec_are_not_unary_and_not_binary;
