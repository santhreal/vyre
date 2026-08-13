//! Contract tests for c ast expression operator builtin contracts.
#![cfg(feature = "c-parser")]
#![allow(clippy::erasing_op)]
#![allow(deprecated)]
#[path = "c_ast_expression_operator_builtin_contracts/builtin_shapes_are_none_not_binary.rs"]
mod builtin_shapes_are_none_not_binary;
#[path = "c_ast_expression_operator_builtin_contracts/bytes.rs"]
mod bytes;
mod c_ast_gpu_parity_support;
