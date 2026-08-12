//! Generated wrapper test crate for c ast expression operator ambiguity contracts.
//!
//! Implementation lives in `contract_cases/` chunks.
#![cfg(feature = "c-parser")]
#![allow(clippy::erasing_op)]
#![allow(deprecated)]
include!("contract_cases/c_ast_expression_operator_ambiguity_contracts__bytes.rs");
include!("contract_cases/c_ast_expression_operator_ambiguity_contracts__plus_binary_is_binary_and_unary_is_unary.rs");
