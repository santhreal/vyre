//! Contract tests for c ast pg lowering deep contracts.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
mod c_ast_gpu_parity_support;
#[path = "c_ast_pg_lowering_deep_contracts/classify.rs"]
mod classify;
#[path = "c_ast_pg_lowering_deep_contracts/function_definition_has_declaration_category.rs"]
mod function_definition_has_declaration_category;
