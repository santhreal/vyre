//! Contract tests for c ast switch case complex body pg lowering contracts.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
mod c_ast_gpu_parity_support;
#[path = "c_ast_switch_case_complex_body_pg_lowering_contracts/classify.rs"]
mod classify;
#[path = "c_ast_switch_case_complex_body_pg_lowering_contracts/cpu_switch_case_with_compound_literal_classifies.rs"]
mod cpu_switch_case_with_compound_literal_classifies;
