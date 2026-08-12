//! CPU-only reference tests for extended GNU asm decomposition and
//! GNU attribute-specific AST kinds.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
include!("contract_cases/c_ast_gnu_asm_decomposition_and_attribute_kinds__new.rs");
include!("contract_cases/c_ast_gnu_asm_decomposition_and_attribute_kinds__cpu_reference_classifies_attribute_naked.rs");
