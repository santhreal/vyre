//! Typedef name resolution in the C11 parser: cast versus expression, shadowing, struct tags, and
//! declarator contexts, on CPU and GPU.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
mod c_ast_gpu_parity_support;
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
#[path = "c11_parser_typedef_contracts/pg_lowering_and_gpu_parity.rs"]
mod pg_lowering_and_gpu_parity;
#[path = "../../tests/support/c_frontend/fixtures/typedef_disambiguation.rs"]
mod typedef_disambiguation;
