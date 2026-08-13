//! Typedef name resolution in the C11 parser: cast versus expression, shadowing, struct tags, and
//! declarator contexts, on CPU and GPU.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
#[path = "c11_parser_typedef_contracts/assert_kind.rs"]
mod assert_kind;
mod c_ast_gpu_parity_support;
#[path = "c11_parser_typedef_contracts/pg_lowering_and_gpu_parity.rs"]
mod pg_lowering_and_gpu_parity;
