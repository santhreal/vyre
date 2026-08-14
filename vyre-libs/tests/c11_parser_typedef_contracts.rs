//! Typedef name resolution in the C11 parser: cast versus expression, shadowing, struct tags, and
//! declarator contexts, on CPU and GPU.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
#[path = "../../tests/support/c_frontend/fixtures/typedef_disambiguation.rs"]
mod typedef_disambiguation;
#[path = "c11_parser_typedef_contracts/cpu_typedef_disambiguation.rs"]
mod cpu_typedef_disambiguation;
