//! Tag separation, GNU attributes, and compound literals in the C AST, with GPU parity against the
//! CPU reference.
#![cfg(feature = "c-parser")]
#![allow(clippy::erasing_op)]
#![allow(deprecated)]
mod c_ast_gpu_parity_support;
#[path = "gemini_c_ast_contracts/cpu_reference_and_gpu_parity.rs"]
mod cpu_reference_and_gpu_parity;
#[path = "gemini_c_ast_contracts/tok.rs"]
mod tok;
