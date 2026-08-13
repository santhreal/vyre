//! Semantic gaps exposed by kernel-grade C: asm aliases, mixed and incomplete initializers,
//! function pointer typedefs, and attribute-bearing declarations.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
mod c_ast_gpu_parity_support;
#[path = "c_ast_semantic_gaps_linux_grade/cpu_asm_alias_classifies.rs"]
mod cpu_asm_alias_classifies;
#[path = "c_ast_semantic_gaps_linux_grade/word_at.rs"]
mod word_at;
