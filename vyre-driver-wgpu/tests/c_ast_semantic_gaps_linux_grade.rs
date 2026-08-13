//! Contract tests for c ast semantic gaps linux grade.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
mod c_ast_gpu_parity_support;
#[path = "c_ast_semantic_gaps_linux_grade/cpu_asm_alias_classifies.rs"]
mod cpu_asm_alias_classifies;
#[path = "c_ast_semantic_gaps_linux_grade/word_at.rs"]
mod word_at;
