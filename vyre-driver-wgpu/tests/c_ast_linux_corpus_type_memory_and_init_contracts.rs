//! Contract tests for c ast linux corpus type memory and init contracts.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
mod c_ast_gpu_parity_support;
#[path = "c_ast_linux_corpus_type_memory_and_init_contracts/function_pointer_table_gpu_pg_lower_matches_cpu.rs"]
mod function_pointer_table_gpu_pg_lower_matches_cpu;
#[path = "c_ast_linux_corpus_type_memory_and_init_contracts/kind_at.rs"]
mod kind_at;
