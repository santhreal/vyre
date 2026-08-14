//! Type, memory, and initializer constructs taken from a Linux corpus, each checked for
//! classification and GPU lowering parity.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
#[path = "c_ast_linux_corpus_type_memory_and_init_contracts/aggregate_types_and_initializers.rs"]
mod aggregate_types_and_initializers;
mod c_ast_gpu_parity_support;
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
#[path = "c_ast_linux_corpus_type_memory_and_init_contracts/kind_at.rs"]
mod kind_at;
