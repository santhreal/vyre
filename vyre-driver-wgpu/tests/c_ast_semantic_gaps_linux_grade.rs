//! Semantic gaps exposed by kernel-grade C: asm aliases, mixed and incomplete initializers,
//! function pointer typedefs, and attribute-bearing declarations.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
mod c_ast_gpu_parity_support;
#[path = "../../tests/support/c_frontend/fixtures/semantic_gap_constructs.rs"]
mod semantic_gap_constructs;
#[path = "c_ast_semantic_gaps_linux_grade/cpu_pg_and_gpu_parity.rs"]
mod cpu_pg_and_gpu_parity;
