//! Property-graph lowering of GNU computed goto, for loops with declarations, and the C11 _Atomic
//! qualifier.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
mod c_ast_gpu_parity_support;
#[path = "c_ast_gnu_computed_goto_and_c11_atomic_contracts/classify.rs"]
mod classify;
#[path = "c_ast_gnu_computed_goto_and_c11_atomic_contracts/pg_lowering_and_gpu_parity.rs"]
mod pg_lowering_and_gpu_parity;
