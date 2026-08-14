//! GPU parity for C function parameter scope and prototype scope.
//!
//! The CPU contracts for parameter scopes and shadowing live in
//! `vyre-libs/tests/c_ast_sema_scope_function_parameter_prototype_contracts`.
//! What stays here is the typedef-restore parity arm: the same fixtures
//! dispatched through the WGPU backend and compared against the CPU reference.

#![cfg(feature = "c-parser")]
#![allow(deprecated)]
mod c_ast_gpu_parity_support;
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;

use c_ast_gpu_parity_support::scope_fixture::*;
use c_ast_gpu_parity_support::scope_gpu_support::{
    run_gpu_annotate, run_gpu_classify, run_gpu_scope_tree,
};
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::parse::vast::{
    reference_c11_classify_vast_node_kinds, C_AST_KIND_POINTER_DECL,
};
use vyre_libs::parsing::c::sema::lookup::DECL_KIND_FUNCTION;

#[path = "c_ast_sema_scope_function_parameter_prototype_contracts/typedef_restore_and_gpu_parity.rs"]
mod typedef_restore_and_gpu_parity;
