//! Deeply nested C scopes, checked CPU against GPU across the whole pipeline.
//!
//! These tests exercise the full C semantic pipeline:
//!   * c_sema_scope (scope-tree extraction)
//!   * c11_annotate_typedef_names (typedef visibility pass)
//!   * c11_classify_vast_node_kinds (type-aware classifier)
//!   * c11_build_vast_nodes (structural VAST builder)
//!   * c_lower_ast_to_pg_nodes (VAST -> PG lowering)
//!
//! Deep nesting targets the GPU walk limits and ensures scope_parent_id
//! chains remain correct beyond shallow ancestors.

#![cfg(feature = "c-parser")]
#![allow(deprecated)]
mod c_ast_gpu_parity_support;
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;

use c_ast_gpu_parity_support::scope_fixture::*;
use c_ast_gpu_parity_support::scope_gpu_support::{
    run_gpu_annotate, run_gpu_classify, run_gpu_pg_lower, run_gpu_scope_tree, run_gpu_vast_builder,
};
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::parse::vast::{
    reference_c11_classify_vast_node_kinds, C_AST_KIND_POINTER_DECL,
};
use vyre_libs::parsing::c::sema::lookup::{DECL_KIND_NONE, DECL_KIND_VARIABLE};

#[path = "c_ast_sema_scope_deep_nesting_cpu_gpu_parity_contracts/deep_block_nesting.rs"]
mod deep_block_nesting;
#[path = "c_ast_sema_scope_deep_nesting_cpu_gpu_parity_contracts/gpu_parity.rs"]
mod gpu_parity;
