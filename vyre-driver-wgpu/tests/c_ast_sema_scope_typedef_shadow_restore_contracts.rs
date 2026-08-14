//! GPU parity for typedef shadowing and scope restoration in C semantic analysis.
//!
//! The CPU scope-tree and annotation contracts live in
//! `vyre-libs/tests/c_ast_sema_scope_typedef_shadow_restore_contracts`. What
//! stays here is the parity arm: the same disjoint-block, parameter, K&R,
//! for-loop, and nested-chain fixtures dispatched through the WGPU backend and
//! compared against the CPU reference.

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
use vyre_primitives::predicate::node_kind;

#[path = "c_ast_sema_scope_typedef_shadow_restore_contracts/annotation_and_gpu_parity.rs"]
mod annotation_and_gpu_parity;
#[path = "c_ast_sema_scope_typedef_shadow_restore_contracts/gpu_parity.rs"]
mod gpu_parity;
