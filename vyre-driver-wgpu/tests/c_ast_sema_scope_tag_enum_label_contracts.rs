//! GPU parity for the C tag namespace, enum constants, and label namespace.
//!
//! The CPU namespace contracts live in
//! `vyre-libs/tests/c_ast_sema_scope_tag_enum_label_contracts`. What stays here
//! is the annotation parity arm: the same fixtures dispatched through the WGPU
//! backend and compared against the CPU reference.

#![cfg(feature = "c-parser")]
#![allow(deprecated)]
mod c_ast_gpu_parity_support;
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;

use c_ast_gpu_parity_support::scope_fixture::*;
use c_ast_gpu_parity_support::scope_gpu_support::{
    run_gpu_annotate, run_gpu_classify, run_gpu_scope_tree,
};
use vyre_libs::parsing::c::parse::vast::{
    reference_c11_classify_vast_node_kinds, C_AST_KIND_CAST_EXPR, C_AST_KIND_POINTER_DECL,
};

#[path = "c_ast_sema_scope_tag_enum_label_contracts/annotation_and_gpu_parity.rs"]
mod annotation_and_gpu_parity;
