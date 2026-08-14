//! GPU parity for C cast/declaration contexts, redeclarations, and struct fields.
//!
//! The CPU classifier and redeclaration-scope contracts these fixtures encode
//! live in `vyre-libs/tests/c_ast_sema_scope_cast_decl_redecl_field_contracts`,
//! beside the passes they exercise. What stays here is the parity arm: the same
//! fixtures dispatched through the WGPU backend and compared against those CPU
//! references.

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
    reference_c11_classify_vast_node_kinds, C_AST_KIND_CAST_EXPR,
};
use vyre_libs::parsing::c::sema::lookup::{DECL_KIND_NONE, DECL_KIND_VARIABLE};

#[path = "c_ast_sema_scope_cast_decl_redecl_field_contracts/struct_fields_and_gpu_parity.rs"]
mod struct_fields_and_gpu_parity;
