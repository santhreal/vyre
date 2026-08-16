//! C typedef shadowing and scope restoration contracts.
//!
//! A typedef name shadowed by an inner declaration must become visible again
//! when that scope closes, and the classifier must follow the visibility it is
//! given. These contracts pin that against the CPU reference passes; the WGPU
//! parity arm over the same fixtures lives in `vyre-driver-wgpu/tests`.

#![cfg(feature = "c-parser")]
#![allow(deprecated)]
use vyre_libs::predicate::node_kind;

#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;

use c_frontend::scope_fixture::*;
use vyre_libs::parsing::c::parse::vast::{C_AST_KIND_CAST_EXPR, C_AST_KIND_UNARY_EXPR};
use vyre_libs::parsing::c::sema::lookup::{DECL_KIND_NONE, DECL_KIND_TYPEDEF, DECL_KIND_VARIABLE};

#[path = "c_ast_sema_scope_typedef_shadow_restore_contracts/scope_tree_and_annotation.rs"]
mod scope_tree_and_annotation;
