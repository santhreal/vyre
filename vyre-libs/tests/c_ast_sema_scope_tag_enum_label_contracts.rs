//! C tag, enum-constant, and label namespace contracts.
//!
//! Struct/union/enum tags, goto labels, and ordinary identifiers occupy
//! separate namespaces, so a tag never shadows a variable of the same name.
//! These contracts pin that against the CPU reference passes; the WGPU parity
//! arm over the same fixtures lives in `vyre-driver-wgpu/tests`.

#![cfg(feature = "c-parser")]
#![allow(deprecated)]
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;

use c_frontend::scope_fixture::*;
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::parse::vast::C_AST_KIND_CAST_EXPR;
use vyre_libs::parsing::c::sema::lookup::{
    DECL_KIND_ENUM_CONSTANT, DECL_KIND_LABEL, DECL_KIND_NONE, DECL_KIND_TYPEDEF, DECL_KIND_VARIABLE,
};

#[path = "c_ast_sema_scope_tag_enum_label_contracts/tag_enum_and_label_namespaces.rs"]
mod tag_enum_and_label_namespaces;
