//! C scope semantics for cast contexts, declarator shapes, and redeclarations.
//!
//! The type-aware classifier decides `(T)*x` by whether `T` is a visible
//! typedef, and the scope tree decides visibility. These contracts pin both
//! against the CPU reference passes. The WGPU parity arm over the same
//! fixtures lives in `vyre-driver-wgpu/tests`.

#![cfg(feature = "c-parser")]
#![allow(deprecated)]
use vyre_primitives::predicate::node_kind;

#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;

use c_frontend::scope_fixture::*;
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::parse::vast::{
    C_AST_KIND_ARRAY_DECL, C_AST_KIND_CAST_EXPR, C_AST_KIND_FUNCTION_DECLARATOR,
    C_AST_KIND_POINTER_DECL,
};
use vyre_libs::parsing::c::sema::lookup::{
    DECL_KIND_FUNCTION, DECL_KIND_FUNCTION_DECL, DECL_KIND_VARIABLE,
};

#[path = "c_ast_sema_scope_cast_decl_redecl_field_contracts/classifier_and_redeclaration_scopes.rs"]
mod classifier_and_redeclaration_scopes;
