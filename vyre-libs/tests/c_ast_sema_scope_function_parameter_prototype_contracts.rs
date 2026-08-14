//! C function parameter scope and prototype scope contracts.
//!
//! A parameter belongs to the function body scope, a prototype parameter does
//! not escape the prototype, and either can shadow a typedef. These contracts
//! pin that against the CPU reference passes; the WGPU parity arm over the same
//! fixtures lives in `vyre-driver-wgpu/tests`.

#![cfg(feature = "c-parser")]
#![allow(deprecated)]
use vyre_primitives::predicate::node_kind;

#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;

use c_frontend::scope_fixture::*;
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::parse::vast::C_AST_KIND_CAST_EXPR;
use vyre_libs::parsing::c::sema::lookup::{DECL_KIND_NONE, DECL_KIND_VARIABLE};

#[path = "c_ast_sema_scope_function_parameter_prototype_contracts/parameter_scopes_and_shadowing.rs"]
mod parameter_scopes_and_shadowing;
