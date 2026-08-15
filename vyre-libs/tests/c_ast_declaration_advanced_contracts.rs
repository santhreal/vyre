//! Advanced C declaration and declarator contracts.
//!
//! Coverage gaps filled:
//!   * deeply nested struct / union / enum definitions
//!   * anonymous struct/union members
//!   * typedefs with multiple complex declarators (struct tag + pointer)
//!   * triple-star pointers with interleaved qualifiers
//!   * storage-class combinations: _Thread_local, _Atomic, register, inline
//!   * bit-fields inside nested structs
//!   * GNU attributes on struct fields and typedef declarations
//!   * pointer-to-function-pointer declarators
//!   * arrays of function pointers with qualified parameters
//!
//! Every test asserts CPU/GPU parity and meaningful AST/VAST/PG invariants.
//! A missing GPU adapter is a configuration failure, never a skip.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
#[path = "../../tests/support/c_frontend/fixtures/declaration_advanced_constructs.rs"]
mod declaration_advanced_constructs;

use crate::c_frontend::rows::{flags_at, kind_at, row_indices, TYPEDEF_FLAG_DECL};
use declaration_advanced_constructs::*;
use vyre_libs::parsing::c::parse::vast::{
    C_AST_KIND_ARRAY_DECL,
    C_AST_KIND_BIT_FIELD_DECL, C_AST_KIND_ENUMERATOR_DECL, C_AST_KIND_ENUM_DECL,
    C_AST_KIND_FIELD_DECL, C_AST_KIND_FUNCTION_DECLARATOR, C_AST_KIND_GNU_ATTRIBUTE,
    C_AST_KIND_POINTER_DECL, C_AST_KIND_STRUCT_DECL, C_AST_KIND_TYPEDEF_DECL,
    C_AST_KIND_UNION_DECL,
};
use vyre_primitives::predicate::node_kind;

#[path = "c_ast_declaration_advanced_contracts/cpu_reference_and_pg_lowering.rs"]
mod cpu_reference_and_pg_lowering;
