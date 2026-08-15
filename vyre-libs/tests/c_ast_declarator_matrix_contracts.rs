//! Integration contracts for Linux-grade C declarator matrices.
//!
//! Coverage:
//!   * pointer-to-array declarators (`int (*p)[4];`)
//!   * storage-class specifiers threaded through multi-declarator lists
//!   * parameter array declarators with `static` / `restrict` (C99)
//!   * nested typedef names inside declarators (function-pointer typedef reuse)
//!   * struct / union / enum tag definitions followed by mixed declarators
//!   * abstract declarators with qualifiers in cast contexts
//!   * GNU `__restrict` normalized to the C restrict qualifier
//!
//! Asserts:
//!   - specifier propagation: standard qualifiers and storage classes stay raw
//!     syntax while declarator identifiers, pointers, arrays and function parens
//!     get precise AST kinds.
//!   - AST classification: POINTER_DECL, ARRAY_DECL, FUNCTION_DECLARATOR,
//!     VARIABLE, FUNCTION_DECL, FIELD_DECL, STRUCT_DECL, UNION_DECL, ENUM_DECL,
//!     ENUMERATOR_DECL.
//!   - typedef annotations: typedef declarations carry TYPEDEF_FLAG_DECL;
//!     typedef uses inside declarator contexts carry TYPEDEF_FLAG_VISIBLE.
//!   - CPU/GPU parity for VAST builder, classifier and PG lowerer, including
//!     stage-specific parity for abstract-declarator casts without typedef names.
//!
//! A missing GPU adapter is a configuration failure, never a silent skip.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
#[path = "../../tests/support/c_frontend/fixtures/declarator_matrix_constructs.rs"]
mod declarator_matrix_constructs;

use crate::c_frontend::rows::{
    flags_at, kind_at, row_indices, TYPEDEF_FLAG_DECL, TYPEDEF_FLAG_VISIBLE,
};
use declarator_matrix_constructs::*;
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::parse::vast::{
    C_AST_KIND_ARRAY_DECL,
    C_AST_KIND_CAST_EXPR, C_AST_KIND_ENUMERATOR_DECL, C_AST_KIND_ENUM_DECL, C_AST_KIND_FIELD_DECL,
    C_AST_KIND_FUNCTION_DECLARATOR, C_AST_KIND_POINTER_DECL, C_AST_KIND_STRUCT_DECL,
    C_AST_KIND_TYPEDEF_DECL, C_AST_KIND_UNION_DECL,
};
use vyre_primitives::predicate::node_kind;

#[path = "c_ast_declarator_matrix_contracts/cpu_reference_and_pg_lowering.rs"]
mod cpu_reference_and_pg_lowering;
