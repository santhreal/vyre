//! Backend parity arm for the advanced C declaration and declarator contracts.
//!
//! The construct list, the four-stage scaffold and the property-graph row
//! contract are owned by `tests/support/c_frontend/parity_matrix` and
//! `tests/support/c_frontend/fixtures/declaration_advanced_constructs`; the CPU
//! arm in `vyre-libs/tests/c_ast_declaration_advanced_contracts` iterates the
//! same `CASES`. What this root adds is the GPU arm those cases run on.
//!
//! Covered constructs:
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
//! A missing GPU adapter is a configuration failure, never a skip.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
mod c_ast_gpu_parity_support;
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
#[path = "../../tests/support/c_frontend/fixtures/declaration_advanced_constructs.rs"]
mod declaration_advanced_constructs;

use c_ast_gpu_parity_support::{assert_family_parity, GpuArm};

#[path = "c_ast_declaration_advanced_contracts/pg_lowering_and_gpu_parity.rs"]
mod pg_lowering_and_gpu_parity;
