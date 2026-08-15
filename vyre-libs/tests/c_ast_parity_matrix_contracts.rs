//! The C-AST parity matrix: its case-list gate, and its CPU-reference arm.
//!
//! Both halves need the same three fixture families and the same shared
//! harness, so they are one target. Splitting them cost a second copy of this
//! module header, which is nine lines of `#[path]` declarations and is exactly
//! the kind of restatement this lane exists to remove.
//!
//! `case_matrix_gate` asserts that every fixture builder is named by its
//! family's parity case table, which is what keeps a construct from being
//! proven on one backend and unproven on another. `cpu_reference_arm` runs the
//! matrix itself on the reference interpreter, so a kernel that disagrees with
//! its oracle fails without a device.
#![cfg(feature = "c-parser")]
#![forbid(unsafe_code)]
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
#[path = "c_ast_parity_matrix_contracts/case_matrix_gate.rs"]
mod case_matrix_gate;
#[path = "c_ast_parity_matrix_contracts/cpu_reference_arm.rs"]
mod cpu_reference_arm;
#[path = "../../tests/support/c_frontend/fixtures/declaration_advanced_constructs.rs"]
mod declaration_advanced_constructs;
#[path = "../../tests/support/c_frontend/fixtures/declarator_matrix_constructs.rs"]
mod declarator_matrix_constructs;
#[path = "../../tests/support/c_frontend/fixtures/semantic_gap_constructs.rs"]
mod semantic_gap_constructs;
