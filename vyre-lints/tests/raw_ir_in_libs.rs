//! End-to-end tests for the `raw_ir_in_libs` lint.
//!
//! Each test writes a synthetic vyre-libs source file to a tempdir,
//! runs the lint, and asserts on the exact violation set.
//!
//! Implementation lives in two chunks under `contract_cases/`:
//! `raw_ir_in_libs__detection_cases.rs` and its child
//! `raw_ir_in_libs__evasion_cases.rs`.

#[path = "contract_cases/raw_ir_in_libs__detection_cases.rs"]
mod raw_ir_in_libs_detection_cases;
