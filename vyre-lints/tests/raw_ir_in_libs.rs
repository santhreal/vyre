//! End-to-end tests for the `raw_ir_in_libs` lint.
//!
//! Each test writes a synthetic vyre-libs source file to a tempdir,
//! runs the lint, and asserts on the exact violation set.
//!
//! Implementation lives in two `include!`-d chunks under `contract_cases/`.

include!("contract_cases/raw_ir_in_libs__write_lib_file.rs");
include!("contract_cases/raw_ir_in_libs__adversarial_module_named_tests_inside_a_real_module.rs");
