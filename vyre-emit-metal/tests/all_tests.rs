//! One binary for every default-feature integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Integration tests from `tests/adversarial_emit_program_matrix.rs`.
#[path = "adversarial_emit_program_matrix.rs"]
pub mod adversarial_emit_program_matrix;

/// Integration tests from `tests/emit_contracts.rs`.
#[path = "emit_contracts.rs"]
pub mod emit_contracts;
