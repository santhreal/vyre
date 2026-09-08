//! One binary for every default-feature integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Shared fixture module from `tests/dispatch_fixtures/mod.rs`.
#[path = "dispatch_fixtures/mod.rs"]
pub mod dispatch_fixtures;

/// Integration tests from `tests/backend_registration.rs`.
#[path = "backend_registration.rs"]
pub mod backend_registration;

/// Integration tests from `tests/generated_boundary_matrix.rs`.
#[path = "generated_boundary_matrix.rs"]
pub mod generated_boundary_matrix;

/// Integration tests from `tests/hostile_input_closure_contract.rs`.
#[path = "hostile_input_closure_contract.rs"]
pub mod hostile_input_closure_contract;

/// Integration tests from `tests/parity_suite.rs`.
#[path = "parity_suite.rs"]
pub mod parity_suite;

/// Integration tests from `tests/semantic_execution.rs`.
#[path = "semantic_execution.rs"]
pub mod semantic_execution;
