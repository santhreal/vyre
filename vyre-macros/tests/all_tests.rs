//! One binary for every default-feature integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

extern crate self as vyre;
extern crate self as vyre_foundation;

/// Shared fixture module from `tests/expansion_fixtures/mod.rs`.
#[macro_use]
#[allow(missing_docs)]
#[path = "expansion_fixtures/mod.rs"]
pub mod expansion_fixtures;

pub use expansion_fixtures::{geometry, ir, numeric, operation, optimizer};

/// Integration tests from `tests/adversarial.rs`.
#[allow(missing_docs)]
#[path = "adversarial.rs"]
pub mod adversarial;

/// Integration tests from `tests/ast_registry_contracts.rs`.
#[allow(missing_docs)]
#[path = "ast_registry_contracts.rs"]
pub mod ast_registry_contracts;

/// Integration tests from `tests/generated_ast_registry_matrix.rs`.
#[allow(missing_docs)]
#[path = "generated_ast_registry_matrix.rs"]
pub mod generated_ast_registry_matrix;

/// Integration tests from `tests/generated_metadata_matrix.rs`.
#[allow(missing_docs)]
#[path = "generated_metadata_matrix.rs"]
pub mod generated_metadata_matrix;

/// Integration tests from `tests/integration.rs`.
#[allow(missing_docs)]
#[path = "integration.rs"]
pub mod integration;

/// Integration tests from `tests/pass_matrix.rs`.
#[allow(missing_docs)]
#[path = "pass_matrix.rs"]
pub mod pass_matrix;

/// Integration tests from `tests/release_surface_contracts.rs`.
#[allow(missing_docs)]
#[path = "release_surface_contracts.rs"]
pub mod release_surface_contracts;
