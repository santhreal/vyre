//! One binary for every default-feature integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Shared fixture module from `tests/fixture_target/mod.rs`.
#[allow(dead_code, unreachable_pub)]
#[path = "fixture_target/mod.rs"]
pub mod fixture_target;

/// Integration tests from `tests/artifact_contracts.rs`.
#[path = "artifact_contracts.rs"]
pub mod artifact_contracts;

/// Integration tests from `tests/bundle_contracts.rs`.
#[path = "bundle_contracts.rs"]
pub mod bundle_contracts;

/// Integration tests from `tests/cache_contracts.rs`.
#[path = "cache_contracts.rs"]
pub mod cache_contracts;

/// Integration tests from `tests/canonical_package.rs`.
#[path = "canonical_package.rs"]
pub mod canonical_package;

/// Integration tests from `tests/compile_smoke.rs`.
#[path = "compile_smoke.rs"]
pub mod compile_smoke;

/// Integration tests from `tests/generated_artifact_manifest_matrix.rs`.
#[path = "generated_artifact_manifest_matrix.rs"]
pub mod generated_artifact_manifest_matrix;

/// Integration tests from `tests/generated_loader_contracts.rs`.
#[allow(dead_code, unreachable_pub)]
#[path = "generated_loader_contracts.rs"]
pub mod generated_loader_contracts;

/// Integration tests from `tests/launcher_contracts.rs`.
#[path = "launcher_contracts.rs"]
pub mod launcher_contracts;

/// Integration tests from `tests/launcher_registry_closure_contracts.rs`.
#[path = "launcher_registry_closure_contracts.rs"]
pub mod launcher_registry_closure_contracts;

/// Integration tests from `tests/manifest_round_trip.rs`.
#[path = "manifest_round_trip.rs"]
pub mod manifest_round_trip;

/// Integration tests from `tests/workspace_error_diagnostic_schema.rs`.
#[path = "workspace_error_diagnostic_schema.rs"]
pub mod workspace_error_diagnostic_schema;
