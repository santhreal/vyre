//! One binary for every default-feature integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Shared fixture module from `tests/workspace_sources/mod.rs`.
#[path = "workspace_sources/mod.rs"]
pub mod workspace_sources;

/// Integration tests from `tests/docs_references.rs`.
#[path = "docs_references.rs"]
pub mod docs_references;

/// Integration tests from `tests/lint_policy_mutations.rs`.
#[path = "lint_policy_mutations.rs"]
pub mod lint_policy_mutations;

/// Integration tests from `tests/release_docs.rs`.
#[path = "release_docs.rs"]
pub mod release_docs;

/// Integration tests from `tests/gate_verdict.rs`.
#[path = "gate_verdict.rs"]
pub mod gate_verdict;

/// Integration tests from `tests/device_test_compilation.rs`.
#[path = "device_test_compilation.rs"]
pub mod device_test_compilation;
