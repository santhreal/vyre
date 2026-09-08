//! One binary for every default-feature integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Integration tests from `tests/cert_contracts.rs`.
#[path = "cert_contracts.rs"]
pub mod cert_contracts;

/// Integration tests from `tests/schema_contract.rs`.
#[path = "schema_contract.rs"]
pub mod schema_contract;

/// Integration tests from `tests/witness_contract.rs`.
#[path = "witness_contract.rs"]
pub mod witness_contract;
