//! One binary for every default-feature integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Shared fixture module from `tests/target_artifacts/mod.rs`.
#[path = "target_artifacts/mod.rs"]
pub mod target_artifacts;

/// Integration tests from `tests/resident_multi_entry_submission.rs`.
#[path = "resident_multi_entry_submission.rs"]
pub mod resident_multi_entry_submission;

/// Integration tests from `tests/shared_target_contract_discrimination.rs`.
#[path = "shared_target_contract_discrimination.rs"]
pub mod shared_target_contract_discrimination;

/// Integration tests from `tests/target_payload_admission_contract.rs`.
#[path = "target_payload_admission_contract.rs"]
pub mod target_payload_admission_contract;
