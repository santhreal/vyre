//! One binary for every default-feature integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Integration tests from `tests/consumer_boundary.rs`.
#[path = "consumer_boundary.rs"]
pub mod consumer_boundary;

/// Integration tests from `tests/feature_organization.rs`.
#[path = "feature_organization.rs"]
pub mod feature_organization;

/// Integration tests from `tests/hardware_registration_safety_rules.rs`.
#[cfg(feature = "hardware")]
#[path = "hardware_registration_safety_rules.rs"]
pub mod hardware_registration_safety_rules;

/// Integration tests from `tests/proptest_wire_roundtrip.rs`.
#[path = "proptest_wire_roundtrip.rs"]
pub mod proptest_wire_roundtrip;

/// Integration tests from `tests/registry_closure.rs`.
#[path = "registry_closure.rs"]
pub mod registry_closure;

/// Integration tests from `tests/registry_oob_clean.rs`.
#[cfg(feature = "inventory-registry")]
#[path = "registry_oob_clean.rs"]
pub mod registry_oob_clean;

/// Integration tests from `tests/wire_differential_std_io.rs`.
#[path = "wire_differential_std_io.rs"]
pub mod wire_differential_std_io;

/// Integration tests from `tests/wire_harness_smoke_test.rs`.
#[path = "wire_harness_smoke_test.rs"]
pub mod wire_harness_smoke_test;

/// Integration tests from `tests/wire_pack_into_contracts.rs`.
#[path = "wire_pack_into_contracts.rs"]
pub mod wire_pack_into_contracts;
