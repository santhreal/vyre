//! One binary for every hardware integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Integration tests from `tests/generated_hardware_f32_matrix.rs`.
#[path = "generated_hardware_f32_matrix.rs"]
pub mod generated_hardware_f32_matrix;

/// Integration tests from `tests/generated_hardware_registry_matrix.rs`.
#[path = "generated_hardware_registry_matrix.rs"]
pub mod generated_hardware_registry_matrix;

/// Integration tests from `tests/generated_hardware_u32_matrix.rs`.
#[path = "generated_hardware_u32_matrix.rs"]
pub mod generated_hardware_u32_matrix;

/// Integration tests from `tests/hardware_conform.rs`.
#[path = "hardware_conform.rs"]
pub mod hardware_conform;

/// Integration tests from `tests/hardware_registry_contract.rs`.
#[path = "hardware_registry_contract.rs"]
pub mod hardware_registry_contract;

/// Integration tests from `tests/integration.rs`.
#[path = "integration.rs"]
pub mod integration;
