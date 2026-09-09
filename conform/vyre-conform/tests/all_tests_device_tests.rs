//! One binary for every device-tests integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Integration tests from `tests/countless_readwrite_output_parity/mod.rs`.
#[path = "countless_readwrite_output_parity/mod.rs"]
pub mod countless_readwrite_output_parity;

/// Integration tests from `tests/lens_parity_device.rs`.
#[path = "lens_parity_device.rs"]
pub mod lens_parity_device;

/// Integration tests from `tests/parity_matrix.rs`.
#[path = "parity_matrix.rs"]
pub mod parity_matrix;

/// Integration tests from `tests/production_route_device.rs`.
#[path = "production_route_device.rs"]
pub mod production_route_device;

/// Integration tests from `tests/reference_parity_classes/mod.rs`.
#[path = "reference_parity_classes/mod.rs"]
pub mod reference_parity_classes;

/// Integration tests from `tests/ulp_audit.rs`.
#[path = "ulp_audit.rs"]
pub mod ulp_audit;

#[test]
fn vyre_conform_worker_entry() {
    vyre_conform::run_worker_from_env_or_exit();
}
