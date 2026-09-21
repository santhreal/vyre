//! One binary for every device-tests integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Integration tests from `tests/dispatch.rs`.
#[cfg(feature = "device-tests")]
#[path = "dispatch.rs"]
pub mod dispatch;

/// Integration tests from `tests/hostile_input_closure_contract.rs`.
#[cfg(feature = "device-tests")]
#[path = "hostile_input_closure_contract.rs"]
pub mod hostile_input_closure_contract;
