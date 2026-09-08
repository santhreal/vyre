//! One binary for every reduce, telemetry integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Shared fixture module from `tests/bounded_compile_policy.rs`.
#[path = "bounded_compile_policy.rs"]
pub mod bounded_compile_policy;

/// Integration tests from `tests/reduction_metrics_via_reference_parity.rs`.
#[path = "reduction_metrics_via_reference_parity.rs"]
pub mod reduction_metrics_via_reference_parity;
