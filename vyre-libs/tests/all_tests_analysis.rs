//! One binary for every analysis integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Shared fixture module from `tests/bounded_compile_policy.rs`.
#[path = "bounded_compile_policy.rs"]
pub mod bounded_compile_policy;

/// Integration tests from `tests/cost_model_predict_runtime_via_reference_parity.rs`.
#[path = "cost_model_predict_runtime_via_reference_parity.rs"]
pub mod cost_model_predict_runtime_via_reference_parity;

/// Integration tests from `tests/semiring_gemm_via_reference_parity.rs`.
#[path = "semiring_gemm_via_reference_parity.rs"]
pub mod semiring_gemm_via_reference_parity;
