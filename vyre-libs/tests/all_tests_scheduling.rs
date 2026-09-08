//! One binary for every scheduling integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Shared fixture module from `tests/bounded_compile_policy.rs`.
#[path = "bounded_compile_policy.rs"]
pub mod bounded_compile_policy;

/// Integration tests from `tests/fusion_scores_via_reference_parity.rs`.
#[path = "fusion_scores_via_reference_parity.rs"]
pub mod fusion_scores_via_reference_parity;

/// Integration tests from `tests/planar_rewrite_via_reference_parity.rs`.
#[path = "planar_rewrite_via_reference_parity.rs"]
pub mod planar_rewrite_via_reference_parity;

/// Integration tests from `tests/shape_spectrum_via_reference_parity.rs`.
#[path = "shape_spectrum_via_reference_parity.rs"]
pub mod shape_spectrum_via_reference_parity;

/// Integration tests from `tests/submodular_retention_via_reference_parity.rs`.
#[path = "submodular_retention_via_reference_parity.rs"]
pub mod submodular_retention_via_reference_parity;
