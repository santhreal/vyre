//! One binary for every math-kernels integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Shared fixture module from `tests/wire_words/mod.rs`.
#[cfg(any(feature = "math", feature = "math-kernels"))]
#[path = "wire_words/mod.rs"]
pub mod wire_words;

/// Integration tests from `tests/adversarial_math.rs`.
#[cfg(feature = "math")]
#[path = "adversarial_math.rs"]
pub mod adversarial_math;

/// Integration tests from `tests/prefix_scan_contract.rs`.
#[cfg(feature = "math-kernels")]
#[path = "prefix_scan_contract.rs"]
pub mod prefix_scan_contract;

/// Integration tests from `tests/sweep_math_prefix_scan_exclusive_volume_oracle_matrix.rs`.
#[cfg(feature = "math")]
#[path = "sweep_math_prefix_scan_exclusive_volume_oracle_matrix.rs"]
pub mod sweep_math_prefix_scan_exclusive_volume_oracle_matrix;

/// Integration tests from `tests/sweep_math_prefix_scan_inclusive_volume_oracle_matrix.rs`.
#[cfg(feature = "math")]
#[path = "sweep_math_prefix_scan_inclusive_volume_oracle_matrix.rs"]
pub mod sweep_math_prefix_scan_inclusive_volume_oracle_matrix;
