//! One binary for every reduce integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Shared fixture module from `tests/gate_fixtures/mod.rs`.
#[macro_use]
#[path = "gate_fixtures/mod.rs"]
pub mod gate_fixtures;

/// Shared fixture module from `tests/wire_words/mod.rs`.
#[path = "wire_words/mod.rs"]
pub mod wire_words;

/// Integration tests from `tests/adversarial_reduce_gather.rs`.
#[path = "adversarial_reduce_gather.rs"]
pub mod adversarial_reduce_gather;

/// Integration tests from `tests/adversarial_reduce_histogram.rs`.
#[path = "adversarial_reduce_histogram.rs"]
pub mod adversarial_reduce_histogram;

/// Integration tests from `tests/adversarial_reduce_radix_sort.rs`.
#[path = "adversarial_reduce_radix_sort.rs"]
pub mod adversarial_reduce_radix_sort;

/// Integration tests from `tests/adversarial_reduce_scatter.rs`.
#[path = "adversarial_reduce_scatter.rs"]
pub mod adversarial_reduce_scatter;

/// Integration tests from `tests/adversarial_reduce_segment_reduce.rs`.
#[allow(clippy::identity_op, dead_code, unused_imports)]
#[path = "adversarial_reduce_segment_reduce.rs"]
pub mod adversarial_reduce_segment_reduce;

/// Integration tests from `tests/sweep_radix_sort_oracle_matrix.rs`.
#[path = "sweep_radix_sort_oracle_matrix.rs"]
pub mod sweep_radix_sort_oracle_matrix;

/// Integration tests from `tests/sweep_reduce_oracle_matrix.rs`.
#[cfg(feature = "reduce")]
#[path = "sweep_reduce_oracle_matrix.rs"]
pub mod sweep_reduce_oracle_matrix;

/// Integration tests from `tests/sweep_segment_reduce_oracle_matrix.rs`.
#[path = "sweep_segment_reduce_oracle_matrix.rs"]
pub mod sweep_segment_reduce_oracle_matrix;
