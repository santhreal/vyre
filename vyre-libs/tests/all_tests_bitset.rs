//! One binary for every bitset integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Shared fixture module from `tests/dense_matvec_cases/mod.rs`.
#[path = "dense_matvec_cases/mod.rs"]
pub mod dense_matvec_cases;

/// Shared fixture module from `tests/gate_fixtures/mod.rs`.
#[macro_use]
#[path = "gate_fixtures/mod.rs"]
pub mod gate_fixtures;

/// Shared fixture module from `tests/wire_words/mod.rs`.
#[path = "wire_words/mod.rs"]
pub mod wire_words;

/// Integration tests from `tests/adversarial_bitset_contains.rs`.
#[path = "adversarial_bitset_contains.rs"]
pub mod adversarial_bitset_contains;

/// Integration tests from `tests/adversarial_bitset_ops.rs`.
#[cfg(feature = "bitset")]
#[path = "adversarial_bitset_ops.rs"]
pub mod adversarial_bitset_ops;

/// Integration tests from `tests/adversarial_bitset_reduce_matrix.rs`.
#[path = "adversarial_bitset_reduce_matrix.rs"]
pub mod adversarial_bitset_reduce_matrix;

/// Integration tests from `tests/adversarial_boolean_packing_four_russians_readiness.rs`.
#[cfg(feature = "bitset")]
#[path = "adversarial_boolean_packing_four_russians_readiness.rs"]
pub mod adversarial_boolean_packing_four_russians_readiness;

/// Integration tests from `tests/bitset_word_contracts.rs`.
#[path = "bitset_word_contracts.rs"]
pub mod bitset_word_contracts;

/// Integration tests from `tests/four_russians_dense_matvec_generated.rs`.
#[path = "four_russians_dense_matvec_generated.rs"]
pub mod four_russians_dense_matvec_generated;

/// Integration tests from `tests/proptest_bitset_zero.rs`.
#[cfg(feature = "bitset")]
#[path = "proptest_bitset_zero.rs"]
pub mod proptest_bitset_zero;

/// Integration tests from `tests/sweep_bitset_oracle_matrix.rs`.
#[cfg(feature = "bitset")]
#[path = "sweep_bitset_oracle_matrix.rs"]
pub mod sweep_bitset_oracle_matrix;
