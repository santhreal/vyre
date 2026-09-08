//! One binary for every predicate integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Integration tests from `tests/arg_of_slot_precision.rs`.
#[cfg(feature = "predicate")]
#[path = "arg_of_slot_precision.rs"]
pub mod arg_of_slot_precision;

/// Integration tests from `tests/node_kind_eq_ir_parity_proptest.rs`.
#[path = "node_kind_eq_ir_parity_proptest.rs"]
pub mod node_kind_eq_ir_parity_proptest;

/// Integration tests from `tests/sweep_predicate_node_kind_oracle_matrix.rs`.
#[path = "sweep_predicate_node_kind_oracle_matrix.rs"]
pub mod sweep_predicate_node_kind_oracle_matrix;
