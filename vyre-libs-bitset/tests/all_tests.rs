//! One binary for every integration test in this crate.

#[macro_use]
#[path = "gate_fixtures/mod.rs"]
pub mod gate_fixtures;

#[path = "dense_matvec_cases/mod.rs"]
pub mod dense_matvec_cases;

#[path = "adversarial_bitset_contains.rs"]
pub mod adversarial_bitset_contains;

#[path = "adversarial_bitset_ops.rs"]
pub mod adversarial_bitset_ops;

#[path = "adversarial_bitset_reduce_matrix.rs"]
pub mod adversarial_bitset_reduce_matrix;

#[path = "adversarial_boolean_packing_four_russians_readiness.rs"]
pub mod adversarial_boolean_packing_four_russians_readiness;

#[path = "bitset_word_contracts.rs"]
pub mod bitset_word_contracts;

#[path = "four_russians_dense_matvec_generated.rs"]
pub mod four_russians_dense_matvec_generated;

#[path = "proptest_bitset_zero.rs"]
pub mod proptest_bitset_zero;

#[path = "sweep_logical_reference_matrix.rs"]
pub mod sweep_logical_reference_matrix;

#[path = "sweep_bitset_oracle_matrix.rs"]
pub mod sweep_bitset_oracle_matrix;
