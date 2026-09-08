//! One binary for every nn-activation integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Shared fixture module from `tests/wire_words/mod.rs`.
#[path = "wire_words/mod.rs"]
pub mod wire_words;

/// Integration tests from `tests/indexed_map_composition_contracts.rs`.
#[path = "indexed_map_composition_contracts.rs"]
pub mod indexed_map_composition_contracts;

/// Integration tests from `tests/int4_primitive_composition.rs`.
#[cfg(feature = "nn-linear-4bit")]
#[path = "int4_primitive_composition.rs"]
pub mod int4_primitive_composition;

/// Integration tests from `tests/mlp_4x_leaky_sq_multi_workgroup_span.rs`.
#[cfg(feature = "nn-activation")]
#[path = "mlp_4x_leaky_sq_multi_workgroup_span.rs"]
pub mod mlp_4x_leaky_sq_multi_workgroup_span;

/// Integration tests from `tests/sigmoid_gate_typed_contract.rs`.
#[path = "sigmoid_gate_typed_contract.rs"]
pub mod sigmoid_gate_typed_contract;
