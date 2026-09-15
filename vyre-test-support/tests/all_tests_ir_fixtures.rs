//! One binary for every ir-fixtures integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Integration tests from `tests/adversarial_and_mutation_contracts.rs`.
#[path = "adversarial_and_mutation_contracts.rs"]
pub mod adversarial_and_mutation_contracts;

/// Integration tests from `tests/binop_parity_tables.rs`.
#[path = "binop_parity_tables.rs"]
pub mod binop_parity_tables;

/// Integration tests from `tests/cast_parity_tables.rs`.
#[path = "cast_parity_tables.rs"]
pub mod cast_parity_tables;

/// Integration tests from `tests/differential_matrix_contracts.rs`.
#[path = "differential_matrix_contracts.rs"]
pub mod differential_matrix_contracts;

/// Integration tests from `tests/expr_variant_coverage.rs`.
#[path = "expr_variant_coverage.rs"]
pub mod expr_variant_coverage;

/// Integration tests from `tests/extension_variant_coverage.rs`.
#[path = "extension_variant_coverage.rs"]
pub mod extension_variant_coverage;

/// Integration tests from `tests/memory_order_coverage.rs`.
#[path = "memory_order_coverage.rs"]
pub mod memory_order_coverage;

/// Integration tests from `tests/registry_nets_fire.rs`.
#[path = "registry_nets_fire.rs"]
pub mod registry_nets_fire;
