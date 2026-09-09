//! One binary for every default-feature integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Integration tests from `tests/adversarial_emit_program_matrix.rs`.
#[path = "adversarial_emit_program_matrix.rs"]
pub mod adversarial_emit_program_matrix;

/// Integration tests from `tests/carrier_scope_regression.rs`.
#[path = "carrier_scope_regression.rs"]
pub mod carrier_scope_regression;

/// Integration tests from `tests/emit_contracts/mod.rs`.
#[path = "emit_contracts/mod.rs"]
pub mod emit_contracts;

/// Integration tests from `tests/emitted_artifact_byte_stability.rs`.
#[path = "emitted_artifact_byte_stability.rs"]
pub mod emitted_artifact_byte_stability;

/// Integration tests from `tests/lowering_digest.rs`.
#[path = "lowering_digest.rs"]
pub mod lowering_digest;

/// Integration tests from `tests/retargeted_diagnostic.rs`.
#[path = "retargeted_diagnostic.rs"]
pub mod retargeted_diagnostic;

/// Integration tests from `tests/target_capabilities.rs`.
#[path = "target_capabilities.rs"]
pub mod target_capabilities;

/// Integration tests from `tests/vec_pack_hazards.rs`.
#[path = "vec_pack_hazards.rs"]
pub mod vec_pack_hazards;

/// Integration tests from `tests/physical_ir_variant_exhaustiveness.rs`.
#[path = "physical_ir_variant_exhaustiveness.rs"]
pub mod physical_ir_variant_exhaustiveness;
