//! One binary for every default-feature integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `tools/xtask/test-harness-isolation.toml`.

/// Shared structural probes from `tests/support/naga_probe.rs`.
#[path = "support/naga_probe.rs"]
mod naga_probe;

/// Integration tests from `tests/adversarial_emit_program_matrix.rs`.
#[path = "adversarial_emit_program_matrix.rs"]
mod adversarial_emit_program_matrix;

/// Integration tests from `tests/carrier_scope_regression.rs`.
#[path = "carrier_scope_regression.rs"]
mod carrier_scope_regression;

/// Integration tests from `tests/emit_contracts/mod.rs`.
#[path = "emit_contracts/mod.rs"]
mod emit_contracts;

/// Integration tests from `tests/retargeted_diagnostic.rs`.
#[path = "retargeted_diagnostic.rs"]
mod retargeted_diagnostic;

/// Integration tests from `tests/target_capabilities.rs`.
#[path = "target_capabilities.rs"]
mod target_capabilities;

/// Integration tests from `tests/vec_pack_hazards.rs`.
#[path = "vec_pack_hazards.rs"]
mod vec_pack_hazards;

/// Integration tests from `tests/physical_ir_variant_exhaustiveness.rs`.
#[path = "physical_ir_variant_exhaustiveness.rs"]
mod physical_ir_variant_exhaustiveness;

/// Integration tests from `tests/grid_sync_dispatch_segments.rs`.
#[path = "grid_sync_dispatch_segments.rs"]
mod grid_sync_dispatch_segments;
