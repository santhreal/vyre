//! One binary for every default-feature integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Shared fixture module from `tests/program_fixtures/mod.rs`.
#[path = "program_fixtures/mod.rs"]
pub mod program_fixtures;

/// Integration tests from `tests/artifact_report.rs`.
#[path = "artifact_report.rs"]
pub mod artifact_report;

/// Integration tests from `tests/cli_find_dangling_exit_codes.rs`.
#[path = "cli_find_dangling_exit_codes.rs"]
pub mod cli_find_dangling_exit_codes;

/// Integration tests from `tests/dangling_ref_contracts.rs`.
#[path = "dangling_ref_contracts.rs"]
pub mod dangling_ref_contracts;

/// Integration tests from `tests/descriptor_diff_contracts.rs`.
#[path = "descriptor_diff_contracts.rs"]
pub mod descriptor_diff_contracts;

/// Integration tests from `tests/descriptor_dump_contracts.rs`.
#[path = "descriptor_dump_contracts.rs"]
pub mod descriptor_dump_contracts;

/// Integration tests from `tests/generated_descriptor_diff_matrix.rs`.
#[path = "generated_descriptor_diff_matrix.rs"]
pub mod generated_descriptor_diff_matrix;

/// Integration tests from `tests/loop_carrier_contracts.rs`.
#[path = "loop_carrier_contracts.rs"]
pub mod loop_carrier_contracts;

/// Integration tests from `tests/naga_trace_contracts.rs`.
#[path = "naga_trace_contracts.rs"]
pub mod naga_trace_contracts;

/// Integration tests from `tests/neutral_target_debug_separation.rs`.
#[path = "neutral_target_debug_separation.rs"]
pub mod neutral_target_debug_separation;

/// Integration tests from `tests/registry_closure.rs`.
#[path = "registry_closure.rs"]
pub mod registry_closure;

/// Integration tests from `tests/sanitizer_contract.rs`.
#[path = "sanitizer_contract.rs"]
pub mod sanitizer_contract;

/// Integration tests from `tests/well_formed_lowering_contracts.rs`.
#[path = "well_formed_lowering_contracts.rs"]
pub mod well_formed_lowering_contracts;

/// Integration tests from `tests/causal_report_contracts.rs`.
#[path = "causal_report_contracts.rs"]
pub mod causal_report_contracts;
/// Integration tests from `tests/wgsl_dump_contracts.rs`.
#[path = "wgsl_dump_contracts.rs"]
pub mod wgsl_dump_contracts;

/// Integration tests from `tests/compiler_level_contracts.rs`.
#[path = "compiler_level_contracts.rs"]
pub mod compiler_level_contracts;
