//! Consolidated integration tests for `xtask::debug` (`--features debug`).

#[path = "debug_suite/program_fixtures/mod.rs"]
pub mod program_fixtures;

#[path = "debug_suite/artifact_report.rs"]
pub mod artifact_report;

#[path = "debug_suite/cli_find_dangling_exit_codes.rs"]
pub mod cli_find_dangling_exit_codes;

#[path = "debug_suite/dangling_ref_contracts.rs"]
pub mod dangling_ref_contracts;

#[path = "debug_suite/descriptor_diff_contracts.rs"]
pub mod descriptor_diff_contracts;

#[path = "debug_suite/descriptor_dump_contracts.rs"]
pub mod descriptor_dump_contracts;

#[path = "debug_suite/generated_descriptor_diff_matrix.rs"]
pub mod generated_descriptor_diff_matrix;

#[path = "debug_suite/loop_carrier_contracts.rs"]
pub mod loop_carrier_contracts;

#[path = "debug_suite/naga_trace_contracts.rs"]
pub mod naga_trace_contracts;

#[path = "debug_suite/neutral_target_debug_separation.rs"]
pub mod neutral_target_debug_separation;

#[path = "debug_suite/sanitizer_contract.rs"]
pub mod sanitizer_contract;

#[path = "debug_suite/well_formed_lowering_contracts.rs"]
pub mod well_formed_lowering_contracts;

#[path = "debug_suite/causal_report_contracts.rs"]
pub mod causal_report_contracts;

#[path = "debug_suite/wgsl_dump_contracts.rs"]
pub mod wgsl_dump_contracts;

#[path = "debug_suite/compiler_level_contracts.rs"]
pub mod compiler_level_contracts;
