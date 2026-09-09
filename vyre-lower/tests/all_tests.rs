//! One binary for every default-feature integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Integration tests from `tests/affine_access_map_contracts.rs`.
#[path = "affine_access_map_contracts.rs"]
pub mod affine_access_map_contracts;

/// Integration tests from `tests/analysis_fixture_corpuses.rs`.
#[path = "analysis_fixture_corpuses.rs"]
pub mod analysis_fixture_corpuses;

/// Integration tests from `tests/analysis_representation_boundary.rs`.
#[path = "analysis_representation_boundary.rs"]
pub mod analysis_representation_boundary;

/// Integration tests from `tests/async_transaction_contracts.rs`.
#[path = "async_transaction_contracts.rs"]
pub mod async_transaction_contracts;

/// Integration tests from `tests/bank_conflict_strategy_contracts.rs`.
#[path = "bank_conflict_strategy_contracts.rs"]
pub mod bank_conflict_strategy_contracts;

/// Integration tests from `tests/device_fact_boundary.rs`.
#[path = "device_fact_boundary.rs"]
pub mod device_fact_boundary;

/// Integration tests from `tests/level_stage_verdict.rs`.
#[path = "level_stage_verdict.rs"]
pub mod level_stage_verdict;

/// Integration tests from `tests/lower_visibility_contract.rs`.
#[path = "lower_visibility_contract.rs"]
pub mod lower_visibility_contract;

/// Integration tests from `tests/lowering_contracts/mod.rs`.
#[path = "lowering_contracts/mod.rs"]
pub mod lowering_contracts;

/// Integration tests from `tests/lowering_equivalence.rs`.
#[path = "lowering_equivalence.rs"]
pub mod lowering_equivalence;

/// Integration tests from `tests/matrix_fragment_contracts.rs`.
#[path = "matrix_fragment_contracts.rs"]
pub mod matrix_fragment_contracts;

/// Integration tests from `tests/physical_schedule_handoff.rs`.
#[path = "physical_schedule_handoff.rs"]
pub mod physical_schedule_handoff;

/// Integration tests from `tests/resource_bounds_contracts.rs`.
#[path = "resource_bounds_contracts.rs"]
pub mod resource_bounds_contracts;

/// Integration tests from `tests/rewrite_layer_contract.rs`.
#[path = "rewrite_layer_contract.rs"]
pub mod rewrite_layer_contract;

/// Integration tests from `tests/shared_store_race_legality.rs`.
#[path = "shared_store_race_legality.rs"]
pub mod shared_store_race_legality;

/// Integration tests from `tests/storage_layout_contracts.rs`.
#[path = "storage_layout_contracts.rs"]
pub mod storage_layout_contracts;

/// Integration tests from `tests/target_capabilities.rs`.
#[path = "target_capabilities.rs"]
pub mod target_capabilities;

/// Integration tests from `tests/verify_result_id_uniqueness.rs`.
#[path = "verify_result_id_uniqueness.rs"]
pub mod verify_result_id_uniqueness;
/// Integration tests from `tests/tile_lowering_contracts.rs`.
#[path = "tile_lowering_contracts.rs"]
pub mod tile_lowering_contracts;
