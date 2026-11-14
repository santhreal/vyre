//! One binary for every default-feature integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `tools/xtask/test-harness-isolation.toml`.

/// Shared fixture module from `tests/support/search_fixtures.rs`.
#[path = "support/search_fixtures.rs"]
pub mod search_fixtures;

/// Integration tests from `tests/algebraic_equivalence.rs`.
#[path = "algebraic_equivalence.rs"]
pub mod algebraic_equivalence;

/// Integration tests from `tests/artifact_contract.rs`.
#[path = "artifact_contract.rs"]
pub mod artifact_contract;

/// Integration tests from `tests/candidate_budget_and_dependency_endpoints.rs`.
#[path = "candidate_budget_and_dependency_endpoints.rs"]
pub mod candidate_budget_and_dependency_endpoints;

/// Integration tests from `tests/compile_objective.rs`.
#[path = "compile_objective.rs"]
pub mod compile_objective;

/// Integration tests from `tests/frontier_topology_selection.rs`.
#[path = "frontier_topology_selection.rs"]
pub mod frontier_topology_selection;

/// Integration tests from `tests/graph_result_values.rs`.
#[path = "graph_result_values.rs"]
pub mod graph_result_values;

/// Integration tests from `tests/law_derived_candidates.rs`.
#[path = "law_derived_candidates.rs"]
pub mod law_derived_candidates;

/// Integration tests from `tests/level_stage_verdict.rs`.
#[path = "level_stage_verdict.rs"]
pub mod level_stage_verdict;

/// Integration tests from `tests/measurement_protocol.rs`.
#[path = "measurement_protocol.rs"]
pub mod measurement_protocol;

/// Integration tests from `tests/multi_fidelity_ladder.rs`.
#[path = "multi_fidelity_ladder.rs"]
pub mod multi_fidelity_ladder;

/// Integration tests from `tests/numeric_budget_legality.rs`.
#[path = "numeric_budget_legality.rs"]
pub mod numeric_budget_legality;

/// Integration tests from `tests/schedule_grammar_contract.rs`.
#[path = "schedule_grammar_contract.rs"]
pub mod schedule_grammar_contract;

/// Integration tests from `tests/selected_geometry_authority.rs`.
#[path = "selected_geometry_authority.rs"]
pub mod selected_geometry_authority;

/// Integration tests from `tests/selection_cost_contract.rs`.
#[path = "selection_cost_contract.rs"]
pub mod selection_cost_contract;

/// Integration tests from `tests/shared_tile_cost.rs`.
#[path = "shared_tile_cost.rs"]
pub mod shared_tile_cost;

/// Integration tests from `tests/staged_input_slots_contract.rs`.
#[path = "staged_input_slots_contract.rs"]
pub mod staged_input_slots_contract;

/// Integration tests from `tests/target_payload_contract.rs`.
#[path = "target_payload_contract.rs"]
pub mod target_payload_contract;

/// Integration tests from `tests/topology_contract.rs`.
#[path = "topology_contract.rs"]
pub mod topology_contract;

/// Integration tests from `tests/region_fusion_legality_contracts.rs`.
#[path = "region_fusion_legality_contracts.rs"]
pub mod region_fusion_legality_contracts;
/// Integration tests from `tests/schedule_calculus_megakernel_contracts.rs`.
#[path = "schedule_calculus_megakernel_contracts.rs"]
pub mod schedule_calculus_megakernel_contracts;

/// Integration tests of the `lower` module.
pub mod lower {

    /// Integration tests from `tests/async_transaction_contracts.rs`.
    #[path = "async_transaction_contracts.rs"]
    pub mod async_transaction_contracts;

    /// Integration tests from `tests/bank_conflict_strategy_contracts.rs`.
    #[path = "bank_conflict_strategy_contracts.rs"]
    pub mod bank_conflict_strategy_contracts;

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

    /// Integration tests from `tests/storage_layout_contracts.rs`.
    #[path = "storage_layout_contracts.rs"]
    pub mod storage_layout_contracts;

    /// Integration tests from `tests/target_capabilities.rs`.
    #[path = "target_capabilities.rs"]
    pub mod target_capabilities;

    /// Integration tests from `tests/tile_lowering_contracts.rs`.
    #[path = "tile_lowering_contracts.rs"]
    pub mod tile_lowering_contracts;
    /// Integration tests from `tests/verify_result_id_uniqueness.rs`.
    #[path = "verify_result_id_uniqueness.rs"]
    pub mod verify_result_id_uniqueness;

    /// Integration tests from `tests/physical_kernel_ir_policy_boundary.rs`.
    #[path = "physical_kernel_ir_policy_boundary.rs"]
    pub mod physical_kernel_ir_policy_boundary;

    /// Integration tests from `tests/physical_ir_emitter_exhaustive_decisions.rs`.
    #[path = "physical_ir_emitter_exhaustive_decisions.rs"]
    pub mod physical_ir_emitter_exhaustive_decisions;

    /// Integration tests from `tests/fragment_operand_provenance.rs`.
    #[path = "fragment_operand_provenance.rs"]
    pub mod fragment_operand_provenance;
}
