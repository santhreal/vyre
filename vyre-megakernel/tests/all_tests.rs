//! One binary for every default-feature integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Shared fixture module from `tests/graph_fixtures/mod.rs`.
#[path = "graph_fixtures/mod.rs"]
pub mod graph_fixtures;

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

/// Integration tests from `tests/compile_portfolio.rs`.
#[path = "compile_portfolio.rs"]
pub mod compile_portfolio;

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

/// Integration tests from `tests/mesh_topology_contract.rs`.
#[path = "mesh_topology_contract.rs"]
pub mod mesh_topology_contract;

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

/// Integration tests from `tests/specialization_contract.rs`.
#[path = "specialization_contract.rs"]
pub mod specialization_contract;

/// Integration tests from `tests/specialization_portfolio.rs`.
#[path = "specialization_portfolio.rs"]
pub mod specialization_portfolio;

/// Integration tests from `tests/target_payload_contract.rs`.
#[path = "target_payload_contract.rs"]
pub mod target_payload_contract;

/// Integration tests from `tests/topology_contract.rs`.
#[path = "topology_contract.rs"]
pub mod topology_contract;

/// Integration tests from `tests/real_time_objectives.rs`.
#[path = "real_time_objectives.rs"]
pub mod real_time_objectives;
