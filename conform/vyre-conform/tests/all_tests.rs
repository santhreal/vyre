//! One binary for every default-feature integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Integration tests from `tests/cert_artifact/mod.rs`.
#[path = "cert_artifact/mod.rs"]
pub mod cert_artifact;

/// Integration tests from `tests/composition_discipline.rs`.
#[path = "composition_discipline.rs"]
pub mod composition_discipline;

/// Integration tests from `tests/connected_graph_conformance.rs`.
#[path = "connected_graph_conformance.rs"]
pub mod connected_graph_conformance;

/// Integration tests from `tests/declared_law_proof.rs`.
#[path = "declared_law_proof.rs"]
pub mod declared_law_proof;

/// Integration tests from `tests/fp_parity_ul_policy_contracts.rs`.
#[path = "fp_parity_ul_policy_contracts.rs"]
pub mod fp_parity_ul_policy_contracts;

/// Integration tests from `tests/grid_sync_production_route.rs`.
#[path = "grid_sync_production_route.rs"]
pub mod grid_sync_production_route;

/// Integration tests from `tests/invariants.rs`.
#[path = "invariants.rs"]
pub mod invariants;

/// Integration tests from `tests/lens_buffer_state_contracts.rs`.
#[path = "lens_buffer_state_contracts.rs"]
pub mod lens_buffer_state_contracts;

/// Integration tests from `tests/lens_fixpoint_contracts.rs`.
#[path = "lens_fixpoint_contracts.rs"]
pub mod lens_fixpoint_contracts;

/// Integration tests from `tests/lens_parity.rs`.
#[path = "lens_parity.rs"]
pub mod lens_parity;

/// Integration tests from `tests/library_contracts/mod.rs`.
#[path = "library_contracts/mod.rs"]
pub mod library_contracts;

/// Integration tests from `tests/mesh_placement_contracts.rs`.
#[path = "mesh_placement_contracts.rs"]
pub mod mesh_placement_contracts;

/// Integration tests from `tests/minimizer_contract.rs`.
#[path = "minimizer_contract.rs"]
pub mod minimizer_contract;

/// Integration tests from `tests/numeric_contract_conformance.rs`.
#[path = "numeric_contract_conformance.rs"]
pub mod numeric_contract_conformance;

/// Integration tests from `tests/op_matrix_truth/mod.rs`.
#[path = "op_matrix_truth/mod.rs"]
pub mod op_matrix_truth;

/// Integration tests from `tests/production_route.rs`.
#[path = "production_route.rs"]
pub mod production_route;

/// Integration tests from `tests/quantized_contract_conformance.rs`.
#[path = "quantized_contract_conformance.rs"]
pub mod quantized_contract_conformance;

/// Integration tests from `tests/replay_capsule_contract.rs`.
#[path = "replay_capsule_contract.rs"]
pub mod replay_capsule_contract;

/// Integration tests from `tests/schema_compatibility.rs`.
#[path = "schema_compatibility.rs"]
pub mod schema_compatibility;

/// Integration tests from `tests/semantic_execution_contracts.rs`.
#[path = "semantic_execution_contracts.rs"]
pub mod semantic_execution_contracts;

/// Integration tests from `tests/target_facet_backend_closure.rs`.
#[path = "target_facet_backend_closure.rs"]
pub mod target_facet_backend_closure;

#[test]
fn vyre_conform_worker_entry() {
    vyre_conform::run_worker_from_env_or_exit();
}
