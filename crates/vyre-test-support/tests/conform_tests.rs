//! Conformance engine integration tests (parity, laws, lenses, minimizer, replay).

#[path = "conform_suite/composition_discipline.rs"]
pub mod composition_discipline;

#[path = "conform_suite/connected_graph_conformance.rs"]
pub mod connected_graph_conformance;

#[path = "conform_suite/declared_law_proof.rs"]
pub mod declared_law_proof;

#[path = "conform_suite/fp_parity_ul_policy_contracts.rs"]
pub mod fp_parity_ul_policy_contracts;

#[path = "conform_suite/grid_sync_production_route.rs"]
pub mod grid_sync_production_route;

#[path = "conform_suite/invariants.rs"]
pub mod invariants;

#[path = "conform_suite/lens_parity.rs"]
pub mod lens_parity;

#[path = "conform_suite/library_contracts/mod.rs"]
pub mod library_contracts;

#[path = "conform_suite/minimizer_contract.rs"]
pub mod minimizer_contract;

#[path = "conform_suite/numeric_contract_conformance.rs"]
pub mod numeric_contract_conformance;

#[path = "conform_suite/production_route.rs"]
pub mod production_route;

#[path = "conform_suite/quantized_contract_conformance.rs"]
pub mod quantized_contract_conformance;

#[path = "conform_suite/replay_capsule_contract.rs"]
pub mod replay_capsule_contract;

#[path = "conform_suite/semantic_execution_contracts.rs"]
pub mod semantic_execution_contracts;

#[cfg(feature = "conform-gpu")]
#[path = "conform_suite/target_facet_backend_closure.rs"]
pub mod target_facet_backend_closure;

#[test]
fn vyre_conform_worker_entry() {
    vyre_test_support::conform::run_worker_from_env_or_exit();
}
