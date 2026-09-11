//! One binary for every default-feature integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Integration tests from `tests/cfg_gate_polarity.rs`.
#[path = "cfg_gate_polarity.rs"]
pub mod cfg_gate_polarity;

/// Integration tests from `tests/checkout_provenance.rs`.
#[path = "checkout_provenance.rs"]
pub mod checkout_provenance;

/// Integration tests from `tests/crate_ownership_registry_reader.rs`.
#[path = "crate_ownership_registry_reader.rs"]
pub mod crate_ownership_registry_reader;

/// Integration tests from `tests/crate_structure_contracts.rs`.
#[path = "crate_structure_contracts.rs"]
pub mod crate_structure_contracts;

/// Integration tests from `tests/deletion_evidence_and_pass_registry.rs`.
#[path = "deletion_evidence_and_pass_registry.rs"]
pub mod deletion_evidence_and_pass_registry;

/// Integration tests from `tests/device_only_routing.rs`.
#[path = "device_only_routing.rs"]
pub mod device_only_routing;

/// Integration tests from `tests/materializer_admission.rs`.
#[path = "materializer_admission.rs"]
pub mod materializer_admission;

/// Integration tests from `tests/module_routes.rs`.
#[path = "module_routes.rs"]
pub mod module_routes;

/// Integration tests from `tests/neutral_vocabulary_contract.rs`.
#[path = "neutral_vocabulary_contract.rs"]
pub mod neutral_vocabulary_contract;

/// Integration tests from `tests/node_child_descent_owner.rs`.
#[path = "node_child_descent_owner.rs"]
pub mod node_child_descent_owner;

/// Integration tests from `tests/published_module_reexports.rs`.
#[path = "published_module_reexports.rs"]
pub mod published_module_reexports;

/// Integration tests from `tests/scan_roster_pruning.rs`.
#[path = "scan_roster_pruning.rs"]
pub mod scan_roster_pruning;

/// Integration tests from `tests/source_directory_placement.rs`.
#[path = "source_directory_placement.rs"]
pub mod source_directory_placement;

/// Integration tests from `tests/string_literals.rs`.
#[path = "string_literals.rs"]
pub mod string_literals;

/// Integration tests from `tests/test_gated_items.rs`.
#[path = "test_gated_items.rs"]
pub mod test_gated_items;

/// Integration tests from `tests/test_gated_modules.rs`.
#[path = "test_gated_modules.rs"]
pub mod test_gated_modules;
