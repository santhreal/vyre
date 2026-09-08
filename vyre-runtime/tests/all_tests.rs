//! One binary for every default-feature integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Shared fixture module from `tests/prefix_cache_fixtures/mod.rs`.
#[allow(clippy::assertions_on_constants, clippy::needless_range_loop)]
#[path = "prefix_cache_fixtures/mod.rs"]
pub mod prefix_cache_fixtures;

/// Shared expectation module from `tests/ring_expectations/mod.rs`.
#[path = "ring_expectations/mod.rs"]
pub mod ring_expectations;

/// Shared fixture module from `tests/artifact_session_fixtures/mod.rs`.
#[path = "artifact_session_fixtures/mod.rs"]
pub mod artifact_session_fixtures;

/// Integration tests from `tests/adversarial_disk.rs`.
#[path = "adversarial_disk.rs"]
pub mod adversarial_disk;

/// Integration tests from `tests/artifact_admission_contract.rs`.
#[path = "artifact_admission_contract.rs"]
pub mod artifact_admission_contract;

/// Integration tests from `tests/artifact_workspace_contract.rs`.
#[path = "artifact_workspace_contract.rs"]
pub mod artifact_workspace_contract;

/// Integration tests from `tests/artifact_typed_resource_ingestion_contracts.rs`.
#[path = "artifact_typed_resource_ingestion_contracts.rs"]
pub mod artifact_typed_resource_ingestion_contracts;

/// Integration tests from `tests/cache_eviction_proptest.rs`.
#[path = "cache_eviction_proptest.rs"]
pub mod cache_eviction_proptest;

/// Integration tests from `tests/concurrency_invariants.rs`.
#[path = "concurrency_invariants.rs"]
pub mod concurrency_invariants;

/// Integration tests from `tests/driver_runtime_lifecycle_boundary.rs`.
#[path = "driver_runtime_lifecycle_boundary.rs"]
pub mod driver_runtime_lifecycle_boundary;
/// Integration tests from `tests/interactive_session_contract.rs`.
#[path = "interactive_session_contract.rs"]
pub mod interactive_session_contract;

/// Integration tests from `tests/multi_tenant_scheduler.rs`.
#[allow(clippy::assertions_on_constants)]
#[path = "multi_tenant_scheduler.rs"]
pub mod multi_tenant_scheduler;

/// Integration tests from `tests/paged_prefix_mtp_contracts.rs`.
#[path = "paged_prefix_mtp_contracts.rs"]
pub mod paged_prefix_mtp_contracts;

/// Integration tests from `tests/pipeline_fingerprint_surface.rs`.
#[path = "pipeline_fingerprint_surface.rs"]
pub mod pipeline_fingerprint_surface;

/// Integration tests from `tests/pipeline_error_fault_classification.rs`.
#[path = "pipeline_error_fault_classification.rs"]
pub mod pipeline_error_fault_classification;
/// Integration tests from `tests/portfolio_admission_contract.rs`.
#[path = "portfolio_admission_contract.rs"]
pub mod portfolio_admission_contract;

/// Integration tests from `tests/registry_closure.rs`.
#[path = "registry_closure.rs"]
pub mod registry_closure;

/// Integration tests from `tests/replay_log.rs`.
#[path = "replay_log.rs"]
pub mod replay_log;

/// Integration tests from `tests/resident_queue_contracts.rs`.
#[path = "resident_queue_contracts.rs"]
pub mod resident_queue_contracts;

/// Integration tests from `tests/resident_ring_error_classification.rs`.
#[path = "resident_ring_error_classification.rs"]
pub mod resident_ring_error_classification;

/// Integration tests from `tests/resident_work_queue_advanced_hierarchical_atomics_contracts.rs`.
#[cfg(feature = "megakernel-batch")]
#[path = "resident_work_queue_advanced_hierarchical_atomics_contracts.rs"]
pub mod resident_work_queue_advanced_hierarchical_atomics_contracts;

/// Integration tests from `tests/resident_work_queue_advanced_parallel_dfa_contracts.rs`.
#[cfg(feature = "megakernel-batch")]
#[path = "resident_work_queue_advanced_parallel_dfa_contracts.rs"]
pub mod resident_work_queue_advanced_parallel_dfa_contracts;

/// Integration tests from `tests/resident_work_queue_advanced_zero_copy_io_contracts.rs`.
#[cfg(feature = "megakernel-batch")]
#[path = "resident_work_queue_advanced_zero_copy_io_contracts.rs"]
pub mod resident_work_queue_advanced_zero_copy_io_contracts;

/// Integration tests from `tests/resident_work_queue_adversarial_buffers.rs`.
#[path = "resident_work_queue_adversarial_buffers.rs"]
pub mod resident_work_queue_adversarial_buffers;

/// Integration tests from `tests/resident_work_queue_adversarial_metrics.rs`.
#[path = "resident_work_queue_adversarial_metrics.rs"]
pub mod resident_work_queue_adversarial_metrics;

/// Integration tests from `tests/resident_work_queue_adversarial_overflow.rs`.
#[path = "resident_work_queue_adversarial_overflow.rs"]
pub mod resident_work_queue_adversarial_overflow;

/// Integration tests from `tests/resident_work_queue_adversarial_packing.rs`.
#[path = "resident_work_queue_adversarial_packing.rs"]
pub mod resident_work_queue_adversarial_packing;

/// Integration tests from `tests/resident_work_queue_allocation_bounds.rs`.
#[path = "resident_work_queue_allocation_bounds.rs"]
pub mod resident_work_queue_allocation_bounds;

/// Integration tests from `tests/resident_work_queue_async_observability.rs`.
#[cfg(feature = "megakernel-batch")]
#[path = "resident_work_queue_async_observability.rs"]
pub mod resident_work_queue_async_observability;

/// Integration tests from `tests/resident_work_queue_automata_worklist_contracts.rs`.
#[path = "resident_work_queue_automata_worklist_contracts.rs"]
pub mod resident_work_queue_automata_worklist_contracts;

/// Integration tests from `tests/resident_work_queue_barrier_elision_variant_closure.rs`.
#[path = "resident_work_queue_barrier_elision_variant_closure.rs"]
pub mod resident_work_queue_barrier_elision_variant_closure;

/// Integration tests from `tests/resident_work_queue_builder_delegation_parity.rs`.
#[path = "resident_work_queue_builder_delegation_parity.rs"]
pub mod resident_work_queue_builder_delegation_parity;

/// Integration tests from `tests/resident_work_queue_core_contracts.rs`.
#[path = "resident_work_queue_core_contracts.rs"]
pub mod resident_work_queue_core_contracts;

/// Integration tests from `tests/resident_work_queue_cpu_fallback_wording.rs`.
#[path = "resident_work_queue_cpu_fallback_wording.rs"]
pub mod resident_work_queue_cpu_fallback_wording;

/// Integration tests from `tests/resident_work_queue_duplicate_packing.rs`.
#[path = "resident_work_queue_duplicate_packing.rs"]
pub mod resident_work_queue_duplicate_packing;

/// Integration tests from `tests/resident_work_queue_host_protocol_contracts.rs`.
#[path = "resident_work_queue_host_protocol_contracts.rs"]
pub mod resident_work_queue_host_protocol_contracts;

/// Integration tests from `tests/resident_work_queue_io_public_errors.rs`.
#[path = "resident_work_queue_io_public_errors.rs"]
pub mod resident_work_queue_io_public_errors;

/// Integration tests from `tests/resident_work_queue_mixed_work_contracts.rs`.
#[path = "resident_work_queue_mixed_work_contracts.rs"]
pub mod resident_work_queue_mixed_work_contracts;

/// Integration tests from `tests/resident_work_queue_overflow_boundaries.rs`.
#[path = "resident_work_queue_overflow_boundaries.rs"]
pub mod resident_work_queue_overflow_boundaries;

/// Integration tests from `tests/resident_work_queue_planner_launch_contracts.rs`.
#[path = "resident_work_queue_planner_launch_contracts.rs"]
pub mod resident_work_queue_planner_launch_contracts;

/// Integration tests from `tests/resident_work_queue_protocol_boundary.rs`.
#[path = "resident_work_queue_protocol_boundary.rs"]
pub mod resident_work_queue_protocol_boundary;

/// Integration tests from `tests/resident_work_queue_protocol_codec_contracts.rs`.
#[path = "resident_work_queue_protocol_codec_contracts.rs"]
pub mod resident_work_queue_protocol_codec_contracts;

/// Integration tests from `tests/resident_work_queue_protocol_edge_cases.rs`.
#[path = "resident_work_queue_protocol_edge_cases.rs"]
pub mod resident_work_queue_protocol_edge_cases;

/// Integration tests from `tests/resident_work_queue_protocol_layout_contracts.rs`.
#[allow(clippy::assertions_on_constants)]
#[path = "resident_work_queue_protocol_layout_contracts.rs"]
pub mod resident_work_queue_protocol_layout_contracts;

/// Integration tests from `tests/resident_work_queue_protocol_strict_contracts.rs`.
#[path = "resident_work_queue_protocol_strict_contracts.rs"]
pub mod resident_work_queue_protocol_strict_contracts;

/// Integration tests from `tests/resident_work_queue_readback_contracts.rs`.
#[path = "resident_work_queue_readback_contracts.rs"]
pub mod resident_work_queue_readback_contracts;

/// Integration tests from `tests/resident_work_queue_rule_catalog_contracts.rs`.
#[cfg(feature = "megakernel-batch")]
#[path = "resident_work_queue_rule_catalog_contracts.rs"]
pub mod resident_work_queue_rule_catalog_contracts;

/// Integration tests from `tests/resident_work_queue_rule_catalog_scratch.rs`.
#[cfg(feature = "megakernel-batch")]
#[path = "resident_work_queue_rule_catalog_scratch.rs"]
pub mod resident_work_queue_rule_catalog_scratch;

/// Integration tests from `tests/resident_work_queue_scheduler_fairness.rs`.
#[allow(clippy::assertions_on_constants)]
#[path = "resident_work_queue_scheduler_fairness.rs"]
pub mod resident_work_queue_scheduler_fairness;

/// Integration tests from `tests/resident_work_queue_sketch_telemetry.rs`.
#[path = "resident_work_queue_sketch_telemetry.rs"]
pub mod resident_work_queue_sketch_telemetry;

/// Integration tests from `tests/resident_work_queue_workspace_layout_contracts.rs`.
#[path = "resident_work_queue_workspace_layout_contracts.rs"]
pub mod resident_work_queue_workspace_layout_contracts;

/// Integration tests from `tests/resource_residency.rs`.
#[path = "resource_residency.rs"]
pub mod resource_residency;

/// Integration tests from `tests/ring_fault_selection_is_typed.rs`.
#[path = "ring_fault_selection_is_typed.rs"]
pub mod ring_fault_selection_is_typed;

/// Integration tests from `tests/routing_policy.rs`.
#[path = "routing_policy.rs"]
pub mod routing_policy;

/// Integration tests from `tests/routing_standard_policy_contracts.rs`.
#[path = "routing_standard_policy_contracts.rs"]
pub mod routing_standard_policy_contracts;

/// Integration tests from `tests/safetensors_transfer_integrity_contracts.rs`.
#[path = "safetensors_transfer_integrity_contracts.rs"]
pub mod safetensors_transfer_integrity_contracts;

/// Integration tests from `tests/scheduler_model_proptest.rs`.
#[allow(clippy::needless_range_loop)]
#[path = "scheduler_model_proptest.rs"]
pub mod scheduler_model_proptest;

/// Integration tests from `tests/socket_ingest.rs`.
#[cfg(target_os = "linux")]
#[path = "socket_ingest.rs"]
pub mod socket_ingest;

/// Integration tests from `tests/sweep_ring_buffer_oracle_matrix.rs`.
#[path = "sweep_ring_buffer_oracle_matrix.rs"]
pub mod sweep_ring_buffer_oracle_matrix;

/// Integration tests from `tests/session_state_machine_contracts.rs`.
#[path = "session_state_machine_contracts.rs"]
pub mod session_state_machine_contracts;

/// Integration tests from `tests/sweep_tenant_policy_oracle_matrix.rs`.
#[path = "sweep_tenant_policy_oracle_matrix.rs"]
pub mod sweep_tenant_policy_oracle_matrix;

/// Integration tests from `tests/uring_completion_pump_contracts.rs`.
#[path = "uring_completion_pump_contracts.rs"]
pub mod uring_completion_pump_contracts;

/// Integration tests from `tests/uring_ingest_telemetry_invariants.rs`.
#[cfg(target_os = "linux")]
#[path = "uring_ingest_telemetry_invariants.rs"]
pub mod uring_ingest_telemetry_invariants;

/// Integration tests from `tests/uring_smoke.rs`.
#[cfg(target_os = "linux")]
#[path = "uring_smoke.rs"]
pub mod uring_smoke;
