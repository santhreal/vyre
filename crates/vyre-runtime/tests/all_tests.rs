//! One binary for every default-feature integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `tools/xtask/test-harness-isolation.toml`.

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

/// Integration tests from `tests/multi_tenant_scheduler.rs`.
#[allow(clippy::assertions_on_constants)]
#[path = "multi_tenant_scheduler.rs"]
pub mod multi_tenant_scheduler;

/// Integration tests from `tests/pipeline_fingerprint_surface.rs`.
#[path = "pipeline_fingerprint_surface.rs"]
pub mod pipeline_fingerprint_surface;

/// Integration tests from `tests/pipeline_error_fault_classification.rs`.
#[path = "pipeline_error_fault_classification.rs"]
pub mod pipeline_error_fault_classification;

/// Integration tests from `tests/registry_closure.rs`.
#[path = "registry_closure.rs"]
pub mod registry_closure;

/// Integration tests from `tests/resident_ring_error_classification.rs`.
#[path = "resident_ring_error_classification.rs"]
pub mod resident_ring_error_classification;

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

/// Integration tests from `tests/resident_work_queue_barrier_elision_variant_closure.rs`.
#[path = "resident_work_queue_barrier_elision_variant_closure.rs"]
pub mod resident_work_queue_barrier_elision_variant_closure;

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

/// Integration tests from `tests/resident_work_queue_scheduler_fairness.rs`.
#[allow(clippy::assertions_on_constants)]
#[path = "resident_work_queue_scheduler_fairness.rs"]
pub mod resident_work_queue_scheduler_fairness;

/// Integration tests from `tests/ring_fault_selection_is_typed.rs`.
#[path = "ring_fault_selection_is_typed.rs"]
pub mod ring_fault_selection_is_typed;

/// Integration tests from `tests/scheduler_model_proptest.rs`.
#[allow(clippy::needless_range_loop)]
#[path = "scheduler_model_proptest.rs"]
pub mod scheduler_model_proptest;

/// Integration tests from `tests/socket_ingest.rs`.
#[cfg(target_os = "linux")]
#[path = "socket_ingest.rs"]
pub mod socket_ingest;

/// Integration tests from `tests/session_state_machine_contracts.rs`.
#[path = "session_state_machine_contracts.rs"]
pub mod session_state_machine_contracts;

/// Integration tests from `tests/uring_ingest_telemetry_invariants.rs`.
#[cfg(target_os = "linux")]
#[path = "uring_ingest_telemetry_invariants.rs"]
pub mod uring_ingest_telemetry_invariants;

/// Integration tests from `tests/uring_smoke.rs`.
#[cfg(target_os = "linux")]
#[path = "uring_smoke.rs"]
pub mod uring_smoke;
