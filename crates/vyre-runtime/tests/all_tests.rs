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

/// Shared fixture module from `tests/fixture_backend/mod.rs`.
#[macro_use]
#[path = "fixture_backend/mod.rs"]
pub mod fixture_backend;

/// Integration tests from `tests/accounting_byte_range_accounting_contracts.rs`.
#[path = "accounting_byte_range_accounting_contracts.rs"]
pub mod accounting_byte_range_accounting_contracts;

/// Integration tests from `tests/accounting_checked_atomic_update_with_order_contracts.rs`.
#[path = "accounting_checked_atomic_update_with_order_contracts.rs"]
pub mod accounting_checked_atomic_update_with_order_contracts;

/// Integration tests from `tests/accounting_pinning_atomic_add_usize_with_order_contracts.rs`.
#[path = "accounting_pinning_atomic_add_usize_with_order_contracts.rs"]
pub mod accounting_pinning_atomic_add_usize_with_order_contracts;

/// Integration tests from `tests/actionable_errors.rs`.
#[path = "actionable_errors.rs"]
pub mod actionable_errors;

/// Integration tests from `tests/allocation_contracts.rs`.
#[path = "allocation_contracts.rs"]
pub mod allocation_contracts;

/// Integration tests from `tests/async_dispatch_contract.rs`.
#[path = "async_dispatch_contract.rs"]
pub mod async_dispatch_contract;

/// Integration tests from `tests/autotune_store_contracts.rs`.
#[path = "autotune_store_contracts.rs"]
pub mod autotune_store_contracts;

/// Integration tests from `tests/backend_launch_validation.rs`.
#[path = "backend_launch_validation.rs"]
pub mod backend_launch_validation;

/// Integration tests from `tests/backend_trait_contract.rs`.
#[path = "backend_trait_contract.rs"]
pub mod backend_trait_contract;

/// Integration tests from `tests/backend_validation_defaults.rs`.
#[path = "backend_validation_defaults.rs"]
pub mod backend_validation_defaults;

/// Integration tests from `tests/cache_eviction_contracts.rs`.
#[path = "cache_eviction_contracts.rs"]
pub mod cache_eviction_contracts;

/// Integration tests from `tests/cache_eviction_heat_contracts.rs`.
#[path = "cache_eviction_heat_contracts.rs"]
pub mod cache_eviction_heat_contracts;

/// Integration tests from `tests/capability_adversarial.rs`.
#[path = "capability_adversarial.rs"]
pub mod capability_adversarial;

/// Integration tests from `tests/consumer_boundary.rs`.
#[path = "consumer_boundary.rs"]
pub mod consumer_boundary;

/// Integration tests from `tests/device_signature_path.rs`.
#[path = "device_signature_path.rs"]
pub mod device_signature_path;

/// Integration tests from `tests/diagnostic_surface.rs`.
#[path = "diagnostic_surface.rs"]
pub mod diagnostic_surface;

/// Integration tests from `tests/dialect_admissible_facts.rs`.
#[path = "dialect_admissible_facts.rs"]
pub mod dialect_admissible_facts;

/// Integration tests from `tests/dispatch_config_surface.rs`.
#[path = "dispatch_config_surface.rs"]
pub mod dispatch_config_surface;

/// Integration tests from `tests/driver_contracts.rs`.
#[path = "driver_contracts.rs"]
pub mod driver_contracts;

/// Integration tests from `tests/driver_lifecycle_e2e.rs`.
#[path = "driver_lifecycle_e2e.rs"]
pub mod driver_lifecycle_e2e;

/// Integration tests from `tests/error_code_catalog.rs`.
#[path = "error_code_catalog.rs"]
pub mod error_code_catalog;

/// Integration tests from `tests/error_code_frozen.rs`.
#[path = "error_code_frozen.rs"]
pub mod error_code_frozen;

/// Integration tests from `tests/float_lowering_refusal.rs`.
#[path = "float_lowering_refusal.rs"]
pub mod float_lowering_refusal;

/// Integration tests from `tests/geometry_admitted_widths.rs`.
#[path = "geometry_admitted_widths.rs"]
pub mod geometry_admitted_widths;

/// Integration tests from `tests/grid_sync_capability_admits_either_route.rs`.
#[path = "grid_sync_capability_admits_either_route.rs"]
pub mod grid_sync_capability_admits_either_route;

/// Integration tests from `tests/grid_sync_detection_reaches_every_body_variant.rs`.
#[path = "grid_sync_detection_reaches_every_body_variant.rs"]
pub mod grid_sync_detection_reaches_every_body_variant;

/// Integration tests from `tests/grid_sync_nested_fence_survives_split.rs`.
#[path = "grid_sync_nested_fence_survives_split.rs"]
pub mod grid_sync_nested_fence_survives_split;

/// Integration tests from `tests/grid_sync_segments_declare_every_referenced_buffer.rs`.
#[path = "grid_sync_segments_declare_every_referenced_buffer.rs"]
pub mod grid_sync_segments_declare_every_referenced_buffer;

/// Integration tests from `tests/grid_sync_split_timing_contracts.rs`.
#[path = "grid_sync_split_timing_contracts.rs"]
pub mod grid_sync_split_timing_contracts;

/// Integration tests from `tests/host_input_abi_closure.rs`.
#[path = "host_input_abi_closure.rs"]
pub mod host_input_abi_closure;

/// Integration tests from `tests/hostile_input_probe_shapes.rs`.
#[cfg(feature = "test-fixtures")]
#[path = "hostile_input_probe_shapes.rs"]
pub mod hostile_input_probe_shapes;

/// Integration tests from `tests/input_identity_contracts.rs`.
#[path = "input_identity_contracts.rs"]
pub mod input_identity_contracts;

/// Integration tests from `tests/launch_grid_axis_folding.rs`.
#[path = "launch_grid_axis_folding.rs"]
pub mod launch_grid_axis_folding;

/// Integration tests from `tests/logical_markers_are_not_backend_operations.rs`.
#[path = "logical_markers_are_not_backend_operations.rs"]
pub mod logical_markers_are_not_backend_operations;

/// Integration tests from `tests/lock_poison_policy.rs`.
#[path = "lock_poison_policy.rs"]
pub mod lock_poison_policy;

/// Integration tests from `tests/numeric_contracts.rs`.
#[path = "numeric_contracts.rs"]
pub mod numeric_contracts;

/// Integration tests from `tests/omitted_launch_geometry.rs`.
#[path = "omitted_launch_geometry.rs"]
pub mod omitted_launch_geometry;

/// Integration tests from `tests/ordering_contracts.rs`.
#[path = "ordering_contracts.rs"]
pub mod ordering_contracts;

/// Integration tests from `tests/output_slots_contracts.rs`.
#[path = "output_slots_contracts.rs"]
pub mod output_slots_contracts;

/// Integration tests from `tests/persistent_contracts.rs`.
#[path = "persistent_contracts.rs"]
pub mod persistent_contracts;

/// Integration tests from `tests/read_only_alias_contracts.rs`.
#[path = "read_only_alias_contracts.rs"]
pub mod read_only_alias_contracts;

/// Integration tests from `tests/reference_oracle_loses_to_a_device.rs`.
#[path = "reference_oracle_loses_to_a_device.rs"]
pub mod reference_oracle_loses_to_a_device;

/// Integration tests from `tests/registry_closure.rs`.
#[path = "backend_registry_closure.rs"]
pub mod backend_registry_closure;

/// Integration tests from `tests/rejection_wording_contract.rs`.
#[path = "rejection_wording_contract.rs"]
pub mod rejection_wording_contract;

/// Integration tests from `tests/reservation_policy_contracts.rs`.
#[path = "reservation_policy_contracts.rs"]
pub mod reservation_policy_contracts;

/// Integration tests from `tests/resident_binding_projection.rs`.
#[path = "resident_binding_projection.rs"]
pub mod resident_binding_projection;

/// Integration tests from `tests/shape_prediction_contracts.rs`.
#[path = "shape_prediction_contracts.rs"]
pub mod shape_prediction_contracts;

/// Integration tests from `tests/speculation_verdict_contracts.rs`.
#[path = "speculation_verdict_contracts.rs"]
pub mod speculation_verdict_contracts;

/// Integration tests from `tests/staged_module_input_gathering.rs`.
#[path = "staged_module_input_gathering.rs"]
pub mod staged_module_input_gathering;

/// Integration tests from `tests/sweep_dispatch_shape_oracle_matrix.rs`.
#[path = "sweep_dispatch_shape_oracle_matrix.rs"]
pub mod sweep_dispatch_shape_oracle_matrix;

/// Integration tests from `tests/sweep_numeric_oracle_matrix.rs`.
#[path = "sweep_numeric_oracle_matrix.rs"]
pub mod sweep_numeric_oracle_matrix;

/// Integration tests from `tests/target_contract.rs`.
#[path = "target_contract.rs"]
pub mod target_contract;

/// Integration tests from `tests/transfer_accounting_contracts.rs`.
#[path = "transfer_accounting_contracts.rs"]
pub mod transfer_accounting_contracts;

/// Integration tests from `tests/unreported_device_facts.rs`.
#[path = "unreported_device_facts.rs"]
pub mod unreported_device_facts;

/// Integration tests from `tests/vyre_backend_forwarding_closure.rs`.
#[path = "vyre_backend_forwarding_closure.rs"]
pub mod vyre_backend_forwarding_closure;

/// Integration tests from `tests/lock_policy_closure.rs`.
#[path = "lock_policy_closure.rs"]
pub mod lock_policy_closure;

/// Integration tests from `tests/target_facet_lowering_arms.rs`.
#[path = "target_facet_lowering_arms.rs"]
pub mod target_facet_lowering_arms;
