//! One binary for every default-feature integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

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

/// Integration tests from `tests/arm_independence_contracts.rs`.
#[path = "arm_independence_contracts.rs"]
pub mod arm_independence_contracts;

/// Integration tests from `tests/async_copy_overlap_contracts.rs`.
#[path = "async_copy_overlap_contracts.rs"]
pub mod async_copy_overlap_contracts;

/// Integration tests from `tests/async_dispatch_contract.rs`.
#[path = "async_dispatch_contract.rs"]
pub mod async_dispatch_contract;

/// Integration tests from `tests/atomic_file_operation_race_policy.rs`.
#[path = "atomic_file_operation_race_policy.rs"]
pub mod atomic_file_operation_race_policy;

/// Integration tests from `tests/autotune_store_contracts.rs`.
#[path = "autotune_store_contracts.rs"]
pub mod autotune_store_contracts;

/// Integration tests from `tests/backend_capability_digests.rs`.
#[path = "backend_capability_digests.rs"]
pub mod backend_capability_digests;

/// Integration tests from `tests/backend_handle_lifetime_provenance.rs`.
#[path = "backend_handle_lifetime_provenance.rs"]
pub mod backend_handle_lifetime_provenance;

/// Integration tests from `tests/backend_launch_validation.rs`.
#[path = "backend_launch_validation.rs"]
pub mod backend_launch_validation;

/// Integration tests from `tests/backend_trait_contract.rs`.
#[path = "backend_trait_contract.rs"]
pub mod backend_trait_contract;

/// Integration tests from `tests/backend_validation_defaults.rs`.
#[path = "backend_validation_defaults.rs"]
pub mod backend_validation_defaults;

/// Integration tests from `tests/backpressure_queue_quota_policy.rs`.
#[path = "backpressure_queue_quota_policy.rs"]
pub mod backpressure_queue_quota_policy;

/// Integration tests from `tests/benchmark_pass_selection_contracts.rs`.
#[path = "benchmark_pass_selection_contracts.rs"]
pub mod benchmark_pass_selection_contracts;

/// Integration tests from `tests/bindless_policy_contracts.rs`.
#[path = "bindless_policy_contracts.rs"]
pub mod bindless_policy_contracts;

/// Integration tests from `tests/cache_eviction_contracts.rs`.
#[path = "cache_eviction_contracts.rs"]
pub mod cache_eviction_contracts;

/// Integration tests from `tests/cache_eviction_heat_contracts.rs`.
#[path = "cache_eviction_heat_contracts.rs"]
pub mod cache_eviction_heat_contracts;

/// Integration tests from `tests/capability_adversarial.rs`.
#[path = "capability_adversarial.rs"]
pub mod capability_adversarial;

/// Integration tests from `tests/command_execution_boundary.rs`.
#[path = "command_execution_boundary.rs"]
pub mod command_execution_boundary;

/// Integration tests from `tests/command_reuse_policy_contracts.rs`.
#[path = "command_reuse_policy_contracts.rs"]
pub mod command_reuse_policy_contracts;

/// Integration tests from `tests/concurrency_schedule_contracts.rs`.
#[path = "concurrency_schedule_contracts.rs"]
pub mod concurrency_schedule_contracts;

/// Integration tests from `tests/consumer_boundary.rs`.
#[path = "consumer_boundary.rs"]
pub mod consumer_boundary;

/// Integration tests from `tests/crypto_rng_key_lifecycle.rs`.
#[path = "crypto_rng_key_lifecycle.rs"]
pub mod crypto_rng_key_lifecycle;

/// Integration tests from `tests/device_convergence_contracts.rs`.
#[path = "device_convergence_contracts.rs"]
pub mod device_convergence_contracts;

/// Integration tests from `tests/device_diagnostic_aggregation_contracts.rs`.
#[path = "device_diagnostic_aggregation_contracts.rs"]
pub mod device_diagnostic_aggregation_contracts;

/// Integration tests from `tests/device_signature_path.rs`.
#[path = "device_signature_path.rs"]
pub mod device_signature_path;

/// Integration tests from `tests/device_work_queue_contracts.rs`.
#[path = "device_work_queue_contracts.rs"]
pub mod device_work_queue_contracts;

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

/// Integration tests from `tests/external_import_order.rs`.
#[path = "external_import_order.rs"]
pub mod external_import_order;

/// Integration tests from `tests/external_resource_path_agreement.rs`.
#[path = "external_resource_path_agreement.rs"]
pub mod external_resource_path_agreement;

/// Integration tests from `tests/extraction_memory_verifier_cost_model.rs`.
#[path = "extraction_memory_verifier_cost_model.rs"]
pub mod extraction_memory_verifier_cost_model;

/// Integration tests from `tests/float_lowering_refusal.rs`.
#[path = "float_lowering_refusal.rs"]
pub mod float_lowering_refusal;

/// Integration tests from `tests/fusion_contracts.rs`.
#[path = "fusion_contracts.rs"]
pub mod fusion_contracts;

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

/// Integration tests from `tests/http_proxy_redirect_policy.rs`.
#[path = "http_proxy_redirect_policy.rs"]
pub mod http_proxy_redirect_policy;

/// Integration tests from `tests/input_identity_contracts.rs`.
#[path = "input_identity_contracts.rs"]
pub mod input_identity_contracts;

/// Integration tests from `tests/intrinsic_registration_contract.rs`.
#[path = "intrinsic_registration_contract.rs"]
pub mod intrinsic_registration_contract;

/// Integration tests from `tests/launch_fusion_contracts.rs`.
#[path = "launch_fusion_contracts.rs"]
pub mod launch_fusion_contracts;

/// Integration tests from `tests/launch_grid_axis_folding.rs`.
#[path = "launch_grid_axis_folding.rs"]
pub mod launch_grid_axis_folding;

/// Integration tests from `tests/logical_markers_are_not_backend_operations.rs`.
#[path = "logical_markers_are_not_backend_operations.rs"]
pub mod logical_markers_are_not_backend_operations;

/// Integration tests from `tests/lock_poison_policy.rs`.
#[path = "lock_poison_policy.rs"]
pub mod lock_poison_policy;

/// Integration tests from `tests/megakernel_execution_contracts.rs`.
#[path = "megakernel_execution_contracts.rs"]
pub mod megakernel_execution_contracts;

/// Integration tests from `tests/mixed_work_autotuning.rs`.
#[path = "mixed_work_autotuning.rs"]
pub mod mixed_work_autotuning;

/// Integration tests from `tests/no_backend_crate_links_host_arithmetic.rs`.
#[path = "no_backend_crate_links_host_arithmetic.rs"]
pub mod no_backend_crate_links_host_arithmetic;

/// Integration tests from `tests/numeric_contracts.rs`.
#[path = "numeric_contracts.rs"]
pub mod numeric_contracts;

/// Integration tests from `tests/omitted_launch_geometry.rs`.
#[path = "omitted_launch_geometry.rs"]
pub mod omitted_launch_geometry;

/// Integration tests from `tests/ordering_contracts.rs`.
#[path = "ordering_contracts.rs"]
pub mod ordering_contracts;

/// Integration tests from `tests/output_slab_provenance.rs`.
#[path = "output_slab_provenance.rs"]
pub mod output_slab_provenance;

/// Integration tests from `tests/output_slots_contracts.rs`.
#[path = "output_slots_contracts.rs"]
pub mod output_slots_contracts;

/// Integration tests from `tests/param_inlining_contracts.rs`.
#[path = "param_inlining_contracts.rs"]
pub mod param_inlining_contracts;

/// Integration tests from `tests/persistent_contracts.rs`.
#[path = "persistent_contracts.rs"]
pub mod persistent_contracts;

/// Integration tests from `tests/pipeline_fusion_contracts.rs`.
#[path = "pipeline_fusion_contracts.rs"]
pub mod pipeline_fusion_contracts;

/// Integration tests from `tests/read_only_alias_contracts.rs`.
#[path = "read_only_alias_contracts.rs"]
pub mod read_only_alias_contracts;

/// Integration tests from `tests/reference_oracle_loses_to_a_device.rs`.
#[path = "reference_oracle_loses_to_a_device.rs"]
pub mod reference_oracle_loses_to_a_device;

/// Integration tests from `tests/registry_closure.rs`.
#[path = "registry_closure.rs"]
pub mod registry_closure;

/// Integration tests from `tests/rejection_wording_contract.rs`.
#[path = "rejection_wording_contract.rs"]
pub mod rejection_wording_contract;

/// Integration tests from `tests/release_publication_boundary.rs`.
#[path = "release_publication_boundary.rs"]
pub mod release_publication_boundary;

/// Integration tests from `tests/reservation_policy_contracts.rs`.
#[path = "reservation_policy_contracts.rs"]
pub mod reservation_policy_contracts;

/// Integration tests from `tests/resident_binding_projection.rs`.
#[path = "resident_binding_projection.rs"]
pub mod resident_binding_projection;

/// Integration tests from `tests/result_compaction_contracts.rs`.
#[path = "result_compaction_contracts.rs"]
pub mod result_compaction_contracts;

/// Integration tests from `tests/routing_registry_surface.rs`.
#[path = "routing_registry_surface.rs"]
pub mod routing_registry_surface;

/// Integration tests from `tests/runtime_watchdog_proofs.rs`.
#[path = "runtime_watchdog_proofs.rs"]
pub mod runtime_watchdog_proofs;

/// Integration tests from `tests/scan_graph_update_classifier_registry.rs`.
#[path = "scan_graph_update_classifier_registry.rs"]
pub mod scan_graph_update_classifier_registry;

/// Integration tests from `tests/shape_prediction_contracts.rs`.
#[path = "shape_prediction_contracts.rs"]
pub mod shape_prediction_contracts;

/// Integration tests from `tests/speculation_verdict_contracts.rs`.
#[path = "speculation_verdict_contracts.rs"]
pub mod speculation_verdict_contracts;

/// Integration tests from `tests/strategy_contracts.rs`.
#[path = "strategy_contracts.rs"]
pub mod strategy_contracts;

/// Integration tests from `tests/sweep_dispatch_shape_oracle_matrix.rs`.
#[path = "sweep_dispatch_shape_oracle_matrix.rs"]
pub mod sweep_dispatch_shape_oracle_matrix;

/// Integration tests from `tests/sweep_numeric_oracle_matrix.rs`.
#[path = "sweep_numeric_oracle_matrix.rs"]
pub mod sweep_numeric_oracle_matrix;

/// Integration tests from `tests/target_contract.rs`.
#[path = "target_contract.rs"]
pub mod target_contract;

/// Integration tests from `tests/trace_context_telemetry_contracts.rs`.
#[path = "trace_context_telemetry_contracts.rs"]
pub mod trace_context_telemetry_contracts;

/// Integration tests from `tests/trace_jit_policy_contracts.rs`.
#[path = "trace_jit_policy_contracts.rs"]
pub mod trace_jit_policy_contracts;

/// Integration tests from `tests/transfer_accounting_contracts.rs`.
#[path = "transfer_accounting_contracts.rs"]
pub mod transfer_accounting_contracts;

/// Integration tests from `tests/unreported_device_facts.rs`.
#[path = "unreported_device_facts.rs"]
pub mod unreported_device_facts;

/// Integration tests from `tests/vyre_backend_forwarding_closure.rs`.
#[path = "vyre_backend_forwarding_closure.rs"]
pub mod vyre_backend_forwarding_closure;

/// Integration tests for domain-neutral resource ABI from `tests/resource_abi_contracts.rs`.
#[path = "resource_abi_contracts.rs"]
pub mod resource_abi_contracts;

/// Integration tests from `tests/lock_policy_closure.rs`.
#[path = "lock_policy_closure.rs"]
pub mod lock_policy_closure;

/// Integration tests from `tests/support_certificate_join.rs`.
#[path = "support_certificate_join.rs"]
pub mod support_certificate_join;

/// Integration tests from `tests/concrete_driver_dependency_boundaries.rs`.
#[path = "concrete_driver_dependency_boundaries.rs"]
pub mod concrete_driver_dependency_boundaries;

/// Integration tests from `tests/target_facet_lowering_arms.rs`.
#[path = "target_facet_lowering_arms.rs"]
pub mod target_facet_lowering_arms;
