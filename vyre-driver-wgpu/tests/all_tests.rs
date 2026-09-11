//! One binary for every default-feature integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Shared fixture module from `tests/binop_parity_fixtures.rs`.
#[cfg(feature = "device-tests")]
#[allow(
    clippy::assertions_on_constants,
    clippy::filter_map_bool_then,
    clippy::needless_range_loop,
    clippy::unnecessary_map_or,
    deprecated,
    missing_docs
)]
#[path = "binop_parity_fixtures.rs"]
pub mod binop_parity_fixtures;

/// Shared fixture module from `tests/harness/mod.rs`.
#[cfg(feature = "device-tests")]
#[allow(
    clippy::assertions_on_constants,
    clippy::filter_map_bool_then,
    clippy::needless_range_loop,
    clippy::unnecessary_map_or,
    deprecated,
    missing_docs
)]
#[path = "harness/mod.rs"]
pub mod harness;

/// Integration tests from `tests/_probe_matmul_wgsl.rs`.
#[cfg(feature = "device-tests")]
#[path = "_probe_matmul_wgsl.rs"]
pub mod _probe_matmul_wgsl;

/// Integration tests from `tests/adapter_limits_not_defaults.rs`.
#[cfg(feature = "device-tests")]
#[path = "adapter_limits_not_defaults.rs"]
pub mod adapter_limits_not_defaults;

/// Integration tests from `tests/adler32_gpu_parity.rs`.
#[cfg(feature = "device-tests")]
#[path = "adler32_gpu_parity.rs"]
pub mod adler32_gpu_parity;

/// Integration tests from `tests/async_capability_innovation.rs`.
#[cfg(feature = "device-tests")]
#[path = "async_capability_innovation.rs"]
pub mod async_capability_innovation;

/// Integration tests from `tests/async_dispatch_contract.rs`.
#[cfg(feature = "device-tests")]
#[path = "async_dispatch_contract.rs"]
pub mod async_dispatch_contract;

/// Integration tests from `tests/async_dispatch_non_blocking.rs`.
#[cfg(feature = "device-tests")]
#[path = "async_dispatch_non_blocking.rs"]
pub mod async_dispatch_non_blocking;

/// Integration tests from `tests/async_transfer_byte_span_parity.rs`.
#[cfg(feature = "device-tests")]
#[path = "async_transfer_byte_span_parity.rs"]
pub mod async_transfer_byte_span_parity;

/// Integration tests from `tests/binding_layout_drift.rs`.
#[cfg(feature = "device-tests")]
#[path = "binding_layout_drift.rs"]
pub mod binding_layout_drift;

/// Integration tests from `tests/bitset_zero_gpu_parity.rs`.
#[cfg(feature = "device-tests")]
#[path = "bitset_zero_gpu_parity.rs"]
pub mod bitset_zero_gpu_parity;

/// Integration tests from `tests/blake3_compress_gpu_parity.rs`.
#[cfg(feature = "device-tests")]
#[path = "blake3_compress_gpu_parity.rs"]
pub mod blake3_compress_gpu_parity;

/// Integration tests from `tests/buf_len_array_length/mod.rs`.
#[cfg(feature = "device-tests")]
#[path = "buf_len_array_length/mod.rs"]
pub mod buf_len_array_length;

/// Integration tests from `tests/capability_contract.rs`.
#[cfg(feature = "device-tests")]
#[path = "capability_contract.rs"]
pub mod capability_contract;

/// Integration tests from `tests/capability_drift.rs`.
#[cfg(feature = "device-tests")]
#[path = "capability_drift.rs"]
pub mod capability_drift;

/// Integration tests from `tests/cat_a_conform.rs`.
#[cfg(feature = "device-tests")]
#[path = "cat_a_conform.rs"]
pub mod cat_a_conform;

/// Integration tests from `tests/cat_a_gpu_differential.rs`.
#[cfg(feature = "device-tests")]
#[allow(deprecated)]
#[path = "cat_a_gpu_differential.rs"]
pub mod cat_a_gpu_differential;

/// Integration tests from `tests/cli_contract.rs`.
#[cfg(feature = "device-tests")]
#[path = "cli_contract.rs"]
pub mod cli_contract;

/// Integration tests from `tests/connected_graph_wgpu_contracts.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "connected_graph_wgpu_contracts.rs"]
pub mod connected_graph_wgpu_contracts;

/// Integration tests from `tests/crc32_gpu_parity.rs`.
#[cfg(feature = "device-tests")]
#[path = "crc32_gpu_parity.rs"]
pub mod crc32_gpu_parity;

/// Integration tests from `tests/decode_hex_gpu_parity.rs`.
#[cfg(feature = "device-tests")]
#[allow(deprecated)]
#[path = "decode_hex_gpu_parity.rs"]
pub mod decode_hex_gpu_parity;

/// Integration tests from `tests/default_workgroup_contract.rs`.
#[cfg(feature = "device-tests")]
#[path = "default_workgroup_contract.rs"]
pub mod default_workgroup_contract;

/// Integration tests from `tests/determinism_contract.rs`.
#[cfg(feature = "device-tests")]
#[path = "determinism_contract.rs"]
pub mod determinism_contract;

/// Integration tests from `tests/device_lost_recovery.rs`.
#[cfg(feature = "device-tests")]
#[path = "device_lost_recovery.rs"]
pub mod device_lost_recovery;

/// Integration tests from `tests/differential_fuzz.rs`.
#[cfg(feature = "device-tests")]
#[allow(deprecated)]
#[path = "differential_fuzz.rs"]
pub mod differential_fuzz;

/// Integration tests from `tests/dispatch_adversarial.rs`.
#[cfg(feature = "device-tests")]
#[path = "dispatch_adversarial.rs"]
pub mod dispatch_adversarial;

/// Integration tests from `tests/dispatch_allocation_contract.rs`.
#[cfg(feature = "device-tests")]
#[allow(missing_docs)]
#[path = "dispatch_allocation_contract.rs"]
pub mod dispatch_allocation_contract;

/// Integration tests from `tests/dispatch_async_deferred.rs`.
#[cfg(feature = "device-tests")]
#[path = "dispatch_async_deferred.rs"]
pub mod dispatch_async_deferred;

/// Integration tests from `tests/dispatch_grid_shape_contract.rs`.
#[cfg(feature = "device-tests")]
#[path = "dispatch_grid_shape_contract.rs"]
pub mod dispatch_grid_shape_contract;

/// Integration tests from `tests/dispatch_hot_path.rs`.
#[cfg(feature = "device-tests")]
#[path = "dispatch_hot_path.rs"]
pub mod dispatch_hot_path;

/// Integration tests from `tests/dispatch_never_cpu_fallback.rs`.
#[cfg(feature = "device-tests")]
#[path = "dispatch_never_cpu_fallback.rs"]
pub mod dispatch_never_cpu_fallback;

/// Integration tests from `tests/dispatch_preemption.rs`.
#[cfg(feature = "device-tests")]
#[path = "dispatch_preemption.rs"]
pub mod dispatch_preemption;

/// Integration tests from `tests/div_zero_shift_mask_parity.rs`.
#[cfg(feature = "device-tests")]
#[path = "div_zero_shift_mask_parity.rs"]
pub mod div_zero_shift_mask_parity;

/// Integration tests from `tests/emitted_wgsl_byte_stability.rs`.
#[path = "emitted_wgsl_byte_stability.rs"]
pub mod emitted_wgsl_byte_stability;

/// Integration tests from `tests/every_op_random_inputs.rs`.
#[cfg(feature = "device-tests")]
#[allow(clippy::filter_map_bool_then, clippy::unnecessary_map_or, deprecated)]
#[path = "every_op_random_inputs.rs"]
pub mod every_op_random_inputs;

/// Integration tests from `tests/f32_no_contraction_contract.rs`.
#[cfg(feature = "device-tests")]
#[path = "f32_no_contraction_contract.rs"]
pub mod f32_no_contraction_contract;

/// Integration tests from `tests/float_to_int_cast_parity.rs`.
#[cfg(feature = "device-tests")]
#[path = "float_to_int_cast_parity.rs"]
pub mod float_to_int_cast_parity;

/// Integration tests from `tests/fnv1a32_gpu_parity.rs`.
#[cfg(feature = "device-tests")]
#[path = "fnv1a32_gpu_parity.rs"]
pub mod fnv1a32_gpu_parity;

/// Integration tests from `tests/fnv1a64_gpu_parity.rs`.
#[cfg(feature = "device-tests")]
#[path = "fnv1a64_gpu_parity.rs"]
pub mod fnv1a64_gpu_parity;

/// Integration tests from `tests/gap_transcendentals_parity.rs`.
#[path = "gap_transcendentals_parity.rs"]
pub mod gap_transcendentals_parity;

/// Integration tests from `tests/input_abi_contracts.rs`.
#[path = "input_abi_contracts.rs"]
pub mod input_abi_contracts;

/// Integration tests from `tests/hit_buffer.rs`.
#[cfg(feature = "device-tests")]
#[allow(deprecated)]
#[path = "hit_buffer.rs"]
pub mod hit_buffer;

/// Integration tests from `tests/lens_gpu_parity.rs`.
#[cfg(feature = "device-tests")]
#[path = "lens_gpu_parity.rs"]
pub mod lens_gpu_parity;

/// Integration tests from `tests/limits_from_adapter_device.rs`.
#[cfg(feature = "device-tests")]
#[path = "limits_from_adapter_device.rs"]
pub mod limits_from_adapter_device;

/// Integration tests from `tests/live_capability_honesty.rs`.
#[cfg(feature = "device-tests")]
#[allow(clippy::assertions_on_constants)]
#[path = "live_capability_honesty.rs"]
pub mod live_capability_honesty;

/// Integration tests from `tests/loader_instance_reuse.rs`.
#[cfg(feature = "device-tests")]
#[path = "loader_instance_reuse.rs"]
pub mod loader_instance_reuse;

/// Integration tests from `tests/loop_carrier_three_level_if_real_dispatch.rs`.
#[cfg(feature = "device-tests")]
#[path = "loop_carrier_three_level_if_real_dispatch.rs"]
pub mod loop_carrier_three_level_if_real_dispatch;

/// Integration tests from `tests/lowering_actionable_errors.rs`.
#[cfg(feature = "device-tests")]
#[path = "lowering_actionable_errors.rs"]
pub mod lowering_actionable_errors;

/// Integration tests from `tests/naga_deeper_regressions.rs`.
#[cfg(feature = "device-tests")]
#[path = "naga_deeper_regressions.rs"]
pub mod naga_deeper_regressions;

/// Integration tests from `tests/naga_findings_followup.rs`.
#[cfg(feature = "device-tests")]
#[path = "naga_findings_followup.rs"]
pub mod naga_findings_followup;

/// Integration tests from `tests/naga_loop_region_followup.rs`.
#[cfg(feature = "device-tests")]
#[path = "naga_loop_region_followup.rs"]
pub mod naga_loop_region_followup;

/// Integration tests from `tests/naga_type_buffer_followup.rs`.
#[cfg(feature = "device-tests")]
#[path = "naga_type_buffer_followup.rs"]
pub mod naga_type_buffer_followup;

/// Integration tests from `tests/narrowing_cast_parity.rs`.
#[cfg(feature = "device-tests")]
#[path = "narrowing_cast_parity.rs"]
pub mod narrowing_cast_parity;

/// Integration tests from `tests/newton_schulz_ir_shape.rs`.
#[cfg(feature = "device-tests")]
#[path = "newton_schulz_ir_shape.rs"]
pub mod newton_schulz_ir_shape;

/// Integration tests from `tests/no_cpu_fallback.rs`.
#[cfg(feature = "device-tests")]
#[allow(clippy::needless_range_loop)]
#[path = "no_cpu_fallback.rs"]
pub mod no_cpu_fallback;

/// Integration tests from `tests/nvme_gpu_ingest_e2e.rs`.
#[path = "nvme_gpu_ingest_e2e.rs"]
pub mod nvme_gpu_ingest_e2e;

/// Integration tests from `tests/op_pairwise/mod.rs`.
#[cfg(feature = "device-tests")]
#[allow(deprecated)]
#[path = "op_pairwise/mod.rs"]
pub mod op_pairwise;

/// Integration tests from `tests/oversized_workgroup_fails_loudly.rs`.
#[cfg(feature = "device-tests")]
#[path = "oversized_workgroup_fails_loudly.rs"]
pub mod oversized_workgroup_fails_loudly;

/// Integration tests from `tests/pipeline_cache_contract.rs`.
#[cfg(feature = "device-tests")]
#[allow(missing_docs)]
#[path = "pipeline_cache_contract.rs"]
pub mod pipeline_cache_contract;

/// Integration tests from `tests/pipeline_cache_persistence.rs`.
#[cfg(feature = "device-tests")]
#[path = "pipeline_cache_persistence.rs"]
pub mod pipeline_cache_persistence;

/// Integration tests from `tests/preferred_dispatch_backend.rs`.
#[cfg(feature = "device-tests")]
#[allow(deprecated)]
#[path = "preferred_dispatch_backend.rs"]
pub mod preferred_dispatch_backend;

/// Integration tests from `tests/readback_ring_liveness_contracts.rs`.
#[cfg(feature = "device-tests")]
#[path = "readback_ring_liveness_contracts.rs"]
pub mod readback_ring_liveness_contracts;

/// Integration tests from `tests/resident_buffer_contracts/mod.rs`.
#[cfg(feature = "device-tests")]
#[path = "resident_buffer_contracts/mod.rs"]
pub mod resident_buffer_contracts;

/// Integration tests from `tests/resident_grid_sync_contracts.rs`.
#[cfg(feature = "device-tests")]
#[path = "resident_grid_sync_contracts.rs"]
pub mod resident_grid_sync_contracts;

/// Integration tests from `tests/resident_output_contracts.rs`.
#[cfg(feature = "device-tests")]
#[path = "resident_output_contracts.rs"]
pub mod resident_output_contracts;

/// Integration tests from `tests/resident_timed_outputs.rs`.
#[cfg(feature = "device-tests")]
#[path = "resident_timed_outputs.rs"]
pub mod resident_timed_outputs;

/// Integration tests from `tests/resident_work_queue_emit.rs`.
#[cfg(feature = "device-tests")]
#[path = "resident_work_queue_emit.rs"]
pub mod resident_work_queue_emit;

/// Integration tests from `tests/runtime_indirect_contracts.rs`.
#[cfg(feature = "device-tests")]
#[path = "runtime_indirect_contracts.rs"]
pub mod runtime_indirect_contracts;

/// Integration tests from `tests/runtime_router_contracts.rs`.
#[cfg(feature = "device-tests")]
#[path = "runtime_router_contracts.rs"]
pub mod runtime_router_contracts;

/// Integration tests from `tests/same_width_store_parity.rs`.
#[cfg(feature = "device-tests")]
#[path = "same_width_store_parity.rs"]
pub mod same_width_store_parity;

/// Integration tests from `tests/self_optimizer_canonicalize_e2e.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "self_optimizer_canonicalize_e2e.rs"]
pub mod self_optimizer_canonicalize_e2e;

/// Integration tests from `tests/self_optimizer_const_fold_e2e.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "self_optimizer_const_fold_e2e.rs"]
pub mod self_optimizer_const_fold_e2e;

/// Integration tests from `tests/self_optimizer_dce_e2e.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "self_optimizer_dce_e2e.rs"]
pub mod self_optimizer_dce_e2e;

/// Integration tests from `tests/self_optimizer_pattern_match_e2e.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "self_optimizer_pattern_match_e2e.rs"]
pub mod self_optimizer_pattern_match_e2e;

/// Integration tests from `tests/self_optimizer_pipeline_e2e.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "self_optimizer_pipeline_e2e.rs"]
pub mod self_optimizer_pipeline_e2e;

/// Integration tests from `tests/self_optimizer_scaling_bench.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "self_optimizer_scaling_bench.rs"]
pub mod self_optimizer_scaling_bench;

/// Integration tests from `tests/semantic_execution.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "semantic_execution.rs"]
pub mod semantic_execution;

/// Integration tests from `tests/shared_backend_contract.rs`.
#[cfg(feature = "device-tests")]
#[path = "shared_backend_contract.rs"]
pub mod shared_backend_contract;

/// Integration tests from `tests/signed_int_op_parity.rs`.
#[cfg(feature = "device-tests")]
#[path = "signed_int_op_parity.rs"]
pub mod signed_int_op_parity;

/// Integration tests from `tests/signed_modulo_parity.rs`.
#[cfg(feature = "device-tests")]
#[path = "signed_modulo_parity.rs"]
pub mod signed_modulo_parity;

/// Integration tests from `tests/sinkhorn_iterate_contract.rs`.
#[cfg(feature = "device-tests")]
#[path = "sinkhorn_iterate_contract.rs"]
pub mod sinkhorn_iterate_contract;

/// Integration tests from `tests/stream_shard_public_error_contracts.rs`.
#[cfg(feature = "device-tests")]
#[path = "stream_shard_public_error_contracts.rs"]
pub mod stream_shard_public_error_contracts;

/// Integration tests from `tests/subgroup_detection.rs`.
#[cfg(feature = "device-tests")]
#[path = "subgroup_detection.rs"]
pub mod subgroup_detection;

/// Integration tests from `tests/subgroup_reporting_honesty.rs`.
#[cfg(feature = "device-tests")]
#[path = "subgroup_reporting_honesty.rs"]
pub mod subgroup_reporting_honesty;

/// Integration tests from `tests/synthetic_binop_parity.rs`.
#[cfg(feature = "device-tests")]
#[path = "synthetic_binop_parity.rs"]
pub mod synthetic_binop_parity;

/// Integration tests from `tests/target_compiler.rs`.
#[cfg(feature = "device-tests")]
#[path = "target_compiler.rs"]
pub mod target_compiler;

/// Integration tests from `tests/timed_dispatch_device_ns.rs`.
#[cfg(feature = "device-tests")]
#[path = "timed_dispatch_device_ns.rs"]
pub mod timed_dispatch_device_ns;

/// Integration tests from `tests/transcendentals_parity.rs`.
#[path = "transcendentals_parity.rs"]
pub mod transcendentals_parity;

/// Integration tests from `tests/trap_propagation.rs`.
#[cfg(feature = "device-tests")]
#[path = "trap_propagation.rs"]
pub mod trap_propagation;

/// Integration tests from `tests/trap_sidecar.rs`.
#[cfg(feature = "device-tests")]
#[path = "trap_sidecar.rs"]
pub mod trap_sidecar;

/// Integration tests from `tests/u32_wrap_arithmetic.rs`.
#[cfg(feature = "device-tests")]
#[path = "u32_wrap_arithmetic.rs"]
pub mod u32_wrap_arithmetic;

/// Integration tests from `tests/unary_int_parity.rs`.
#[cfg(feature = "device-tests")]
#[path = "unary_int_parity.rs"]
pub mod unary_int_parity;

/// Integration tests from `tests/validation_cross_backend.rs`.
#[cfg(feature = "device-tests")]
#[path = "validation_cross_backend.rs"]
pub mod validation_cross_backend;

/// Integration tests from `tests/wgpu_command_reuse_classifier.rs`.
#[cfg(feature = "device-tests")]
#[path = "wgpu_command_reuse_classifier.rs"]
pub mod wgpu_command_reuse_classifier;

/// Integration tests from `tests/wgpu_subgroup_capability_diagnostics.rs`.
#[cfg(feature = "device-tests")]
#[path = "wgpu_subgroup_capability_diagnostics.rs"]
pub mod wgpu_subgroup_capability_diagnostics;

/// Integration tests from `tests/wgpu_subgroup_scan_plan_registry.rs`.
#[cfg(feature = "device-tests")]
#[path = "wgpu_subgroup_scan_plan_registry.rs"]
pub mod wgpu_subgroup_scan_plan_registry;

/// Integration tests from `tests/wgsl_scan_uniformity_certificates.rs`.
#[cfg(feature = "device-tests")]
#[path = "wgsl_scan_uniformity_certificates.rs"]
pub mod wgsl_scan_uniformity_certificates;

/// Integration tests from `tests/external_resource_wgpu_contracts.rs`.
#[path = "external_resource_wgpu_contracts.rs"]
pub mod external_resource_wgpu_contracts;
/// Integration tests from `tests/widening_cast_64_parity.rs`.
#[cfg(feature = "device-tests")]
#[path = "widening_cast_64_parity.rs"]
pub mod widening_cast_64_parity;
