# Testing `vyre-driver-wgpu`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-driver-wgpu
```

Own pure WGSL target compilation, portable GPU acquisition, materialization, dispatch, graph execution, and backend evidence.

The crate lives at `vyre-driver-wgpu`. The `portable-driver` owner maintains its
`concrete-backend` testing contract.

## Commands

```console
./cargo_full test -p vyre-driver-wgpu
```

```console
./cargo_full test -p vyre-driver-wgpu --all-features
```

```console
./cargo_full test -p vyre-driver-wgpu -- --ignored --nocapture
```

## Feature sets

- Default feature members: None
- Available manifest features: `default`, `math-linalg`, `math-scan`, `nn-attention`, `parity-testing`, `pattern-dfa`, `pattern-nfa`, `pattern-substring`, `wgpu`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `bin` | `vyre` | `vyre-driver-wgpu/src/bin/vyre.rs` | None | `./cargo_full test -p vyre-driver-wgpu --bin vyre` |
| `bin` | `vyre-wgpu` | `vyre-driver-wgpu/src/bin/vyre.rs` | None | `./cargo_full test -p vyre-driver-wgpu --bin vyre-wgpu` |
| `example` | `wgpu_release_surface` | `vyre-driver-wgpu/examples/wgpu_release_surface.rs` | None | `./cargo_full test -p vyre-driver-wgpu --example wgpu_release_surface` |
| `lib` | `vyre_driver_wgpu` | `vyre-driver-wgpu/src/lib.rs` | None | `./cargo_full test -p vyre-driver-wgpu` |
| `test` | `_probe_matmul_wgsl` | `vyre-driver-wgpu/tests/_probe_matmul_wgsl.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test _probe_matmul_wgsl` |
| `test` | `adapter_limits_not_defaults` | `vyre-driver-wgpu/tests/adapter_limits_not_defaults.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test adapter_limits_not_defaults` |
| `test` | `adler32_gpu_parity` | `vyre-driver-wgpu/tests/adler32_gpu_parity.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test adler32_gpu_parity` |
| `test` | `async_capability_innovation` | `vyre-driver-wgpu/tests/async_capability_innovation.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test async_capability_innovation` |
| `test` | `async_dispatch_contract` | `vyre-driver-wgpu/tests/async_dispatch_contract.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test async_dispatch_contract` |
| `test` | `async_dispatch_non_blocking` | `vyre-driver-wgpu/tests/async_dispatch_non_blocking.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test async_dispatch_non_blocking` |
| `test` | `binding_layout_drift` | `vyre-driver-wgpu/tests/binding_layout_drift.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test binding_layout_drift` |
| `test` | `binop_parity_fixtures` | `vyre-driver-wgpu/tests/binop_parity_fixtures.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test binop_parity_fixtures` |
| `test` | `bitset_zero_gpu_parity` | `vyre-driver-wgpu/tests/bitset_zero_gpu_parity.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test bitset_zero_gpu_parity` |
| `test` | `blake3_compress_gpu_parity` | `vyre-driver-wgpu/tests/blake3_compress_gpu_parity.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test blake3_compress_gpu_parity` |
| `test` | `buf_len_array_length` | `vyre-driver-wgpu/tests/buf_len_array_length/mod.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test buf_len_array_length` |
| `test` | `capability_contract` | `vyre-driver-wgpu/tests/capability_contract.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test capability_contract` |
| `test` | `capability_drift` | `vyre-driver-wgpu/tests/capability_drift.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test capability_drift` |
| `test` | `cat_a_conform` | `vyre-driver-wgpu/tests/cat_a_conform.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test cat_a_conform` |
| `test` | `cat_a_gpu_differential` | `vyre-driver-wgpu/tests/cat_a_gpu_differential.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test cat_a_gpu_differential` |
| `test` | `cli_contract` | `vyre-driver-wgpu/tests/cli_contract.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test cli_contract` |
| `test` | `crc32_gpu_parity` | `vyre-driver-wgpu/tests/crc32_gpu_parity.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test crc32_gpu_parity` |
| `test` | `decode_hex_gpu_parity` | `vyre-driver-wgpu/tests/decode_hex_gpu_parity.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test decode_hex_gpu_parity` |
| `test` | `default_workgroup_contract` | `vyre-driver-wgpu/tests/default_workgroup_contract.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test default_workgroup_contract` |
| `test` | `determinism_contract` | `vyre-driver-wgpu/tests/determinism_contract.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test determinism_contract` |
| `test` | `device_lost_recovery` | `vyre-driver-wgpu/tests/device_lost_recovery.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test device_lost_recovery` |
| `test` | `differential_fuzz` | `vyre-driver-wgpu/tests/differential_fuzz.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test differential_fuzz` |
| `test` | `dispatch_adversarial` | `vyre-driver-wgpu/tests/dispatch_adversarial.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test dispatch_adversarial` |
| `test` | `dispatch_allocation_contract` | `vyre-driver-wgpu/tests/dispatch_allocation_contract.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test dispatch_allocation_contract` |
| `test` | `dispatch_async_deferred` | `vyre-driver-wgpu/tests/dispatch_async_deferred.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test dispatch_async_deferred` |
| `test` | `dispatch_grid_shape_contract` | `vyre-driver-wgpu/tests/dispatch_grid_shape_contract.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test dispatch_grid_shape_contract` |
| `test` | `dispatch_hot_path` | `vyre-driver-wgpu/tests/dispatch_hot_path.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test dispatch_hot_path` |
| `test` | `dispatch_never_cpu_fallback` | `vyre-driver-wgpu/tests/dispatch_never_cpu_fallback.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test dispatch_never_cpu_fallback` |
| `test` | `dispatch_preemption` | `vyre-driver-wgpu/tests/dispatch_preemption.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test dispatch_preemption` |
| `test` | `div_zero_shift_mask_parity` | `vyre-driver-wgpu/tests/div_zero_shift_mask_parity.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test div_zero_shift_mask_parity` |
| `test` | `emitted_wgsl_byte_stability` | `vyre-driver-wgpu/tests/emitted_wgsl_byte_stability.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test emitted_wgsl_byte_stability` |
| `test` | `every_op_random_inputs` | `vyre-driver-wgpu/tests/every_op_random_inputs.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test every_op_random_inputs` |
| `test` | `f32_no_contraction_contract` | `vyre-driver-wgpu/tests/f32_no_contraction_contract.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test f32_no_contraction_contract` |
| `test` | `float_to_int_cast_parity` | `vyre-driver-wgpu/tests/float_to_int_cast_parity.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test float_to_int_cast_parity` |
| `test` | `fnv1a32_gpu_parity` | `vyre-driver-wgpu/tests/fnv1a32_gpu_parity.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test fnv1a32_gpu_parity` |
| `test` | `fnv1a64_gpu_parity` | `vyre-driver-wgpu/tests/fnv1a64_gpu_parity.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test fnv1a64_gpu_parity` |
| `test` | `gap_transcendentals_parity` | `vyre-driver-wgpu/tests/gap_transcendentals_parity.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test gap_transcendentals_parity` |
| `test` | `hit_buffer` | `vyre-driver-wgpu/tests/hit_buffer.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test hit_buffer` |
| `test` | `lens_gpu_parity` | `vyre-driver-wgpu/tests/lens_gpu_parity.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test lens_gpu_parity` |
| `test` | `limits_from_adapter_device` | `vyre-driver-wgpu/tests/limits_from_adapter_device.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test limits_from_adapter_device` |
| `test` | `live_capability_honesty` | `vyre-driver-wgpu/tests/live_capability_honesty.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test live_capability_honesty` |
| `test` | `loop_carrier_three_level_if_real_dispatch` | `vyre-driver-wgpu/tests/loop_carrier_three_level_if_real_dispatch.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test loop_carrier_three_level_if_real_dispatch` |
| `test` | `lowering_actionable_errors` | `vyre-driver-wgpu/tests/lowering_actionable_errors.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test lowering_actionable_errors` |
| `test` | `naga_deeper_regressions` | `vyre-driver-wgpu/tests/naga_deeper_regressions.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test naga_deeper_regressions` |
| `test` | `naga_findings_followup` | `vyre-driver-wgpu/tests/naga_findings_followup.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test naga_findings_followup` |
| `test` | `naga_loop_region_followup` | `vyre-driver-wgpu/tests/naga_loop_region_followup.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test naga_loop_region_followup` |
| `test` | `naga_type_buffer_followup` | `vyre-driver-wgpu/tests/naga_type_buffer_followup.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test naga_type_buffer_followup` |
| `test` | `narrowing_cast_parity` | `vyre-driver-wgpu/tests/narrowing_cast_parity.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test narrowing_cast_parity` |
| `test` | `newton_schulz_ir_shape` | `vyre-driver-wgpu/tests/newton_schulz_ir_shape.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test newton_schulz_ir_shape` |
| `test` | `no_cpu_fallback` | `vyre-driver-wgpu/tests/no_cpu_fallback.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test no_cpu_fallback` |
| `test` | `nvme_gpu_ingest_e2e` | `vyre-driver-wgpu/tests/nvme_gpu_ingest_e2e.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test nvme_gpu_ingest_e2e` |
| `test` | `op_pairwise` | `vyre-driver-wgpu/tests/op_pairwise/mod.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test op_pairwise` |
| `test` | `oversized_workgroup_fails_loudly` | `vyre-driver-wgpu/tests/oversized_workgroup_fails_loudly.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test oversized_workgroup_fails_loudly` |
| `test` | `pipeline_cache_contract` | `vyre-driver-wgpu/tests/pipeline_cache_contract.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test pipeline_cache_contract` |
| `test` | `pipeline_cache_persistence` | `vyre-driver-wgpu/tests/pipeline_cache_persistence.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test pipeline_cache_persistence` |
| `test` | `preferred_dispatch_backend` | `vyre-driver-wgpu/tests/preferred_dispatch_backend.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test preferred_dispatch_backend` |
| `test` | `readback_ring_liveness_contracts` | `vyre-driver-wgpu/tests/readback_ring_liveness_contracts.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test readback_ring_liveness_contracts` |
| `test` | `resident_buffer_contracts` | `vyre-driver-wgpu/tests/resident_buffer_contracts/mod.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test resident_buffer_contracts` |
| `test` | `resident_grid_sync_contracts` | `vyre-driver-wgpu/tests/resident_grid_sync_contracts.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test resident_grid_sync_contracts` |
| `test` | `resident_output_contracts` | `vyre-driver-wgpu/tests/resident_output_contracts.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test resident_output_contracts` |
| `test` | `resident_timed_outputs` | `vyre-driver-wgpu/tests/resident_timed_outputs.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test resident_timed_outputs` |
| `test` | `resident_work_queue_emit` | `vyre-driver-wgpu/tests/resident_work_queue_emit.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test resident_work_queue_emit` |
| `test` | `runtime_indirect_contracts` | `vyre-driver-wgpu/tests/runtime_indirect_contracts.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test runtime_indirect_contracts` |
| `test` | `runtime_router_contracts` | `vyre-driver-wgpu/tests/runtime_router_contracts.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test runtime_router_contracts` |
| `test` | `same_width_store_parity` | `vyre-driver-wgpu/tests/same_width_store_parity.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test same_width_store_parity` |
| `test` | `self_optimizer_canonicalize_e2e` | `vyre-driver-wgpu/tests/self_optimizer_canonicalize_e2e.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test self_optimizer_canonicalize_e2e` |
| `test` | `self_optimizer_const_fold_e2e` | `vyre-driver-wgpu/tests/self_optimizer_const_fold_e2e.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test self_optimizer_const_fold_e2e` |
| `test` | `self_optimizer_dce_e2e` | `vyre-driver-wgpu/tests/self_optimizer_dce_e2e.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test self_optimizer_dce_e2e` |
| `test` | `self_optimizer_pattern_match_e2e` | `vyre-driver-wgpu/tests/self_optimizer_pattern_match_e2e.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test self_optimizer_pattern_match_e2e` |
| `test` | `self_optimizer_pipeline_e2e` | `vyre-driver-wgpu/tests/self_optimizer_pipeline_e2e.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test self_optimizer_pipeline_e2e` |
| `test` | `self_optimizer_scaling_bench` | `vyre-driver-wgpu/tests/self_optimizer_scaling_bench.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test self_optimizer_scaling_bench` |
| `test` | `shared_backend_contract` | `vyre-driver-wgpu/tests/shared_backend_contract.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test shared_backend_contract` |
| `test` | `signed_int_op_parity` | `vyre-driver-wgpu/tests/signed_int_op_parity.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test signed_int_op_parity` |
| `test` | `signed_modulo_parity` | `vyre-driver-wgpu/tests/signed_modulo_parity.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test signed_modulo_parity` |
| `test` | `sinkhorn_iterate_contract` | `vyre-driver-wgpu/tests/sinkhorn_iterate_contract.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test sinkhorn_iterate_contract` |
| `test` | `spirv_backend_contracts` | `vyre-driver-wgpu/tests/spirv_backend_contracts.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test spirv_backend_contracts` |
| `test` | `stream_shard_public_error_contracts` | `vyre-driver-wgpu/tests/stream_shard_public_error_contracts.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test stream_shard_public_error_contracts` |
| `test` | `subgroup_detection` | `vyre-driver-wgpu/tests/subgroup_detection.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test subgroup_detection` |
| `test` | `subgroup_reporting_honesty` | `vyre-driver-wgpu/tests/subgroup_reporting_honesty.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test subgroup_reporting_honesty` |
| `test` | `synthetic_binop_parity` | `vyre-driver-wgpu/tests/synthetic_binop_parity.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test synthetic_binop_parity` |
| `test` | `target_compiler` | `vyre-driver-wgpu/tests/target_compiler.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test target_compiler` |
| `test` | `timed_dispatch_device_ns` | `vyre-driver-wgpu/tests/timed_dispatch_device_ns.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test timed_dispatch_device_ns` |
| `test` | `transcendentals_parity` | `vyre-driver-wgpu/tests/transcendentals_parity.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test transcendentals_parity` |
| `test` | `trap_propagation` | `vyre-driver-wgpu/tests/trap_propagation.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test trap_propagation` |
| `test` | `trap_sidecar` | `vyre-driver-wgpu/tests/trap_sidecar.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test trap_sidecar` |
| `test` | `u32_wrap_arithmetic` | `vyre-driver-wgpu/tests/u32_wrap_arithmetic.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test u32_wrap_arithmetic` |
| `test` | `unary_int_parity` | `vyre-driver-wgpu/tests/unary_int_parity.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test unary_int_parity` |
| `test` | `validation_cross_backend` | `vyre-driver-wgpu/tests/validation_cross_backend.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test validation_cross_backend` |
| `test` | `wgpu_command_reuse_classifier` | `vyre-driver-wgpu/tests/wgpu_command_reuse_classifier.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test wgpu_command_reuse_classifier` |
| `test` | `wgpu_subgroup_capability_diagnostics` | `vyre-driver-wgpu/tests/wgpu_subgroup_capability_diagnostics.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test wgpu_subgroup_capability_diagnostics` |
| `test` | `wgpu_subgroup_scan_plan_registry` | `vyre-driver-wgpu/tests/wgpu_subgroup_scan_plan_registry.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test wgpu_subgroup_scan_plan_registry` |
| `test` | `wgsl_scan_uniformity_certificates` | `vyre-driver-wgpu/tests/wgsl_scan_uniformity_certificates.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test wgsl_scan_uniformity_certificates` |
| `test` | `widening_cast_64_parity` | `vyre-driver-wgpu/tests/widening_cast_64_parity.rs` | None | `./cargo_full test -p vyre-driver-wgpu --test widening_cast_64_parity` |

## Test classes

- Device and capability contracts
- Lowering and artifact semantics
- Dispatch, graph, memory, and backend parity tests

## Hardware requirements

You need a supported physical GPU adapter on the execution host (axiomexec) for device dispatch and ignored physical-adapter tests. A requested adapter that cannot initialize is an error.

## Evidence outputs

- `release/evidence/conformance/release-all-backends-certificate.json`
- Command status and exact portable-backend parity assertions

## Skips and failures

The default command omits only tests marked `#[ignore]`. Run the ignored-test command on a configured GPU host (axiomexec). Backend initialization failures must remain visible.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
