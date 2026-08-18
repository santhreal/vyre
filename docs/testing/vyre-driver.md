# Testing `vyre-driver`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-driver
```

Define backend-neutral device, target compiler registration, artifact materialization, binding, submission, completion, capability, dispatch, and evidence contracts.

The crate lives at `vyre-driver`. The `backend-contract` owner maintains its
`backend-neutral` testing contract.

## Commands

```console
./cargo_full test -p vyre-driver
```

```console
./cargo_full test -p vyre-driver --all-features
```

## Feature sets

- Default feature members: None
- Available manifest features: `default`, `libs-compositions`, `test-fixtures`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `example` | `vyre_driver_release_surface` | `vyre-driver/examples/vyre_driver_release_surface.rs` | None | `./cargo_full test -p vyre-driver --example vyre_driver_release_surface` |
| `lib` | `vyre_driver` | `vyre-driver/src/lib.rs` | None | `./cargo_full test -p vyre-driver` |
| `test` | `accounting_byte_range_accounting_contracts` | `vyre-driver/tests/accounting_byte_range_accounting_contracts.rs` | None | `./cargo_full test -p vyre-driver --test accounting_byte_range_accounting_contracts` |
| `test` | `accounting_checked_atomic_update_with_order_contracts` | `vyre-driver/tests/accounting_checked_atomic_update_with_order_contracts.rs` | None | `./cargo_full test -p vyre-driver --test accounting_checked_atomic_update_with_order_contracts` |
| `test` | `accounting_pinning_atomic_add_usize_with_order_contracts` | `vyre-driver/tests/accounting_pinning_atomic_add_usize_with_order_contracts.rs` | None | `./cargo_full test -p vyre-driver --test accounting_pinning_atomic_add_usize_with_order_contracts` |
| `test` | `actionable_errors` | `vyre-driver/tests/actionable_errors.rs` | None | `./cargo_full test -p vyre-driver --test actionable_errors` |
| `test` | `allocation_contracts` | `vyre-driver/tests/allocation_contracts.rs` | None | `./cargo_full test -p vyre-driver --test allocation_contracts` |
| `test` | `arm_independence_contracts` | `vyre-driver/tests/arm_independence_contracts.rs` | None | `./cargo_full test -p vyre-driver --test arm_independence_contracts` |
| `test` | `artifact_invocation_grid` | `vyre-driver/tests/artifact_invocation_grid.rs` | None | `./cargo_full test -p vyre-driver --test artifact_invocation_grid` |
| `test` | `async_copy_overlap_contracts` | `vyre-driver/tests/async_copy_overlap_contracts.rs` | None | `./cargo_full test -p vyre-driver --test async_copy_overlap_contracts` |
| `test` | `async_dispatch_contract` | `vyre-driver/tests/async_dispatch_contract.rs` | None | `./cargo_full test -p vyre-driver --test async_dispatch_contract` |
| `test` | `atomic_file_operation_race_policy` | `vyre-driver/tests/atomic_file_operation_race_policy.rs` | None | `./cargo_full test -p vyre-driver --test atomic_file_operation_race_policy` |
| `test` | `autotune_store_contracts` | `vyre-driver/tests/autotune_store_contracts.rs` | None | `./cargo_full test -p vyre-driver --test autotune_store_contracts` |
| `test` | `backend_capability_digests` | `vyre-driver/tests/backend_capability_digests.rs` | None | `./cargo_full test -p vyre-driver --test backend_capability_digests` |
| `test` | `backend_handle_lifetime_provenance` | `vyre-driver/tests/backend_handle_lifetime_provenance.rs` | None | `./cargo_full test -p vyre-driver --test backend_handle_lifetime_provenance` |
| `test` | `backend_launch_validation` | `vyre-driver/tests/backend_launch_validation.rs` | None | `./cargo_full test -p vyre-driver --test backend_launch_validation` |
| `test` | `backend_registry` | `vyre-driver/tests/backend_registry.rs` | None | `./cargo_full test -p vyre-driver --test backend_registry` |
| `test` | `backend_registry_duplicate_provider` | `vyre-driver/tests/backend_registry_duplicate_provider.rs` | None | `./cargo_full test -p vyre-driver --test backend_registry_duplicate_provider` |
| `test` | `backend_trait_contract` | `vyre-driver/tests/backend_trait_contract.rs` | None | `./cargo_full test -p vyre-driver --test backend_trait_contract` |
| `test` | `backend_validation_defaults` | `vyre-driver/tests/backend_validation_defaults.rs` | None | `./cargo_full test -p vyre-driver --test backend_validation_defaults` |
| `test` | `backpressure_queue_quota_policy` | `vyre-driver/tests/backpressure_queue_quota_policy.rs` | None | `./cargo_full test -p vyre-driver --test backpressure_queue_quota_policy` |
| `test` | `benchmark_pass_selection_contracts` | `vyre-driver/tests/benchmark_pass_selection_contracts.rs` | None | `./cargo_full test -p vyre-driver --test benchmark_pass_selection_contracts` |
| `test` | `bindless_policy_contracts` | `vyre-driver/tests/bindless_policy_contracts.rs` | None | `./cargo_full test -p vyre-driver --test bindless_policy_contracts` |
| `test` | `cache_eviction_contracts` | `vyre-driver/tests/cache_eviction_contracts.rs` | None | `./cargo_full test -p vyre-driver --test cache_eviction_contracts` |
| `test` | `cache_eviction_heat_contracts` | `vyre-driver/tests/cache_eviction_heat_contracts.rs` | None | `./cargo_full test -p vyre-driver --test cache_eviction_heat_contracts` |
| `test` | `cache_invalidation_default` | `vyre-driver/tests/cache_invalidation_default.rs` | None | `./cargo_full test -p vyre-driver --test cache_invalidation_default` |
| `test` | `cache_invalidation_default` | `vyre-driver/tests/cache_invalidation_default.rs` | `libs-compositions` | `./cargo_full test -p vyre-driver --test cache_invalidation_default` |
| `test` | `capability_adversarial` | `vyre-driver/tests/capability_adversarial.rs` | None | `./cargo_full test -p vyre-driver --test capability_adversarial` |
| `test` | `command_execution_boundary` | `vyre-driver/tests/command_execution_boundary.rs` | None | `./cargo_full test -p vyre-driver --test command_execution_boundary` |
| `test` | `command_reuse_policy_contracts` | `vyre-driver/tests/command_reuse_policy_contracts.rs` | None | `./cargo_full test -p vyre-driver --test command_reuse_policy_contracts` |
| `test` | `concurrency_schedule_contracts` | `vyre-driver/tests/concurrency_schedule_contracts.rs` | None | `./cargo_full test -p vyre-driver --test concurrency_schedule_contracts` |
| `test` | `consumer_boundary` | `vyre-driver/tests/consumer_boundary.rs` | None | `./cargo_full test -p vyre-driver --test consumer_boundary` |
| `test` | `crypto_rng_key_lifecycle` | `vyre-driver/tests/crypto_rng_key_lifecycle.rs` | None | `./cargo_full test -p vyre-driver --test crypto_rng_key_lifecycle` |
| `test` | `device_convergence_contracts` | `vyre-driver/tests/device_convergence_contracts.rs` | None | `./cargo_full test -p vyre-driver --test device_convergence_contracts` |
| `test` | `device_diagnostic_aggregation_contracts` | `vyre-driver/tests/device_diagnostic_aggregation_contracts.rs` | None | `./cargo_full test -p vyre-driver --test device_diagnostic_aggregation_contracts` |
| `test` | `device_signature_path` | `vyre-driver/tests/device_signature_path.rs` | None | `./cargo_full test -p vyre-driver --test device_signature_path` |
| `test` | `device_work_queue_contracts` | `vyre-driver/tests/device_work_queue_contracts.rs` | None | `./cargo_full test -p vyre-driver --test device_work_queue_contracts` |
| `test` | `diagnostic_surface` | `vyre-driver/tests/diagnostic_surface.rs` | None | `./cargo_full test -p vyre-driver --test diagnostic_surface` |
| `test` | `dispatch_config_surface` | `vyre-driver/tests/dispatch_config_surface.rs` | None | `./cargo_full test -p vyre-driver --test dispatch_config_surface` |
| `test` | `driver_contracts` | `vyre-driver/tests/driver_contracts.rs` | None | `./cargo_full test -p vyre-driver --test driver_contracts` |
| `test` | `driver_lifecycle_e2e` | `vyre-driver/tests/driver_lifecycle_e2e.rs` | None | `./cargo_full test -p vyre-driver --test driver_lifecycle_e2e` |
| `test` | `error_code_catalog` | `vyre-driver/tests/error_code_catalog.rs` | None | `./cargo_full test -p vyre-driver --test error_code_catalog` |
| `test` | `error_code_frozen` | `vyre-driver/tests/error_code_frozen.rs` | None | `./cargo_full test -p vyre-driver --test error_code_frozen` |
| `test` | `extraction_memory_verifier_cost_model` | `vyre-driver/tests/extraction_memory_verifier_cost_model.rs` | None | `./cargo_full test -p vyre-driver --test extraction_memory_verifier_cost_model` |
| `test` | `fusion_contracts` | `vyre-driver/tests/fusion_contracts.rs` | None | `./cargo_full test -p vyre-driver --test fusion_contracts` |
| `test` | `gap_duplicate_op_id` | `vyre-driver/tests/gap_duplicate_op_id.rs` | None | `./cargo_full test -p vyre-driver --test gap_duplicate_op_id` |
| `test` | `geometry_lowering_plan_search` | `vyre-driver/tests/geometry_lowering_plan_search.rs` | None | `./cargo_full test -p vyre-driver --test geometry_lowering_plan_search` |
| `test` | `grid_sync_detection_reaches_every_body_variant` | `vyre-driver/tests/grid_sync_detection_reaches_every_body_variant.rs` | None | `./cargo_full test -p vyre-driver --test grid_sync_detection_reaches_every_body_variant` |
| `test` | `grid_sync_nested_fence_survives_split` | `vyre-driver/tests/grid_sync_nested_fence_survives_split.rs` | None | `./cargo_full test -p vyre-driver --test grid_sync_nested_fence_survives_split` |
| `test` | `grid_sync_segments_declare_every_referenced_buffer` | `vyre-driver/tests/grid_sync_segments_declare_every_referenced_buffer.rs` | None | `./cargo_full test -p vyre-driver --test grid_sync_segments_declare_every_referenced_buffer` |
| `test` | `hostile_input_probe_shapes` | `vyre-driver/tests/hostile_input_probe_shapes.rs` | None | `./cargo_full test -p vyre-driver --test hostile_input_probe_shapes` |
| `test` | `http_proxy_redirect_policy` | `vyre-driver/tests/http_proxy_redirect_policy.rs` | None | `./cargo_full test -p vyre-driver --test http_proxy_redirect_policy` |
| `test` | `input_identity_contracts` | `vyre-driver/tests/input_identity_contracts.rs` | None | `./cargo_full test -p vyre-driver --test input_identity_contracts` |
| `test` | `intrinsic_registration_contract` | `vyre-driver/tests/intrinsic_registration_contract.rs` | None | `./cargo_full test -p vyre-driver --test intrinsic_registration_contract` |
| `test` | `launch_fusion_contracts` | `vyre-driver/tests/launch_fusion_contracts.rs` | None | `./cargo_full test -p vyre-driver --test launch_fusion_contracts` |
| `test` | `megakernel_execution_contracts` | `vyre-driver/tests/megakernel_execution_contracts.rs` | None | `./cargo_full test -p vyre-driver --test megakernel_execution_contracts` |
| `test` | `mixed_work_autotuning` | `vyre-driver/tests/mixed_work_autotuning.rs` | None | `./cargo_full test -p vyre-driver --test mixed_work_autotuning` |
| `test` | `no_backend_crate_links_host_arithmetic` | `vyre-driver/tests/no_backend_crate_links_host_arithmetic.rs` | None | `./cargo_full test -p vyre-driver --test no_backend_crate_links_host_arithmetic` |
| `test` | `numeric_contracts` | `vyre-driver/tests/numeric_contracts.rs` | None | `./cargo_full test -p vyre-driver --test numeric_contracts` |
| `test` | `ordering_contracts` | `vyre-driver/tests/ordering_contracts.rs` | None | `./cargo_full test -p vyre-driver --test ordering_contracts` |
| `test` | `output_slab_provenance` | `vyre-driver/tests/output_slab_provenance.rs` | None | `./cargo_full test -p vyre-driver --test output_slab_provenance` |
| `test` | `output_slots_contracts` | `vyre-driver/tests/output_slots_contracts.rs` | None | `./cargo_full test -p vyre-driver --test output_slots_contracts` |
| `test` | `param_inlining_contracts` | `vyre-driver/tests/param_inlining_contracts.rs` | None | `./cargo_full test -p vyre-driver --test param_inlining_contracts` |
| `test` | `persistent_contracts` | `vyre-driver/tests/persistent_contracts.rs` | None | `./cargo_full test -p vyre-driver --test persistent_contracts` |
| `test` | `pipeline_fusion_contracts` | `vyre-driver/tests/pipeline_fusion_contracts.rs` | None | `./cargo_full test -p vyre-driver --test pipeline_fusion_contracts` |
| `test` | `reference_oracle_is_never_implicit` | `vyre-driver/tests/reference_oracle_is_never_implicit.rs` | None | `./cargo_full test -p vyre-driver --test reference_oracle_is_never_implicit` |
| `test` | `reference_oracle_loses_to_a_device` | `vyre-driver/tests/reference_oracle_loses_to_a_device.rs` | None | `./cargo_full test -p vyre-driver --test reference_oracle_loses_to_a_device` |
| `test` | `registry_closure` | `vyre-driver/tests/registry_closure.rs` | None | `./cargo_full test -p vyre-driver --test registry_closure` |
| `test` | `rejection_wording_contract` | `vyre-driver/tests/rejection_wording_contract.rs` | None | `./cargo_full test -p vyre-driver --test rejection_wording_contract` |
| `test` | `release_publication_boundary` | `vyre-driver/tests/release_publication_boundary.rs` | None | `./cargo_full test -p vyre-driver --test release_publication_boundary` |
| `test` | `reservation_policy_contracts` | `vyre-driver/tests/reservation_policy_contracts.rs` | None | `./cargo_full test -p vyre-driver --test reservation_policy_contracts` |
| `test` | `resident_binding_projection` | `vyre-driver/tests/resident_binding_projection.rs` | None | `./cargo_full test -p vyre-driver --test resident_binding_projection` |
| `test` | `result_compaction_contracts` | `vyre-driver/tests/result_compaction_contracts.rs` | None | `./cargo_full test -p vyre-driver --test result_compaction_contracts` |
| `test` | `routing_registry_surface` | `vyre-driver/tests/routing_registry_surface.rs` | None | `./cargo_full test -p vyre-driver --test routing_registry_surface` |
| `test` | `runtime_watchdog_proofs` | `vyre-driver/tests/runtime_watchdog_proofs.rs` | None | `./cargo_full test -p vyre-driver --test runtime_watchdog_proofs` |
| `test` | `scan_graph_update_classifier_registry` | `vyre-driver/tests/scan_graph_update_classifier_registry.rs` | None | `./cargo_full test -p vyre-driver --test scan_graph_update_classifier_registry` |
| `test` | `shadow_contracts` | `vyre-driver/tests/shadow_contracts.rs` | None | `./cargo_full test -p vyre-driver --test shadow_contracts` |
| `test` | `shape_prediction_contracts` | `vyre-driver/tests/shape_prediction_contracts.rs` | None | `./cargo_full test -p vyre-driver --test shape_prediction_contracts` |
| `test` | `speculation_verdict_contracts` | `vyre-driver/tests/speculation_verdict_contracts.rs` | None | `./cargo_full test -p vyre-driver --test speculation_verdict_contracts` |
| `test` | `strategy_contracts` | `vyre-driver/tests/strategy_contracts.rs` | None | `./cargo_full test -p vyre-driver --test strategy_contracts` |
| `test` | `sweep_dispatch_shape_oracle_matrix` | `vyre-driver/tests/sweep_dispatch_shape_oracle_matrix.rs` | None | `./cargo_full test -p vyre-driver --test sweep_dispatch_shape_oracle_matrix` |
| `test` | `sweep_numeric_oracle_matrix` | `vyre-driver/tests/sweep_numeric_oracle_matrix.rs` | None | `./cargo_full test -p vyre-driver --test sweep_numeric_oracle_matrix` |
| `test` | `target_contract` | `vyre-driver/tests/target_contract.rs` | None | `./cargo_full test -p vyre-driver --test target_contract` |
| `test` | `trace_context_telemetry_contracts` | `vyre-driver/tests/trace_context_telemetry_contracts.rs` | None | `./cargo_full test -p vyre-driver --test trace_context_telemetry_contracts` |
| `test` | `trace_jit_policy_contracts` | `vyre-driver/tests/trace_jit_policy_contracts.rs` | None | `./cargo_full test -p vyre-driver --test trace_jit_policy_contracts` |
| `test` | `transfer_accounting_contracts` | `vyre-driver/tests/transfer_accounting_contracts.rs` | None | `./cargo_full test -p vyre-driver --test transfer_accounting_contracts` |
| `test` | `vyre_backend_forwarding_closure` | `vyre-driver/tests/vyre_backend_forwarding_closure.rs` | None | `./cargo_full test -p vyre-driver --test vyre_backend_forwarding_closure` |

## Test classes

- Backend trait and capability contracts
- Dispatch, artifact, evidence, and error semantics
- Backend-neutral lifecycle and concurrency tests

## Hardware requirements

No accelerator is required for the default suite.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
