# Testing `vyre-driver`

Run the default crate suite from the workspace root:

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver
```

Define backend-neutral device, target compiler registration, artifact materialization, binding, submission, completion, capability, dispatch, and evidence contracts.

The crate lives at `vyre-driver`. The `backend-contract` owner maintains its
`backend-neutral` testing contract.

## Commands

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver
```

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --all-features
```

## Feature sets

- Default feature members: None
- Available manifest features: `default`, `self-substrate-adapters`, `test-fixtures`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `example` | `vyre_driver_release_surface` | `vyre-driver/examples/vyre_driver_release_surface.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --example vyre_driver_release_surface` |
| `lib` | `vyre_driver` | `vyre-driver/src/lib.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver` |
| `test` | `actionable_errors` | `vyre-driver/tests/actionable_errors.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test actionable_errors` |
| `test` | `artifact_invocation_grid` | `vyre-driver/tests/artifact_invocation_grid.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test artifact_invocation_grid` |
| `test` | `async_dispatch_always_nonblocking` | `vyre-driver/tests/async_dispatch_always_nonblocking.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test async_dispatch_always_nonblocking` |
| `test` | `async_dispatch_contract` | `vyre-driver/tests/async_dispatch_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test async_dispatch_contract` |
| `test` | `atomic_file_operation_race_policy` | `vyre-driver/tests/atomic_file_operation_race_policy.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test atomic_file_operation_race_policy` |
| `test` | `backend_capability_digests` | `vyre-driver/tests/backend_capability_digests.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test backend_capability_digests` |
| `test` | `backend_capability_negotiation` | `vyre-driver/tests/backend_capability_negotiation.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test backend_capability_negotiation` |
| `test` | `backend_contract` | `vyre-driver/tests/backend_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test backend_contract` |
| `test` | `backend_handle_lifetime_provenance` | `vyre-driver/tests/backend_handle_lifetime_provenance.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test backend_handle_lifetime_provenance` |
| `test` | `backend_launch_validation` | `vyre-driver/tests/backend_launch_validation.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test backend_launch_validation` |
| `test` | `backend_registry_duplicate_provider` | `vyre-driver/tests/backend_registry_duplicate_provider.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test backend_registry_duplicate_provider` |
| `test` | `backend_trait_compatibility` | `vyre-driver/tests/backend_trait_compatibility.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test backend_trait_compatibility` |
| `test` | `backend_validation_defaults` | `vyre-driver/tests/backend_validation_defaults.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test backend_validation_defaults` |
| `test` | `backpressure_queue_quota_policy` | `vyre-driver/tests/backpressure_queue_quota_policy.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test backpressure_queue_quota_policy` |
| `test` | `cache_invalidation_default` | `vyre-driver/tests/cache_invalidation_default.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test cache_invalidation_default` |
| `test` | `cache_invalidation_default` | `vyre-driver/tests/cache_invalidation_default.rs` | `self-substrate-adapters` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test cache_invalidation_default` |
| `test` | `capability_adversarial` | `vyre-driver/tests/capability_adversarial.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test capability_adversarial` |
| `test` | `command_execution_boundary` | `vyre-driver/tests/command_execution_boundary.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test command_execution_boundary` |
| `test` | `concurrency_schedule_contracts` | `vyre-driver/tests/concurrency_schedule_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test concurrency_schedule_contracts` |
| `test` | `consumer_boundary` | `vyre-driver/tests/consumer_boundary.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test consumer_boundary` |
| `test` | `crypto_rng_key_lifecycle` | `vyre-driver/tests/crypto_rng_key_lifecycle.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test crypto_rng_key_lifecycle` |
| `test` | `d_series_integration` | `vyre-driver/tests/d_series_integration.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test d_series_integration` |
| `test` | `device_signature_path` | `vyre-driver/tests/device_signature_path.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test device_signature_path` |
| `test` | `diagnostic_surface` | `vyre-driver/tests/diagnostic_surface.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test diagnostic_surface` |
| `test` | `dispatch_config_surface` | `vyre-driver/tests/dispatch_config_surface.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test dispatch_config_surface` |
| `test` | `driver_contracts` | `vyre-driver/tests/driver_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test driver_contracts` |
| `test` | `driver_lifecycle_e2e` | `vyre-driver/tests/driver_lifecycle_e2e.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test driver_lifecycle_e2e` |
| `test` | `error_code_catalog` | `vyre-driver/tests/error_code_catalog.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test error_code_catalog` |
| `test` | `error_code_frozen` | `vyre-driver/tests/error_code_frozen.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test error_code_frozen` |
| `test` | `extraction_memory_verifier_cost_model` | `vyre-driver/tests/extraction_memory_verifier_cost_model.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test extraction_memory_verifier_cost_model` |
| `test` | `gap_duplicate_op_id` | `vyre-driver/tests/gap_duplicate_op_id.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test gap_duplicate_op_id` |
| `test` | `gap_error_code_catalog` | `vyre-driver/tests/gap_error_code_catalog.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test gap_error_code_catalog` |
| `test` | `grid_sync_detection_reaches_every_body_variant` | `vyre-driver/tests/grid_sync_detection_reaches_every_body_variant.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test grid_sync_detection_reaches_every_body_variant` |
| `test` | `grid_sync_nested_fence_survives_split` | `vyre-driver/tests/grid_sync_nested_fence_survives_split.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test grid_sync_nested_fence_survives_split` |
| `test` | `hostile_input_probe_shapes` | `vyre-driver/tests/hostile_input_probe_shapes.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test hostile_input_probe_shapes` |
| `test` | `http_proxy_redirect_policy` | `vyre-driver/tests/http_proxy_redirect_policy.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test http_proxy_redirect_policy` |
| `test` | `intrinsic_registration_contract` | `vyre-driver/tests/intrinsic_registration_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test intrinsic_registration_contract` |
| `test` | `mixed_work_autotuning` | `vyre-driver/tests/mixed_work_autotuning.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test mixed_work_autotuning` |
| `test` | `output_slab_provenance` | `vyre-driver/tests/output_slab_provenance.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test output_slab_provenance` |
| `test` | `registry_closure` | `vyre-driver/tests/registry_closure.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test registry_closure` |
| `test` | `release_publication_boundary` | `vyre-driver/tests/release_publication_boundary.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test release_publication_boundary` |
| `test` | `routing_registry_surface` | `vyre-driver/tests/routing_registry_surface.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test routing_registry_surface` |
| `test` | `runtime_watchdog_proofs` | `vyre-driver/tests/runtime_watchdog_proofs.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test runtime_watchdog_proofs` |
| `test` | `scan_graph_update_classifier_registry` | `vyre-driver/tests/scan_graph_update_classifier_registry.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test scan_graph_update_classifier_registry` |
| `test` | `sweep_dispatch_shape_oracle_matrix` | `vyre-driver/tests/sweep_dispatch_shape_oracle_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test sweep_dispatch_shape_oracle_matrix` |
| `test` | `sweep_numeric_oracle_matrix` | `vyre-driver/tests/sweep_numeric_oracle_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test sweep_numeric_oracle_matrix` |
| `test` | `trace_context_telemetry_contracts` | `vyre-driver/tests/trace_context_telemetry_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test trace_context_telemetry_contracts` |
| `test` | `vyre_backend_forwarding_closure` | `vyre-driver/tests/vyre_backend_forwarding_closure.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver --test vyre_backend_forwarding_closure` |

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
