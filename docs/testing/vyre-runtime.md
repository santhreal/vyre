# Testing `vyre-runtime`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-runtime
```

Execute the artifact's selected persistence: sessions, recovery, residency, scheduling, caches, telemetry, readback, and IO. Does not decide whether to be persistent.

The crate lives at `vyre-runtime`. The `runtime` owner maintains its
`runtime` testing contract.

## Commands

```console
./cargo_full test -p vyre-runtime
```

```console
./cargo_full test -p vyre-runtime --all-features
```

## Feature sets

- Default feature members: None
- Available manifest features: `default`, `libs-compositions`, `megakernel-batch`, `remote-cache`, `subgroup-ops`, `uring-cmd-nvme`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `example` | `vyre_runtime_release_surface` | `vyre-runtime/examples/vyre_runtime_release_surface.rs` | None | `./cargo_full test -p vyre-runtime --example vyre_runtime_release_surface` |
| `lib` | `vyre_runtime` | `vyre-runtime/src/lib.rs` | None | `./cargo_full test -p vyre-runtime` |
| `test` | `adversarial_disk` | `vyre-runtime/tests/adversarial_disk.rs` | None | `./cargo_full test -p vyre-runtime --test adversarial_disk` |
| `test` | `artifact_admission_contract` | `vyre-runtime/tests/artifact_admission_contract.rs` | None | `./cargo_full test -p vyre-runtime --test artifact_admission_contract` |
| `test` | `artifact_workspace_contract` | `vyre-runtime/tests/artifact_workspace_contract.rs` | None | `./cargo_full test -p vyre-runtime --test artifact_workspace_contract` |
| `test` | `cache_eviction_proptest` | `vyre-runtime/tests/cache_eviction_proptest.rs` | None | `./cargo_full test -p vyre-runtime --test cache_eviction_proptest` |
| `test` | `concurrency_invariants` | `vyre-runtime/tests/concurrency_invariants.rs` | None | `./cargo_full test -p vyre-runtime --test concurrency_invariants` |
| `test` | `driver_runtime_lifecycle_boundary` | `vyre-runtime/tests/driver_runtime_lifecycle_boundary.rs` | None | `./cargo_full test -p vyre-runtime --test driver_runtime_lifecycle_boundary` |
| `test` | `multi_tenant_scheduler` | `vyre-runtime/tests/multi_tenant_scheduler.rs` | None | `./cargo_full test -p vyre-runtime --test multi_tenant_scheduler` |
| `test` | `paged_prefix_mtp_contracts` | `vyre-runtime/tests/paged_prefix_mtp_contracts.rs` | None | `./cargo_full test -p vyre-runtime --test paged_prefix_mtp_contracts` |
| `test` | `pipeline_fingerprint_surface` | `vyre-runtime/tests/pipeline_fingerprint_surface.rs` | None | `./cargo_full test -p vyre-runtime --test pipeline_fingerprint_surface` |
| `test` | `registry_closure` | `vyre-runtime/tests/registry_closure.rs` | None | `./cargo_full test -p vyre-runtime --test registry_closure` |
| `test` | `replay_log` | `vyre-runtime/tests/replay_log.rs` | None | `./cargo_full test -p vyre-runtime --test replay_log` |
| `test` | `resident_queue_contracts` | `vyre-runtime/tests/resident_queue_contracts.rs` | None | `./cargo_full test -p vyre-runtime --test resident_queue_contracts` |
| `test` | `resident_work_queue_advanced_hierarchical_atomics_contracts` | `vyre-runtime/tests/resident_work_queue_advanced_hierarchical_atomics_contracts.rs` | None | `./cargo_full test -p vyre-runtime --test resident_work_queue_advanced_hierarchical_atomics_contracts` |
| `test` | `resident_work_queue_advanced_parallel_dfa_contracts` | `vyre-runtime/tests/resident_work_queue_advanced_parallel_dfa_contracts.rs` | None | `./cargo_full test -p vyre-runtime --test resident_work_queue_advanced_parallel_dfa_contracts` |
| `test` | `resident_work_queue_advanced_zero_copy_io_contracts` | `vyre-runtime/tests/resident_work_queue_advanced_zero_copy_io_contracts.rs` | None | `./cargo_full test -p vyre-runtime --test resident_work_queue_advanced_zero_copy_io_contracts` |
| `test` | `resident_work_queue_adversarial_buffers` | `vyre-runtime/tests/resident_work_queue_adversarial_buffers.rs` | None | `./cargo_full test -p vyre-runtime --test resident_work_queue_adversarial_buffers` |
| `test` | `resident_work_queue_adversarial_metrics` | `vyre-runtime/tests/resident_work_queue_adversarial_metrics.rs` | None | `./cargo_full test -p vyre-runtime --test resident_work_queue_adversarial_metrics` |
| `test` | `resident_work_queue_adversarial_overflow` | `vyre-runtime/tests/resident_work_queue_adversarial_overflow.rs` | None | `./cargo_full test -p vyre-runtime --test resident_work_queue_adversarial_overflow` |
| `test` | `resident_work_queue_adversarial_packing` | `vyre-runtime/tests/resident_work_queue_adversarial_packing.rs` | None | `./cargo_full test -p vyre-runtime --test resident_work_queue_adversarial_packing` |
| `test` | `resident_work_queue_allocation_bounds` | `vyre-runtime/tests/resident_work_queue_allocation_bounds.rs` | None | `./cargo_full test -p vyre-runtime --test resident_work_queue_allocation_bounds` |
| `test` | `resident_work_queue_async_observability` | `vyre-runtime/tests/resident_work_queue_async_observability.rs` | None | `./cargo_full test -p vyre-runtime --test resident_work_queue_async_observability` |
| `test` | `resident_work_queue_automata_worklist_contracts` | `vyre-runtime/tests/resident_work_queue_automata_worklist_contracts.rs` | None | `./cargo_full test -p vyre-runtime --test resident_work_queue_automata_worklist_contracts` |
| `test` | `resident_work_queue_barrier_elision_variant_closure` | `vyre-runtime/tests/resident_work_queue_barrier_elision_variant_closure.rs` | None | `./cargo_full test -p vyre-runtime --test resident_work_queue_barrier_elision_variant_closure` |
| `test` | `resident_work_queue_builder_delegation_parity` | `vyre-runtime/tests/resident_work_queue_builder_delegation_parity.rs` | None | `./cargo_full test -p vyre-runtime --test resident_work_queue_builder_delegation_parity` |
| `test` | `resident_work_queue_core_contracts` | `vyre-runtime/tests/resident_work_queue_core_contracts.rs` | None | `./cargo_full test -p vyre-runtime --test resident_work_queue_core_contracts` |
| `test` | `resident_work_queue_cpu_fallback_wording` | `vyre-runtime/tests/resident_work_queue_cpu_fallback_wording.rs` | None | `./cargo_full test -p vyre-runtime --test resident_work_queue_cpu_fallback_wording` |
| `test` | `resident_work_queue_duplicate_packing` | `vyre-runtime/tests/resident_work_queue_duplicate_packing.rs` | None | `./cargo_full test -p vyre-runtime --test resident_work_queue_duplicate_packing` |
| `test` | `resident_work_queue_host_protocol_contracts` | `vyre-runtime/tests/resident_work_queue_host_protocol_contracts.rs` | None | `./cargo_full test -p vyre-runtime --test resident_work_queue_host_protocol_contracts` |
| `test` | `resident_work_queue_io_public_errors` | `vyre-runtime/tests/resident_work_queue_io_public_errors.rs` | None | `./cargo_full test -p vyre-runtime --test resident_work_queue_io_public_errors` |
| `test` | `resident_work_queue_mixed_work_contracts` | `vyre-runtime/tests/resident_work_queue_mixed_work_contracts.rs` | None | `./cargo_full test -p vyre-runtime --test resident_work_queue_mixed_work_contracts` |
| `test` | `resident_work_queue_overflow_boundaries` | `vyre-runtime/tests/resident_work_queue_overflow_boundaries.rs` | None | `./cargo_full test -p vyre-runtime --test resident_work_queue_overflow_boundaries` |
| `test` | `resident_work_queue_planner_launch_contracts` | `vyre-runtime/tests/resident_work_queue_planner_launch_contracts.rs` | None | `./cargo_full test -p vyre-runtime --test resident_work_queue_planner_launch_contracts` |
| `test` | `resident_work_queue_protocol_boundary` | `vyre-runtime/tests/resident_work_queue_protocol_boundary.rs` | None | `./cargo_full test -p vyre-runtime --test resident_work_queue_protocol_boundary` |
| `test` | `resident_work_queue_protocol_codec_contracts` | `vyre-runtime/tests/resident_work_queue_protocol_codec_contracts.rs` | None | `./cargo_full test -p vyre-runtime --test resident_work_queue_protocol_codec_contracts` |
| `test` | `resident_work_queue_protocol_edge_cases` | `vyre-runtime/tests/resident_work_queue_protocol_edge_cases.rs` | None | `./cargo_full test -p vyre-runtime --test resident_work_queue_protocol_edge_cases` |
| `test` | `resident_work_queue_protocol_layout_contracts` | `vyre-runtime/tests/resident_work_queue_protocol_layout_contracts.rs` | None | `./cargo_full test -p vyre-runtime --test resident_work_queue_protocol_layout_contracts` |
| `test` | `resident_work_queue_protocol_strict_contracts` | `vyre-runtime/tests/resident_work_queue_protocol_strict_contracts.rs` | None | `./cargo_full test -p vyre-runtime --test resident_work_queue_protocol_strict_contracts` |
| `test` | `resident_work_queue_readback_contracts` | `vyre-runtime/tests/resident_work_queue_readback_contracts.rs` | None | `./cargo_full test -p vyre-runtime --test resident_work_queue_readback_contracts` |
| `test` | `resident_work_queue_rule_catalog_contracts` | `vyre-runtime/tests/resident_work_queue_rule_catalog_contracts.rs` | None | `./cargo_full test -p vyre-runtime --test resident_work_queue_rule_catalog_contracts` |
| `test` | `resident_work_queue_rule_catalog_scratch` | `vyre-runtime/tests/resident_work_queue_rule_catalog_scratch.rs` | None | `./cargo_full test -p vyre-runtime --test resident_work_queue_rule_catalog_scratch` |
| `test` | `resident_work_queue_scheduler_fairness` | `vyre-runtime/tests/resident_work_queue_scheduler_fairness.rs` | None | `./cargo_full test -p vyre-runtime --test resident_work_queue_scheduler_fairness` |
| `test` | `resident_work_queue_sketch_telemetry` | `vyre-runtime/tests/resident_work_queue_sketch_telemetry.rs` | None | `./cargo_full test -p vyre-runtime --test resident_work_queue_sketch_telemetry` |
| `test` | `resident_work_queue_workspace_layout_contracts` | `vyre-runtime/tests/resident_work_queue_workspace_layout_contracts.rs` | None | `./cargo_full test -p vyre-runtime --test resident_work_queue_workspace_layout_contracts` |
| `test` | `resource_residency` | `vyre-runtime/tests/resource_residency.rs` | None | `./cargo_full test -p vyre-runtime --test resource_residency` |
| `test` | `routing_policy` | `vyre-runtime/tests/routing_policy.rs` | None | `./cargo_full test -p vyre-runtime --test routing_policy` |
| `test` | `routing_standard_policy_contracts` | `vyre-runtime/tests/routing_standard_policy_contracts.rs` | None | `./cargo_full test -p vyre-runtime --test routing_standard_policy_contracts` |
| `test` | `safetensors_transfer_integrity_contracts` | `vyre-runtime/tests/safetensors_transfer_integrity_contracts.rs` | None | `./cargo_full test -p vyre-runtime --test safetensors_transfer_integrity_contracts` |
| `test` | `scheduler_model_proptest` | `vyre-runtime/tests/scheduler_model_proptest.rs` | None | `./cargo_full test -p vyre-runtime --test scheduler_model_proptest` |
| `test` | `socket_ingest` | `vyre-runtime/tests/socket_ingest.rs` | None | `./cargo_full test -p vyre-runtime --test socket_ingest` |
| `test` | `sweep_ring_buffer_oracle_matrix` | `vyre-runtime/tests/sweep_ring_buffer_oracle_matrix.rs` | None | `./cargo_full test -p vyre-runtime --test sweep_ring_buffer_oracle_matrix` |
| `test` | `sweep_tenant_policy_oracle_matrix` | `vyre-runtime/tests/sweep_tenant_policy_oracle_matrix.rs` | None | `./cargo_full test -p vyre-runtime --test sweep_tenant_policy_oracle_matrix` |
| `test` | `uring_completion_pump_contracts` | `vyre-runtime/tests/uring_completion_pump_contracts.rs` | None | `./cargo_full test -p vyre-runtime --test uring_completion_pump_contracts` |
| `test` | `uring_ingest_telemetry_invariants` | `vyre-runtime/tests/uring_ingest_telemetry_invariants.rs` | None | `./cargo_full test -p vyre-runtime --test uring_ingest_telemetry_invariants` |
| `test` | `uring_smoke` | `vyre-runtime/tests/uring_smoke.rs` | None | `./cargo_full test -p vyre-runtime --test uring_smoke` |

## Test classes

- Execution planning and cache contracts
- Persistent runtime state transitions
- IO, telemetry, and failure semantics

## Hardware requirements

Backend-neutral runtime tests are host-capable. Device integration tests require the selected concrete backend on the execution host (axiomexec) and treat unavailable requested hardware as an error.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
