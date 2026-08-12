# Testing `vyre-runtime`

Run the default crate suite from the workspace root:

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-runtime
```

Own compile-to-materialize orchestration, artifact sessions, recovery, persistence, residency, scheduling, caches, telemetry, readback, and IO.

The crate lives at `vyre-runtime`. The `runtime` owner maintains its
`runtime` testing contract.

## Commands

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-runtime
```

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-runtime --all-features
```

## Feature sets

- Default feature members: None
- Available manifest features: `default`, `megakernel-batch`, `remote-cache`, `self-substrate-adapters`, `subgroup-ops`, `uring-cmd-nvme`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `example` | `vyre_runtime_release_surface` | `vyre-runtime/examples/vyre_runtime_release_surface.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-runtime --example vyre_runtime_release_surface` |
| `lib` | `vyre_runtime` | `vyre-runtime/src/lib.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-runtime` |
| `test` | `adversarial_disk` | `vyre-runtime/tests/adversarial_disk.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-runtime --test adversarial_disk` |
| `test` | `artifact_admission_contract` | `vyre-runtime/tests/artifact_admission_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-runtime --test artifact_admission_contract` |
| `test` | `cache_eviction_proptest` | `vyre-runtime/tests/cache_eviction_proptest.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-runtime --test cache_eviction_proptest` |
| `test` | `concurrency_invariants` | `vyre-runtime/tests/concurrency_invariants.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-runtime --test concurrency_invariants` |
| `test` | `fingerprint_cross_host` | `vyre-runtime/tests/fingerprint_cross_host.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-runtime --test fingerprint_cross_host` |
| `test` | `megakernel_adversarial_buffers` | `vyre-runtime/tests/megakernel_adversarial_buffers.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-runtime --test megakernel_adversarial_buffers` |
| `test` | `megakernel_adversarial_metrics` | `vyre-runtime/tests/megakernel_adversarial_metrics.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-runtime --test megakernel_adversarial_metrics` |
| `test` | `megakernel_adversarial_overflow` | `vyre-runtime/tests/megakernel_adversarial_overflow.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-runtime --test megakernel_adversarial_overflow` |
| `test` | `megakernel_adversarial_packing` | `vyre-runtime/tests/megakernel_adversarial_packing.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-runtime --test megakernel_adversarial_packing` |
| `test` | `megakernel_allocation_bounds` | `vyre-runtime/tests/megakernel_allocation_bounds.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-runtime --test megakernel_allocation_bounds` |
| `test` | `megakernel_async_observability` | `vyre-runtime/tests/megakernel_async_observability.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-runtime --test megakernel_async_observability` |
| `test` | `megakernel_builder_delegation_parity` | `vyre-runtime/tests/megakernel_builder_delegation_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-runtime --test megakernel_builder_delegation_parity` |
| `test` | `megakernel_core_contracts` | `vyre-runtime/tests/megakernel_core_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-runtime --test megakernel_core_contracts` |
| `test` | `megakernel_cpu_fallback_wording` | `vyre-runtime/tests/megakernel_cpu_fallback_wording.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-runtime --test megakernel_cpu_fallback_wording` |
| `test` | `megakernel_duplicate_packing` | `vyre-runtime/tests/megakernel_duplicate_packing.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-runtime --test megakernel_duplicate_packing` |
| `test` | `megakernel_host_protocol_contracts` | `vyre-runtime/tests/megakernel_host_protocol_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-runtime --test megakernel_host_protocol_contracts` |
| `test` | `megakernel_io_public_errors` | `vyre-runtime/tests/megakernel_io_public_errors.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-runtime --test megakernel_io_public_errors` |
| `test` | `megakernel_overflow_boundaries` | `vyre-runtime/tests/megakernel_overflow_boundaries.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-runtime --test megakernel_overflow_boundaries` |
| `test` | `megakernel_protocol_boundary` | `vyre-runtime/tests/megakernel_protocol_boundary.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-runtime --test megakernel_protocol_boundary` |
| `test` | `megakernel_protocol_edge_cases` | `vyre-runtime/tests/megakernel_protocol_edge_cases.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-runtime --test megakernel_protocol_edge_cases` |
| `test` | `megakernel_protocol_layout_contracts` | `vyre-runtime/tests/megakernel_protocol_layout_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-runtime --test megakernel_protocol_layout_contracts` |
| `test` | `megakernel_protocol_strict_contracts` | `vyre-runtime/tests/megakernel_protocol_strict_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-runtime --test megakernel_protocol_strict_contracts` |
| `test` | `megakernel_rule_catalog_scratch` | `vyre-runtime/tests/megakernel_rule_catalog_scratch.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-runtime --test megakernel_rule_catalog_scratch` |
| `test` | `megakernel_scheduler_fairness` | `vyre-runtime/tests/megakernel_scheduler_fairness.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-runtime --test megakernel_scheduler_fairness` |
| `test` | `megakernel_sketch_telemetry` | `vyre-runtime/tests/megakernel_sketch_telemetry.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-runtime --test megakernel_sketch_telemetry` |
| `test` | `multi_tenant_scheduler` | `vyre-runtime/tests/multi_tenant_scheduler.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-runtime --test multi_tenant_scheduler` |
| `test` | `pipeline_fingerprint_surface` | `vyre-runtime/tests/pipeline_fingerprint_surface.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-runtime --test pipeline_fingerprint_surface` |
| `test` | `resident_queue_contracts` | `vyre-runtime/tests/resident_queue_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-runtime --test resident_queue_contracts` |
| `test` | `resource_residency` | `vyre-runtime/tests/resource_residency.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-runtime --test resource_residency` |
| `test` | `routing_policy` | `vyre-runtime/tests/routing_policy.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-runtime --test routing_policy` |
| `test` | `scheduler_model_proptest` | `vyre-runtime/tests/scheduler_model_proptest.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-runtime --test scheduler_model_proptest` |
| `test` | `socket_ingest` | `vyre-runtime/tests/socket_ingest.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-runtime --test socket_ingest` |
| `test` | `sweep_ring_buffer_oracle_matrix` | `vyre-runtime/tests/sweep_ring_buffer_oracle_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-runtime --test sweep_ring_buffer_oracle_matrix` |
| `test` | `sweep_tenant_policy_oracle_matrix` | `vyre-runtime/tests/sweep_tenant_policy_oracle_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-runtime --test sweep_tenant_policy_oracle_matrix` |
| `test` | `uring_ingest_telemetry_invariants` | `vyre-runtime/tests/uring_ingest_telemetry_invariants.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-runtime --test uring_ingest_telemetry_invariants` |
| `test` | `uring_smoke` | `vyre-runtime/tests/uring_smoke.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-runtime --test uring_smoke` |

## Test classes

- Execution planning and cache contracts
- Persistent runtime state transitions
- IO, telemetry, and failure semantics

## Hardware requirements

Backend-neutral runtime tests are host-capable. Device integration tests require the selected concrete backend and treat unavailable requested hardware as an error.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
