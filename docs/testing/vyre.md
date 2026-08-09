# Testing `vyre`

Run the default crate suite from the workspace root:

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre
```

Expose the public Vyre API and feature-gated backend selection surface.

The crate lives at `vyre`. The `public-facade` owner maintains its
`facade` testing contract.

## Commands

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre
```

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre --all-features
```

## Feature sets

- Default feature members: None
- Available manifest features: `cpu-parity`, `cuda`, `default`, `wgpu`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `example` | `vyre_core_release_surface` | `vyre/examples/vyre_core_release_surface.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre --example vyre_core_release_surface` |
| `lib` | `vyre` | `vyre/src/lib.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre` |
| `test` | `artifact_workflow` | `vyre/tests/artifact_workflow.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre --test artifact_workflow` |
| `test` | `cross_crate_import_path_migration_contract` | `vyre/tests/cross_crate_import_path_migration_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre --test cross_crate_import_path_migration_contract` |
| `test` | `cross_crate_perf_contracts` | `vyre/tests/cross_crate_perf_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre --test cross_crate_perf_contracts` |
| `test` | `gpu_boundary_contracts` | `vyre/tests/gpu_boundary_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre --test gpu_boundary_contracts` |
| `test` | `ops` | `vyre/tests/ops.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre --test ops` |
| `test` | `wire_malformed_adversarial` | `vyre/tests/wire_malformed_adversarial.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre --test wire_malformed_adversarial` |
| `test` | `wire_v1_round_trip` | `vyre/tests/wire_v1_round_trip.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre --test wire_v1_round_trip` |

## Test classes

- Public facade and feature-routing contracts
- Backend-neutral API and dispatch integration
- Documentation examples and compatibility tests

## Hardware requirements

The default suite is host-only. Tests that execute concrete GPU features require the corresponding device and driver, and a requested acquisition failure is an error.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
