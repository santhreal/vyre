# Testing `vyre-runtime`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-runtime
```

Execute the artifact's selected persistence: sessions, recovery, residency, scheduling, caches, telemetry, readback, and IO. Does not decide whether to be persistent.

The crate lives at `vyre-runtime` and owns the `runtime` seam in the `runtime` layer.

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
| `test` | `all_tests` | `vyre-runtime/tests/all_tests.rs` | None | `./cargo_full test -p vyre-runtime --test all_tests` |

## Test classes

- Execution planning and cache contracts
- Persistent runtime state transitions
- IO, telemetry, and failure semantics

## Hardware requirements

Backend-neutral runtime tests are host-capable. Device integration tests require the selected concrete backend on the execution host and treat unavailable requested hardware as an error.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
