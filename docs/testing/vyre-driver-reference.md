# Testing `vyre-driver-reference`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-driver-reference
```

Adapt the reference interpreter to the backend contract for deterministic conformance execution.

The crate lives at `vyre-driver-reference` and owns the `reference-driver` seam in the `concrete-backend` layer.

## Commands

```console
./cargo_full test -p vyre-driver-reference
```

## Feature sets

This crate declares no Cargo features.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `example` | `vyre_driver_reference_release_surface` | `vyre-driver-reference/examples/vyre_driver_reference_release_surface.rs` | None | `./cargo_full test -p vyre-driver-reference --example vyre_driver_reference_release_surface` |
| `lib` | `vyre_driver_reference` | `vyre-driver-reference/src/lib.rs` | None | `./cargo_full test -p vyre-driver-reference` |
| `test` | `all_tests` | `vyre-driver-reference/tests/all_tests.rs` | None | `./cargo_full test -p vyre-driver-reference --test all_tests` |

## Test classes

- Device and capability contracts
- Lowering and artifact semantics
- Dispatch, graph, memory, and backend parity tests

## Hardware requirements

No accelerator is required. This backend must remain executable on the host reference path.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
