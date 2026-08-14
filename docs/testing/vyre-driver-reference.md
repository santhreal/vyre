# Testing `vyre-driver-reference`

Run the default crate suite from the workspace root:

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-reference
```

Adapt the reference interpreter to the backend contract for deterministic conformance execution.

The crate lives at `vyre-driver-reference`. The `reference-driver` owner maintains its
`concrete-backend` testing contract.

## Commands

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-reference
```

## Feature sets

This crate declares no Cargo features.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `example` | `vyre_driver_reference_release_surface` | `vyre-driver-reference/examples/vyre_driver_reference_release_surface.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-reference --example vyre_driver_reference_release_surface` |
| `lib` | `vyre_driver_reference` | `vyre-driver-reference/src/lib.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-reference` |
| `test` | `backend_registration` | `vyre-driver-reference/tests/backend_registration.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-reference --test backend_registration` |
| `test` | `generated_boundary_matrix` | `vyre-driver-reference/tests/generated_boundary_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-reference --test generated_boundary_matrix` |
| `test` | `hostile_input_closure_contract` | `vyre-driver-reference/tests/hostile_input_closure_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-reference --test hostile_input_closure_contract` |
| `test` | `parity_suite` | `vyre-driver-reference/tests/parity_suite.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-reference --test parity_suite` |

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
