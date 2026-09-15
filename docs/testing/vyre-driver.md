# Testing `vyre-driver`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-driver
```

Define backend-neutral device, target compiler registration, artifact materialization, binding, submission, completion, capability, dispatch, and evidence contracts.

The crate lives at `vyre-driver` and owns the `backend-contract` seam in the `backend-neutral` layer.

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
| `test` | `all_tests` | `vyre-driver/tests/all_tests.rs` | None | `./cargo_full test -p vyre-driver --test all_tests` |
| `test` | `all_tests_libs_compositions` | `vyre-driver/tests/all_tests_libs_compositions.rs` | `libs-compositions` | `./cargo_full test -p vyre-driver --test all_tests_libs_compositions` |
| `test` | `backend_registry` | `vyre-driver/tests/backend_registry.rs` | None | `./cargo_full test -p vyre-driver --test backend_registry` |
| `test` | `backend_registry_duplicate_provider` | `vyre-driver/tests/backend_registry_duplicate_provider.rs` | None | `./cargo_full test -p vyre-driver --test backend_registry_duplicate_provider` |
| `test` | `gap_duplicate_op_id` | `vyre-driver/tests/gap_duplicate_op_id.rs` | None | `./cargo_full test -p vyre-driver --test gap_duplicate_op_id` |
| `test` | `reference_oracle_is_never_implicit` | `vyre-driver/tests/reference_oracle_is_never_implicit.rs` | None | `./cargo_full test -p vyre-driver --test reference_oracle_is_never_implicit` |

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
