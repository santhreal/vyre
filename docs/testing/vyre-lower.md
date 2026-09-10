# Testing `vyre-lower`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-lower
```

Consume validated selected schedule phases and semantic programs, then own the single validated PhysicalKernel lowering boundary plus its backend-neutral pre-emission transforms.

The crate lives at `vyre-lower` and owns the `lowering` seam in the `lowering` layer.

## Commands

```console
./cargo_full test -p vyre-lower
```

```console
./cargo_full test -p vyre-lower --all-features
```

## Feature sets

- Default feature members: None
- Available manifest features: `default`, `test-fixtures`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `lib` | `vyre_lower` | `vyre-lower/src/lib.rs` | None | `./cargo_full test -p vyre-lower` |
| `test` | `all_tests` | `vyre-lower/tests/all_tests.rs` | None | `./cargo_full test -p vyre-lower --test all_tests` |

## Test classes

- Backend-neutral lowering transforms
- Pre-emission invariants
- Invalid IR and target rejection

## Hardware requirements

No accelerator is required for the default suite.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
