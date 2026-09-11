# Testing `structure-gate`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p structure-gate
```

Enforce the crate roster, one operation identity per semantic operation, one home per concept, and one place per module. Depends on no vyre crate so it keeps running while the workspace does not compile.

The crate lives at `structure-gate` and owns the `source-structure` seam in the `standalone-tooling` layer.

## Commands

```console
./cargo_full test -p structure-gate
```

## Feature sets

This crate declares no Cargo features.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `bin` | `structure-gate` | `structure-gate/src/main.rs` | None | `./cargo_full test -p structure-gate --bin structure-gate` |
| `lib` | `structure_gate` | `structure-gate/src/lib.rs` | None | `./cargo_full test -p structure-gate` |
| `test` | `all_tests` | `structure-gate/tests/all_tests.rs` | None | `./cargo_full test -p structure-gate --test all_tests` |

## Test classes

- Repository contract scans
- Checkout resolution and path boundaries
- Failure diagnostics that name the correction

## Hardware requirements

No accelerator is required for the default suite.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
