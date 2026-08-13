# Testing `structure-gate`

Run the default crate suite from the workspace root:

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p structure-gate
```

Enforce the crate roster, one operation identity per semantic operation, and one home per concept. Depends on no vyre crate so it keeps running while the workspace does not compile.

The crate lives at `structure-gate`. The `release-tooling` owner maintains its
`tooling` testing contract.

## Commands

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p structure-gate
```

## Feature sets

This crate declares no Cargo features.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `bin` | `structure-gate` | `structure-gate/src/main.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p structure-gate --bin structure-gate` |
| `lib` | `structure_gate` | `structure-gate/src/lib.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p structure-gate` |
| `test` | `crate_structure_contracts` | `structure-gate/tests/crate_structure_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p structure-gate --test crate_structure_contracts` |
| `test` | `materializer_admission` | `structure-gate/tests/materializer_admission.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p structure-gate --test materializer_admission` |

## Test classes

- Command and policy behavior
- Evidence schema and regeneration contracts
- Failure diagnostics and repository boundaries

## Hardware requirements

No accelerator is required for the default suite.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
