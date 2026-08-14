# Testing `xtask-registry`

Run the default crate suite from the workspace root:

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p xtask-registry
```

Own the xtask subcommands that must observe the live operation registry, the primitive catalog behind it, or a linked backend driver.

The crate lives at `xtask-registry`. The `release-tooling` owner maintains its
`tooling` testing contract.

## Commands

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p xtask-registry
```

## Feature sets

This crate declares no Cargo features.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `bin` | `xtask-registry` | `xtask-registry/src/main.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p xtask-registry --bin xtask-registry` |
| `lib` | `xtask_registry` | `xtask-registry/src/lib.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p xtask-registry` |
| `test` | `cli_docs` | `xtask-registry/tests/cli_docs.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p xtask-registry --test cli_docs` |
| `test` | `operation_schema` | `xtask-registry/tests/operation_schema.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p xtask-registry --test operation_schema` |

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
