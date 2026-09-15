# Testing `xtask-registry`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p xtask-registry
```

Own the xtask subcommands that must observe the live operation registry, the primitive catalog behind it, or a linked backend driver.

The crate lives at `xtask-registry` and owns the `live-registry-gates` seam in the `tooling` layer.

## Commands

```console
./cargo_full test -p xtask-registry
```

## Feature sets

This crate declares no Cargo features.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `bin` | `xtask-registry` | `xtask-registry/src/main.rs` | None | `./cargo_full test -p xtask-registry --bin xtask-registry` |
| `lib` | `xtask_registry` | `xtask-registry/src/lib.rs` | None | `./cargo_full test -p xtask-registry` |
| `test` | `registry_contracts` | `xtask-registry/tests/registry_contracts/main.rs` | None | `./cargo_full test -p xtask-registry --test registry_contracts` |

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
