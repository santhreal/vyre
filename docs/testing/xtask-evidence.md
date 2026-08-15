# Testing `xtask-evidence`

Run the default crate suite from the workspace root:

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p xtask-evidence
```

Own the xtask subcommands that decide whether a recorded benchmark or release measurement still describes this tree.

The crate lives at `xtask-evidence`. The `release-tooling` owner maintains its
`tooling` testing contract.

## Commands

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p xtask-evidence
```

## Feature sets

This crate declares no Cargo features.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `bin` | `xtask-evidence` | `xtask-evidence/src/main.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p xtask-evidence --bin xtask-evidence` |
| `lib` | `xtask_evidence` | `xtask-evidence/src/lib.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p xtask-evidence` |
| `test` | `release_evidence_dispatch` | `xtask-evidence/tests/release_evidence_dispatch.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p xtask-evidence --test release_evidence_dispatch` |

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
