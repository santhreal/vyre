# Testing `vyre-lints`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-lints
```

Enforce source-level project policies without depending on runtime crates.

The crate lives at `vyre-lints`. The `lint-policy` owner maintains its
`tooling` testing contract.

## Commands

```console
./cargo_full test -p vyre-lints
```

## Feature sets

This crate declares no Cargo features.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `bin` | `vyre-lints` | `vyre-lints/src/main.rs` | None | `./cargo_full test -p vyre-lints --bin vyre-lints` |
| `example` | `vyre_lints_release_surface` | `vyre-lints/examples/vyre_lints_release_surface.rs` | None | `./cargo_full test -p vyre-lints --example vyre_lints_release_surface` |
| `lib` | `vyre_lints` | `vyre-lints/src/lib.rs` | None | `./cargo_full test -p vyre-lints` |
| `test` | `all_tests` | `vyre-lints/tests/all_tests.rs` | None | `./cargo_full test -p vyre-lints --test all_tests` |

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
