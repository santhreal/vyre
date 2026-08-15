# Testing `vyre-macros`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-macros
```

Provide compile-time registration and declaration macros without depending on runtime crates.

The crate lives at `vyre-macros`. The `registration-macros` owner maintains its
`foundation` testing contract.

## Commands

```console
./cargo_full test -p vyre-macros
```

## Feature sets

This crate declares no Cargo features.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `example` | `vyre_macros_release_surface` | `vyre-macros/examples/vyre_macros_release_surface.rs` | None | `./cargo_full test -p vyre-macros --example vyre_macros_release_surface` |
| `lib` | `vyre_macros` | `vyre-macros/src/lib.rs` | None | `./cargo_full test -p vyre-macros` |
| `test` | `adversarial` | `vyre-macros/tests/adversarial.rs` | None | `./cargo_full test -p vyre-macros --test adversarial` |
| `test` | `ast_registry_contracts` | `vyre-macros/tests/ast_registry_contracts.rs` | None | `./cargo_full test -p vyre-macros --test ast_registry_contracts` |
| `test` | `generated_ast_registry_matrix` | `vyre-macros/tests/generated_ast_registry_matrix.rs` | None | `./cargo_full test -p vyre-macros --test generated_ast_registry_matrix` |
| `test` | `generated_metadata_matrix` | `vyre-macros/tests/generated_metadata_matrix.rs` | None | `./cargo_full test -p vyre-macros --test generated_metadata_matrix` |
| `test` | `integration` | `vyre-macros/tests/integration.rs` | None | `./cargo_full test -p vyre-macros --test integration` |
| `test` | `pass_matrix` | `vyre-macros/tests/pass_matrix.rs` | None | `./cargo_full test -p vyre-macros --test pass_matrix` |
| `test` | `release_surface_contracts` | `vyre-macros/tests/release_surface_contracts.rs` | None | `./cargo_full test -p vyre-macros --test release_surface_contracts` |
| `test` | `trybuild` | `vyre-macros/tests/trybuild.rs` | None | `./cargo_full test -p vyre-macros --test trybuild` |

## Test classes

- IR construction and serialization contracts
- Validation and optimizer semantics
- Adversarial, property, and compatibility tests

## Hardware requirements

No accelerator is required for the default suite.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
