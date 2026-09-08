# Testing `vyre-spec`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-spec
```

Own stable schemas, operation definitions, and compatibility contracts without runtime dependencies.

The crate lives at `vyre-spec`. The `specification` owner maintains its
`foundation` testing contract.

## Commands

```console
./cargo_full test -p vyre-spec
```

## Feature sets

This crate declares no Cargo features.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `example` | `vyre_spec_release_surface` | `vyre-spec/examples/vyre_spec_release_surface.rs` | None | `./cargo_full test -p vyre-spec --example vyre_spec_release_surface` |
| `lib` | `vyre_spec` | `vyre-spec/src/lib.rs` | None | `./cargo_full test -p vyre-spec` |
| `test` | `all_tests` | `vyre-spec/tests/all_tests.rs` | None | `./cargo_full test -p vyre-spec --test all_tests` |

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
