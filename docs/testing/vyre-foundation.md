# Testing `vyre-foundation`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-foundation
```

Own validated ProgramGraph, versioned schedule-free LogicalProgramGraph domains, versioned backend-neutral schedule IR and transform legality, semantic identity, neutral schedule-constraint composition, diagnostics, serialization, semantic operation registration, and backend-neutral optimization.

The crate lives at `vyre-foundation` and owns the `foundation-ir` seam in the `foundation` layer.

## Commands

```console
./cargo_full test -p vyre-foundation
```

```console
./cargo_full test -p vyre-foundation --all-features
```

## Feature sets

- Default feature members: None
- Available manifest features: `default`, `serde`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `bench` | `optimizer_pipeline` | `vyre-foundation/benches/optimizer_pipeline.rs` | None | `./cargo_full test -p vyre-foundation --bench optimizer_pipeline` |
| `example` | `vyre_foundation_release_surface` | `vyre-foundation/examples/vyre_foundation_release_surface.rs` | None | `./cargo_full test -p vyre-foundation --example vyre_foundation_release_surface` |
| `lib` | `vyre_foundation` | `vyre-foundation/src/lib.rs` | None | `./cargo_full test -p vyre-foundation` |
| `test` | `all_tests` | `vyre-foundation/tests/all_tests.rs` | None | `./cargo_full test -p vyre-foundation --test all_tests` |

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
