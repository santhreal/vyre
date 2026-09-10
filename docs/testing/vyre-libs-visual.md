# Testing `vyre-libs-visual`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-libs-visual
```

Own visual rendering and compositing effects including blur, shadow, blend modes, gradients, and color conversions. Does not own display presentation, backend lowering, or runtime execution.

The crate lives at `vyre-libs-visual` and owns the `libs-visual` seam in the `libraries` layer.

## Commands

```console
./cargo_full test -p vyre-libs-visual
```

```console
./cargo_full test -p vyre-libs-visual --all-features
```

## Feature sets

- Default feature members: `visual`
- Available manifest features: `default`, `visual`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `lib` | `vyre_libs_visual` | `vyre-libs-visual/src/lib.rs` | None | `./cargo_full test -p vyre-libs-visual` |
| `test` | `all_tests` | `vyre-libs-visual/tests/all_tests.rs` | None | `./cargo_full test -p vyre-libs-visual --test all_tests` |

## Test classes

- Product-library exact behavior
- Primitive-to-library composition
- Reference and backend parity

## Hardware requirements

No accelerator is required for the default suite.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
