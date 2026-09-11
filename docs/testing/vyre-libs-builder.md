# Testing `vyre-libs-builder`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-libs-builder
```

Own shared IR composition infrastructure, child region skeletons, operand wrappers, and registration link anchors for library builders. Does not own concrete domain operations, backend lowering, or runtime execution.

The crate lives at `vyre-libs-builder` and owns the `libs-builder` seam in the `libraries` layer.

## Commands

```console
./cargo_full test -p vyre-libs-builder
```

```console
./cargo_full test -p vyre-libs-builder --all-features
```

## Feature sets

- Default feature members: `builder`, `builder-ops`, `cat-a-builder-options`, `telemetry`
- Available manifest features: `builder`, `builder-ops`, `cat-a-builder-options`, `default`, `telemetry`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `lib` | `vyre_libs_builder` | `vyre-libs-builder/src/lib.rs` | None | `./cargo_full test -p vyre-libs-builder` |
| `test` | `all_tests` | `vyre-libs-builder/tests/all_tests.rs` | None | `./cargo_full test -p vyre-libs-builder --test all_tests` |

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
