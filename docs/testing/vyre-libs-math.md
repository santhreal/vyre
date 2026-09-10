# Testing `vyre-libs-math`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-libs-math
```

Own linear algebra, matrix operations, broadcasting, scans, optimization builders, and geometric algebra IR compositions. Does not own neural network layers, backend lowering, or runtime execution.

The crate lives at `vyre-libs-math` and owns the `libs-math` seam in the `libraries` layer.

## Commands

```console
./cargo_full test -p vyre-libs-math
```

```console
./cargo_full test -p vyre-libs-math --all-features
```

## Feature sets

- Default feature members: `math`, `geom`, `opt`, `representation`
- Available manifest features: `default`, `geom`, `math`, `math-algebra`, `math-broadcast`, `math-dialect`, `math-kernels`, `math-linalg`, `math-scan`, `math-succinct`, `opt`, `representation`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `lib` | `vyre_libs_math` | `vyre-libs-math/src/lib.rs` | None | `./cargo_full test -p vyre-libs-math` |
| `test` | `all_tests` | `vyre-libs-math/tests/all_tests.rs` | None | `./cargo_full test -p vyre-libs-math --test all_tests` |

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
