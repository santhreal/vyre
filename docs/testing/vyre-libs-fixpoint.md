# Testing `vyre-libs-fixpoint`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-libs-fixpoint
```

Own deterministic fixpoint iteration kernels, routing contracts, and grid synchronization barriers. Does not own domain dataflow analysis, backend lowering, or runtime execution.

The crate lives at `vyre-libs-fixpoint` and owns the `libs-fixpoint` seam in the `libraries` layer.

## Commands

```console
./cargo_full test -p vyre-libs-fixpoint
```

```console
./cargo_full test -p vyre-libs-fixpoint --all-features
```

## Feature sets

- Default feature members: `fixpoint`
- Available manifest features: `default`, `fixpoint`, `test-fixtures`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `lib` | `vyre_libs_fixpoint` | `vyre-libs-fixpoint/src/lib.rs` | None | `./cargo_full test -p vyre-libs-fixpoint` |
| `test` | `all_tests` | `vyre-libs-fixpoint/tests/all_tests.rs` | None | `./cargo_full test -p vyre-libs-fixpoint --test all_tests` |

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
