# Testing `vyre-libs-bitset`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-libs-bitset
```

Own packed u32 bitset operations, word utilities, and logical bitwise IR compositions. Does not own reduction tree implementations, backend lowering, or runtime execution.

The crate lives at `vyre-libs-bitset` and owns the `libs-bitset` seam in the `libraries` layer.

## Commands

```console
./cargo_full test -p vyre-libs-bitset
```

```console
./cargo_full test -p vyre-libs-bitset --all-features
```

## Feature sets

- Default feature members: `bitset`, `logical`
- Available manifest features: `bitset`, `default`, `logical`, `test-fixtures`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `lib` | `vyre_libs_bitset` | `vyre-libs-bitset/src/lib.rs` | None | `./cargo_full test -p vyre-libs-bitset` |
| `test` | `all_tests` | `vyre-libs-bitset/tests/all_tests.rs` | None | `./cargo_full test -p vyre-libs-bitset --test all_tests` |

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
