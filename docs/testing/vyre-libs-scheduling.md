# Testing `vyre-libs-scheduling`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-libs-scheduling
```

Own compiler-internal scheduling, fusion, batching, and dispatch strategy compositions. Does not own physical lowering, device execution, or runtime lifecycle policy.

The crate lives at `vyre-libs-scheduling`. The `product-libraries` owner maintains its
`libraries` testing contract.

## Commands

```console
./cargo_full test -p vyre-libs-scheduling
```

```console
./cargo_full test -p vyre-libs-scheduling --all-features
```

## Feature sets

- Default feature members: `scheduling`
- Available manifest features: `default`, `scheduling`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `lib` | `vyre_libs_scheduling` | `vyre-libs-scheduling/src/lib.rs` | None | `./cargo_full test -p vyre-libs-scheduling` |
| `test` | `all_tests` | `vyre-libs-scheduling/tests/all_tests.rs` | None | `./cargo_full test -p vyre-libs-scheduling --test all_tests` |

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
