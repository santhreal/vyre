# Testing `vyre-libs-reduce`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-libs-reduce
```

Own workgroup reduction trees, atomic scalar reductions, and prefix scan IR compositions. Does not own domain-specific algorithms, backend lowering, or runtime execution.

The crate lives at `vyre-libs-reduce`. The `product-libraries` owner maintains its
`libraries` testing contract.

## Commands

```console
./cargo_full test -p vyre-libs-reduce
```

```console
./cargo_full test -p vyre-libs-reduce --all-features
```

## Feature sets

- Default feature members: `reduce`
- Available manifest features: `default`, `reduce`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `lib` | `vyre_libs_reduce` | `vyre-libs-reduce/src/lib.rs` | None | `./cargo_full test -p vyre-libs-reduce` |
| `test` | `all_tests` | `vyre-libs-reduce/tests/all_tests.rs` | None | `./cargo_full test -p vyre-libs-reduce --test all_tests` |

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
