# Testing `vyre-libs-analysis`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-libs-analysis
```

Own compiler-internal static analysis, cost models, dataflow fixpoint routines, and diagnostic aggregation. Does not own pass engine scheduling, backend lowering, or runtime execution.

The crate lives at `vyre-libs-analysis`. The `product-libraries` owner maintains its
`libraries` testing contract.

## Commands

```console
./cargo_full test -p vyre-libs-analysis
```

```console
./cargo_full test -p vyre-libs-analysis --all-features
```

## Feature sets

- Default feature members: `analysis`
- Available manifest features: `analysis`, `default`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `lib` | `vyre_libs_analysis` | `vyre-libs-analysis/src/lib.rs` | None | `./cargo_full test -p vyre-libs-analysis` |
| `test` | `all_tests` | `vyre-libs-analysis/tests/all_tests.rs` | None | `./cargo_full test -p vyre-libs-analysis --test all_tests` |

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
