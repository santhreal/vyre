# Testing `vyre-harness`

Run the default crate suite from the workspace root:

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-harness
```

Provide reusable backend-neutral harness utilities for executing and comparing programs.

The crate lives at `vyre-harness`. The `runtime-harness` owner maintains its
`tooling` testing contract.

## Commands

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-harness
```

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-harness --all-features
```

## Feature sets

- Default feature members: None
- Available manifest features: `default`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `example` | `vyre_harness_release_surface` | `vyre-harness/examples/vyre_harness_release_surface.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-harness --example vyre_harness_release_surface` |
| `lib` | `vyre_harness` | `vyre-harness/src/lib.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-harness` |
| `test` | `categorical_laws_proptest` | `vyre-harness/tests/categorical_laws_proptest.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-harness --test categorical_laws_proptest` |
| `test` | `op_tier_classification` | `vyre-harness/tests/op_tier_classification.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-harness --test op_tier_classification` |
| `test` | `primitive_vs_consumer` | `vyre-harness/tests/primitive_vs_consumer.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-harness --test primitive_vs_consumer` |
| `test` | `provenance_closure` | `vyre-harness/tests/provenance_closure.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-harness --test provenance_closure` |
| `test` | `self_consumer_conform` | `vyre-harness/tests/self_consumer_conform.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-harness --test self_consumer_conform` |
| `test` | `sheaf_clustering` | `vyre-harness/tests/sheaf_clustering.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-harness --test sheaf_clustering` |

## Test classes

- Command and policy behavior
- Evidence schema and regeneration contracts
- Failure diagnostics and repository boundaries

## Hardware requirements

No accelerator is required for the default suite.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
