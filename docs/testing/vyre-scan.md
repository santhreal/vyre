# Testing `vyre-scan`

Run the default crate suite from the workspace root:

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-scan
```

Own scan compilation, database codecs, artifact sessions, paging, residency, execution, and readback.

The crate lives at `vyre-scan`. The `scan-product` owner maintains its
`runtime` testing contract.

## Commands

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-scan
```

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-scan --all-features
```

## Feature sets

- Default feature members: None
- Available manifest features: `cpu-parity`, `default`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `lib` | `vyre_scan` | `vyre-scan/src/lib.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-scan` |
| `test` | `artifact_route` | `vyre-scan/tests/artifact_route.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-scan --test artifact_route` |

## Test classes

- Execution planning and cache contracts
- Persistent runtime state transitions
- IO, telemetry, and failure semantics

## Hardware requirements

No accelerator is required for the default suite.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
