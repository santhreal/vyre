# Testing `vyre-alloc-probe`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-alloc-probe
```

Count heap traffic per thread behind a GlobalAlloc wrapper, so an allocation budget measures the code under test rather than the test schedule. Depends on no vyre crate, so a harness binary links it at run time and a driver suite links it as a dev-dependency.

The crate lives at `vyre-alloc-probe`. The `benchmarks` owner maintains its
`standalone-tooling` testing contract.

## Commands

```console
./cargo_full test -p vyre-alloc-probe
```

## Feature sets

This crate declares no Cargo features.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `lib` | `vyre_alloc_probe` | `vyre-alloc-probe/src/lib.rs` | None | `./cargo_full test -p vyre-alloc-probe` |
| `test` | `all_tests` | `vyre-alloc-probe/tests/all_tests.rs` | None | `./cargo_full test -p vyre-alloc-probe --test all_tests` |

## Test classes

- Repository contract scans
- Checkout resolution and path boundaries
- Failure diagnostics that name the correction

## Hardware requirements

No accelerator is required for the default suite.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
