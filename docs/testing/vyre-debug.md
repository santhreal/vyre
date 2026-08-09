# Testing `vyre-debug`

Run the default crate suite from the workspace root:

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-debug
```

Inspect, explain, and diagnose typed programs, lowering, and product-library composition.

The crate lives at `vyre-debug`. The `debugging` owner maintains its
`tooling` testing contract.

## Commands

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-debug
```

## Feature sets

This crate declares no Cargo features.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `bin` | `vyre-dbg` | `vyre-debug/src/bin/vyre_dbg.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-debug --bin vyre-dbg` |
| `bin` | `vyre_dbg` | `vyre-debug/src/bin/vyre_dbg.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-debug --bin vyre_dbg` |
| `example` | `vyre_debug_release_surface` | `vyre-debug/examples/vyre_debug_release_surface.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-debug --example vyre_debug_release_surface` |
| `lib` | `vyre_debug` | `vyre-debug/src/lib.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-debug` |
| `test` | `artifact_report` | `vyre-debug/tests/artifact_report.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-debug --test artifact_report` |
| `test` | `carriers_tests` | `vyre-debug/tests/carriers_tests.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-debug --test carriers_tests` |
| `test` | `cli_tests` | `vyre-debug/tests/cli_tests.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-debug --test cli_tests` |
| `test` | `dangling_tests` | `vyre-debug/tests/dangling_tests.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-debug --test dangling_tests` |
| `test` | `descriptor_diff_tests` | `vyre-debug/tests/descriptor_diff_tests.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-debug --test descriptor_diff_tests` |
| `test` | `descriptor_dump_tests` | `vyre-debug/tests/descriptor_dump_tests.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-debug --test descriptor_dump_tests` |
| `test` | `generated_descriptor_diff_matrix` | `vyre-debug/tests/generated_descriptor_diff_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-debug --test generated_descriptor_diff_matrix` |
| `test` | `well_formed_lowering_contracts` | `vyre-debug/tests/well_formed_lowering_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-debug --test well_formed_lowering_contracts` |
| `test` | `wgsl_tests` | `vyre-debug/tests/wgsl_tests.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-debug --test wgsl_tests` |

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
