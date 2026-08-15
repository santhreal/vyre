# Testing `vyre-debug`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-debug
```

Inspect, explain, and diagnose typed programs, lowering, and product-library composition.

The crate lives at `vyre-debug`. The `debugging` owner maintains its
`tooling` testing contract.

## Commands

```console
./cargo_full test -p vyre-debug
```

## Feature sets

This crate declares no Cargo features.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `bin` | `vyre-dbg` | `vyre-debug/src/bin/vyre_dbg.rs` | None | `./cargo_full test -p vyre-debug --bin vyre-dbg` |
| `bin` | `vyre_dbg` | `vyre-debug/src/bin/vyre_dbg.rs` | None | `./cargo_full test -p vyre-debug --bin vyre_dbg` |
| `example` | `vyre_debug_release_surface` | `vyre-debug/examples/vyre_debug_release_surface.rs` | None | `./cargo_full test -p vyre-debug --example vyre_debug_release_surface` |
| `lib` | `vyre_debug` | `vyre-debug/src/lib.rs` | None | `./cargo_full test -p vyre-debug` |
| `test` | `artifact_report` | `vyre-debug/tests/artifact_report.rs` | None | `./cargo_full test -p vyre-debug --test artifact_report` |
| `test` | `cli_find_dangling_exit_codes` | `vyre-debug/tests/cli_find_dangling_exit_codes.rs` | None | `./cargo_full test -p vyre-debug --test cli_find_dangling_exit_codes` |
| `test` | `dangling_ref_contracts` | `vyre-debug/tests/dangling_ref_contracts.rs` | None | `./cargo_full test -p vyre-debug --test dangling_ref_contracts` |
| `test` | `descriptor_diff_contracts` | `vyre-debug/tests/descriptor_diff_contracts.rs` | None | `./cargo_full test -p vyre-debug --test descriptor_diff_contracts` |
| `test` | `descriptor_dump_contracts` | `vyre-debug/tests/descriptor_dump_contracts.rs` | None | `./cargo_full test -p vyre-debug --test descriptor_dump_contracts` |
| `test` | `generated_descriptor_diff_matrix` | `vyre-debug/tests/generated_descriptor_diff_matrix.rs` | None | `./cargo_full test -p vyre-debug --test generated_descriptor_diff_matrix` |
| `test` | `loop_carrier_contracts` | `vyre-debug/tests/loop_carrier_contracts.rs` | None | `./cargo_full test -p vyre-debug --test loop_carrier_contracts` |
| `test` | `registry_closure` | `vyre-debug/tests/registry_closure.rs` | None | `./cargo_full test -p vyre-debug --test registry_closure` |
| `test` | `well_formed_lowering_contracts` | `vyre-debug/tests/well_formed_lowering_contracts.rs` | None | `./cargo_full test -p vyre-debug --test well_formed_lowering_contracts` |
| `test` | `wgsl_dump_contracts` | `vyre-debug/tests/wgsl_dump_contracts.rs` | None | `./cargo_full test -p vyre-debug --test wgsl_dump_contracts` |

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
