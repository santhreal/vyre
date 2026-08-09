# Testing `vyre-lints`

Run the default crate suite from the workspace root:

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-lints
```

Enforce source-level project policies without depending on runtime crates.

The crate lives at `vyre-lints`. The `lint-policy` owner maintains its
`tooling` testing contract.

## Commands

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-lints
```

## Feature sets

This crate declares no Cargo features.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `bin` | `vyre-lints` | `vyre-lints/src/main.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-lints --bin vyre-lints` |
| `example` | `vyre_lints_release_surface` | `vyre-lints/examples/vyre_lints_release_surface.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-lints --example vyre_lints_release_surface` |
| `lib` | `vyre_lints` | `vyre-lints/src/lib.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-lints` |
| `test` | `consumer_coupling` | `vyre-lints/tests/consumer_coupling.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-lints --test consumer_coupling` |
| `test` | `gpu_skip_guards` | `vyre-lints/tests/gpu_skip_guards.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-lints --test gpu_skip_guards` |
| `test` | `module_forks` | `vyre-lints/tests/module_forks.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-lints --test module_forks` |
| `test` | `production_cpu_fallbacks` | `vyre-lints/tests/production_cpu_fallbacks.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-lints --test production_cpu_fallbacks` |
| `test` | `raw_ir_in_libs` | `vyre-lints/tests/raw_ir_in_libs.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-lints --test raw_ir_in_libs` |

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
