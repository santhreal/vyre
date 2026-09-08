# Testing `vyre-pass-engine`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-pass-engine
```

Execute optimizer passes as Vyre Programs through compiler-owned semantic compilation and admitted artifact submission.

The crate lives at `vyre-pass-engine`. The `pass-engine` owner maintains its
`pass-engine` testing contract.

## Commands

```console
./cargo_full test -p vyre-pass-engine
```

```console
./cargo_full test -p vyre-pass-engine --all-features
```

## Feature sets

- Default feature members: `optimizer`
- Available manifest features: `all-solvers`, `default`, `optimizer`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `example` | `vyre_pass_engine_release_surface` | `vyre-pass-engine/examples/vyre_pass_engine_release_surface.rs` | None | `./cargo_full test -p vyre-pass-engine --example vyre_pass_engine_release_surface` |
| `lib` | `vyre_pass_engine` | `vyre-pass-engine/src/lib.rs` | None | `./cargo_full test -p vyre-pass-engine` |
| `test` | `all_tests` | `vyre-pass-engine/tests/all_tests.rs` | None | `./cargo_full test -p vyre-pass-engine --test all_tests` |
| `test` | `all_tests_all_solvers` | `vyre-pass-engine/tests/all_tests_all_solvers.rs` | `all-solvers` | `./cargo_full test -p vyre-pass-engine --test all_tests_all_solvers` |
| `test` | `all_tests_optimizer` | `vyre-pass-engine/tests/all_tests_optimizer.rs` | `optimizer` | `./cargo_full test -p vyre-pass-engine --test all_tests_optimizer` |

## Test classes

- Encoded-pass Program semantics
- Reference-parity of every dispatched pass
- Determinism and boundary tests

## Hardware requirements

No accelerator is required for the default suite.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
