# Testing `vyre-conform`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-conform
```

Execute production artifacts against independent reference semantics, minimize counterexamples, check algebraic laws, and issue versioned certificates and replay records through one library and thin CLI.

The crate lives at `conform/vyre-conform`. The `conformance` owner maintains its
`conformance` testing contract.

## Commands

```console
./cargo_full test -p vyre-conform
```

```console
./cargo_full test -p vyre-conform --all-features
```

```console
./cargo_full test -p vyre-conform --all-features -- --ignored --nocapture
```

## Feature sets

- Default feature members: `gpu`
- Available manifest features: `default`, `device-tests`, `gpu`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `bin` | `vyre-conform` | `conform/vyre-conform/src/main.rs` | None | `./cargo_full test -p vyre-conform --bin vyre-conform` |
| `example` | `vyre_conform_runner_release_surface` | `conform/vyre-conform/examples/vyre_conform_runner_release_surface.rs` | None | `./cargo_full test -p vyre-conform --example vyre_conform_runner_release_surface` |
| `lib` | `vyre_conform` | `conform/vyre-conform/src/lib.rs` | None | `./cargo_full test -p vyre-conform` |
| `test` | `all_tests` | `conform/vyre-conform/tests/all_tests.rs` | None | `./cargo_full test -p vyre-conform --test all_tests` |
| `test` | `all_tests_device_tests` | `conform/vyre-conform/tests/all_tests_device_tests.rs` | `device-tests` | `./cargo_full test -p vyre-conform --test all_tests_device_tests` |
| `test` | `cert_regression_pin` | `conform/vyre-conform/tests/cert_regression_pin/main.rs` | None | `./cargo_full test -p vyre-conform --test cert_regression_pin` |

## Test classes

- Case and certificate schema contracts
- Generator determinism
- Cross-backend semantic conformance

## Hardware requirements

Cross-backend certificates require every selected physical backend on the execution host. Missing selected hardware is a failed conformance run.

## Evidence outputs

- `release/evidence/conformance/release-all-backends-certificate.json`
- Command status and exact cross-backend results

## Skips and failures

The default command omits tests marked `#[ignore]`. The ignored command is the explicit physical-backend run on the execution host and cannot silently skip a selected backend.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
