# Testing `vyre`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre
```

Public facade. Re-export IR, driver, runtime, and the artifact compiler. Own no logic.

The crate lives at `vyre`. The `public-facade` owner maintains its
`facade` testing contract.

## Commands

```console
./cargo_full test -p vyre
```

```console
./cargo_full test -p vyre --all-features
```

## Feature sets

- Default feature members: None
- Available manifest features: `cpu-parity`, `cuda`, `default`, `wgpu`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `example` | `vyre_release_surface` | `vyre/examples/vyre_release_surface.rs` | None | `./cargo_full test -p vyre --example vyre_release_surface` |
| `lib` | `vyre` | `vyre/src/lib.rs` | None | `./cargo_full test -p vyre` |
| `test` | `all_tests` | `vyre/tests/all_tests.rs` | None | `./cargo_full test -p vyre --test all_tests` |

## Test classes

- Public facade and feature-routing contracts
- Backend-neutral API and dispatch integration
- Documentation examples and compatibility tests

## Hardware requirements

The default suite is host-only. Tests that execute concrete GPU features require the corresponding device and driver, and a requested acquisition failure is an error.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
