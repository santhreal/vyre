# Testing `vyre-driver-metal`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-driver-metal
```

Own pure MSL target compilation, native Apple device acquisition, materialization, dispatch, and backend evidence.

The crate lives at `vyre-driver-metal` and owns the `metal-driver` seam in the `concrete-backend` layer.

## Commands

```console
./cargo_full test -p vyre-driver-metal
```

```console
./cargo_full test -p vyre-driver-metal --all-features
```

```console
./cargo_full test -p vyre-driver-metal -- --ignored --nocapture
```

## Feature sets

- Default feature members: None
- Available manifest features: `default`, `device-tests`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `example` | `metal_release_surface` | `vyre-driver-metal/examples/metal_release_surface.rs` | None | `./cargo_full test -p vyre-driver-metal --example metal_release_surface` |
| `lib` | `vyre_driver_metal` | `vyre-driver-metal/src/lib.rs` | None | `./cargo_full test -p vyre-driver-metal` |
| `test` | `all_tests` | `vyre-driver-metal/tests/all_tests.rs` | None | `./cargo_full test -p vyre-driver-metal --test all_tests` |

## Test classes

- Device and capability contracts
- Lowering and artifact semantics
- Dispatch, graph, memory, and backend parity tests

## Hardware requirements

Native device execution requires macOS or iOS with a Metal-capable device. Other targets must prove the explicit unsupported error instead of silently substituting another backend.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command omits only tests marked `#[ignore]`. Run physical device tests on Apple hardware; non-Apple contract tests must execute and assert the unsupported result.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
