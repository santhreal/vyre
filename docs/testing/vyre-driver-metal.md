# Testing `vyre-driver-metal`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-driver-metal
```

Own pure MSL target compilation, native Apple device acquisition, materialization, dispatch, and backend evidence.

The crate lives at `vyre-driver-metal`. The `metal-driver` owner maintains its
`concrete-backend` testing contract.

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
| `test` | `apple_math_comparators` | `vyre-driver-metal/tests/apple_math_comparators.rs` | None | `./cargo_full test -p vyre-driver-metal --test apple_math_comparators` |
| `test` | `metal_hazard_certificates` | `vyre-driver-metal/tests/metal_hazard_certificates.rs` | None | `./cargo_full test -p vyre-driver-metal --test metal_hazard_certificates` |
| `test` | `metal_icb_dispatch_replay` | `vyre-driver-metal/tests/metal_icb_dispatch_replay.rs` | None | `./cargo_full test -p vyre-driver-metal --test metal_icb_dispatch_replay` |
| `test` | `metal_simd_scan_plan_registry` | `vyre-driver-metal/tests/metal_simd_scan_plan_registry.rs` | None | `./cargo_full test -p vyre-driver-metal --test metal_simd_scan_plan_registry` |
| `test` | `resident_async` | `vyre-driver-metal/tests/resident_async.rs` | None | `./cargo_full test -p vyre-driver-metal --test resident_async` |
| `test` | `target_compiler` | `vyre-driver-metal/tests/target_compiler.rs` | None | `./cargo_full test -p vyre-driver-metal --test target_compiler` |

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
