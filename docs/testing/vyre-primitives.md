# Testing `vyre-primitives`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-primitives
```

Own marker types and uncomposable hardware intrinsics. A composition belongs in vyre-libs, not here.

The crate lives at `vyre-primitives` and owns the `primitive-library` seam in the `primitives` layer.

## Commands

```console
./cargo_full test -p vyre-primitives
```

```console
./cargo_full test -p vyre-primitives --all-features
```

## Feature sets

- Default feature members: None
- Available manifest features: `cpu-parity`, `default`, `gpu`, `hardware`, `inventory-registry`, `vyre-foundation`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `bench` | `wire_throughput` | `vyre-primitives/benches/wire_throughput.rs` | None | `./cargo_full test -p vyre-primitives --bench wire_throughput` |
| `example` | `vyre_primitives_release_surface` | `vyre-primitives/examples/vyre_primitives_release_surface.rs` | None | `./cargo_full test -p vyre-primitives --example vyre_primitives_release_surface` |
| `example` | `wire_harness_smoke` | `vyre-primitives/examples/wire_harness_smoke.rs` | None | `./cargo_full test -p vyre-primitives --example wire_harness_smoke` |
| `lib` | `vyre_primitives` | `vyre-primitives/src/lib.rs` | None | `./cargo_full test -p vyre-primitives` |
| `test` | `all_tests` | `vyre-primitives/tests/all_tests.rs` | None | `./cargo_full test -p vyre-primitives --test all_tests` |
| `test` | `all_tests_hardware` | `vyre-primitives/tests/all_tests_hardware.rs` | `hardware` | `./cargo_full test -p vyre-primitives --test all_tests_hardware` |

## Test classes

- Primitive builder semantics
- Reference and backend parity
- Boundary, property, and composition contracts

## Hardware requirements

Builder and reference suites are host-capable. Concrete backend parity tests require the selected device on the execution host and fail visibly when unavailable.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
