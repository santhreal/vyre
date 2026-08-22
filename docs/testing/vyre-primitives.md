# Testing `vyre-primitives`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-primitives
```

Own marker types and uncomposable hardware intrinsics. A composition belongs in vyre-libs, not here.

The crate lives at `vyre-primitives`. The `primitive-library` owner maintains its
`primitives` testing contract.

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
| `test` | `consumer_boundary` | `vyre-primitives/tests/consumer_boundary.rs` | None | `./cargo_full test -p vyre-primitives --test consumer_boundary` |
| `test` | `generated_hardware_f32_matrix` | `vyre-primitives/tests/generated_hardware_f32_matrix.rs` | `hardware` | `./cargo_full test -p vyre-primitives --test generated_hardware_f32_matrix` |
| `test` | `generated_hardware_registry_matrix` | `vyre-primitives/tests/generated_hardware_registry_matrix.rs` | `hardware` | `./cargo_full test -p vyre-primitives --test generated_hardware_registry_matrix` |
| `test` | `generated_hardware_u32_matrix` | `vyre-primitives/tests/generated_hardware_u32_matrix.rs` | `hardware` | `./cargo_full test -p vyre-primitives --test generated_hardware_u32_matrix` |
| `test` | `hardware_conform` | `vyre-primitives/tests/hardware_conform.rs` | `hardware` | `./cargo_full test -p vyre-primitives --test hardware_conform` |
| `test` | `hardware_registration_safety_rules` | `vyre-primitives/tests/hardware_registration_safety_rules.rs` | None | `./cargo_full test -p vyre-primitives --test hardware_registration_safety_rules` |
| `test` | `hardware_registry_contract` | `vyre-primitives/tests/hardware_registry_contract.rs` | `hardware` | `./cargo_full test -p vyre-primitives --test hardware_registry_contract` |
| `test` | `integration` | `vyre-primitives/tests/integration.rs` | `hardware` | `./cargo_full test -p vyre-primitives --test integration` |
| `test` | `proptest_wire_roundtrip` | `vyre-primitives/tests/proptest_wire_roundtrip.rs` | None | `./cargo_full test -p vyre-primitives --test proptest_wire_roundtrip` |
| `test` | `registry_closure` | `vyre-primitives/tests/registry_closure.rs` | None | `./cargo_full test -p vyre-primitives --test registry_closure` |
| `test` | `registry_oob_clean` | `vyre-primitives/tests/registry_oob_clean.rs` | None | `./cargo_full test -p vyre-primitives --test registry_oob_clean` |
| `test` | `wire_differential_std_io` | `vyre-primitives/tests/wire_differential_std_io.rs` | None | `./cargo_full test -p vyre-primitives --test wire_differential_std_io` |
| `test` | `wire_harness_smoke_test` | `vyre-primitives/tests/wire_harness_smoke_test.rs` | None | `./cargo_full test -p vyre-primitives --test wire_harness_smoke_test` |
| `test` | `wire_pack_into_contracts` | `vyre-primitives/tests/wire_pack_into_contracts.rs` | None | `./cargo_full test -p vyre-primitives --test wire_pack_into_contracts` |

## Test classes

- Primitive builder semantics
- Reference and backend parity
- Boundary, property, and composition contracts

## Hardware requirements

Builder and reference suites are host-capable. Concrete backend parity tests require the selected device on the execution host (axiomexec) and fail visibly when unavailable.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
