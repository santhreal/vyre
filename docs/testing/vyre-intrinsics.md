# Testing `vyre-intrinsics`

Run the default crate suite from the workspace root:

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-intrinsics
```

Own registered hardware-mapped intrinsic contracts and their neutral program builders.

The crate lives at `vyre-intrinsics`. The `hardware-intrinsics` owner maintains its
`intrinsics` testing contract.

## Commands

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-intrinsics
```

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-intrinsics --all-features
```

## Feature sets

- Default feature members: `all`, `subgroup-ops`
- Available manifest features: `all`, `default`, `hardware`, `subgroup-ops`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `example` | `vyre_intrinsics_release_surface` | `vyre-intrinsics/examples/vyre_intrinsics_release_surface.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-intrinsics --example vyre_intrinsics_release_surface` |
| `lib` | `vyre_intrinsics` | `vyre-intrinsics/src/lib.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-intrinsics` |
| `test` | `generated_hardware_f32_matrix` | `vyre-intrinsics/tests/generated_hardware_f32_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-intrinsics --test generated_hardware_f32_matrix` |
| `test` | `generated_hardware_registry_matrix` | `vyre-intrinsics/tests/generated_hardware_registry_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-intrinsics --test generated_hardware_registry_matrix` |
| `test` | `generated_hardware_u32_matrix` | `vyre-intrinsics/tests/generated_hardware_u32_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-intrinsics --test generated_hardware_u32_matrix` |
| `test` | `hardware_conform` | `vyre-intrinsics/tests/hardware_conform.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-intrinsics --test hardware_conform` |
| `test` | `registry_contract` | `vyre-intrinsics/tests/registry_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-intrinsics --test registry_contract` |

## Test classes

- Intrinsic registration and contract semantics
- Reference oracle parity
- Backend lowering and algebraic laws

## Hardware requirements

Registration and reference-oracle tests are host-capable. Concrete lowering parity requires the selected backend device and must surface unsupported hardware.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
