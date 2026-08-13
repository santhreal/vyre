# Testing `vyre-driver-spirv`

Run the default crate suite from the workspace root:

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-spirv
```

Own SPIR-V target compilation, immutable module-bundle emission, Vulkan materialization and dispatch integration, and backend evidence.

The crate lives at `vyre-driver-spirv`. The `spirv-driver` owner maintains its
`concrete-backend` testing contract.

## Commands

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-spirv
```

## Feature sets

This crate declares no Cargo features.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `example` | `vulkan_probe` | `vyre-driver-spirv/examples/vulkan_probe.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-spirv --example vulkan_probe` |
| `lib` | `vyre_driver_spirv` | `vyre-driver-spirv/src/lib.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-spirv` |
| `test` | `dispatch` | `vyre-driver-spirv/tests/dispatch.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-spirv --test dispatch` |
| `test` | `spirv_parity` | `vyre-driver-spirv/tests/spirv_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-spirv --test spirv_parity` |
| `test` | `target_payload_admission_contract` | `vyre-driver-spirv/tests/target_payload_admission_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-spirv --test target_payload_admission_contract` |

## Test classes

- Device and capability contracts
- Lowering and artifact semantics
- Dispatch, graph, memory, and backend parity tests

## Hardware requirements

The default suite validates lowering without a device. Physical Vulkan-style execution tests require a compatible adapter and must report acquisition failure.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
