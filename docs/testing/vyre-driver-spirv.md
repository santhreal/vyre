# Testing `vyre-driver-spirv`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-driver-spirv
```

Own SPIR-V target compilation, immutable module-bundle emission, Vulkan materialization and dispatch integration, and backend evidence.

The crate lives at `vyre-driver-spirv`. The `spirv-driver` owner maintains its
`concrete-backend` testing contract.

## Commands

```console
./cargo_full test -p vyre-driver-spirv
```

```console
./cargo_full test -p vyre-driver-spirv --all-features
```

## Feature sets

- Default feature members: None
- Available manifest features: `default`, `device-tests`, `spirv-val`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `example` | `vulkan_probe` | `vyre-driver-spirv/examples/vulkan_probe.rs` | None | `./cargo_full test -p vyre-driver-spirv --example vulkan_probe` |
| `lib` | `vyre_driver_spirv` | `vyre-driver-spirv/src/lib.rs` | None | `./cargo_full test -p vyre-driver-spirv` |
| `test` | `dispatch` | `vyre-driver-spirv/tests/dispatch.rs` | `device-tests` | `./cargo_full test -p vyre-driver-spirv --test dispatch` |
| `test` | `hostile_input_closure_contract` | `vyre-driver-spirv/tests/hostile_input_closure_contract.rs` | `device-tests` | `./cargo_full test -p vyre-driver-spirv --test hostile_input_closure_contract` |
| `test` | `resident_multi_entry_submission` | `vyre-driver-spirv/tests/resident_multi_entry_submission.rs` | None | `./cargo_full test -p vyre-driver-spirv --test resident_multi_entry_submission` |
| `test` | `shared_target_contract_discrimination` | `vyre-driver-spirv/tests/shared_target_contract_discrimination.rs` | None | `./cargo_full test -p vyre-driver-spirv --test shared_target_contract_discrimination` |
| `test` | `spirv_parity` | `vyre-driver-spirv/tests/spirv_parity.rs` | `spirv-val` | `./cargo_full test -p vyre-driver-spirv --test spirv_parity` |
| `test` | `target_payload_admission_contract` | `vyre-driver-spirv/tests/target_payload_admission_contract.rs` | None | `./cargo_full test -p vyre-driver-spirv --test target_payload_admission_contract` |

## Test classes

- Device and capability contracts
- Lowering and artifact semantics
- Dispatch, graph, memory, and backend parity tests

## Hardware requirements

The default suite validates lowering without a device. Physical Vulkan-style execution tests require a compatible adapter on the execution host (axiomexec) and must report acquisition failure.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
