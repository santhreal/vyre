# Testing `vyre-emit-spirv`

Run the default crate suite from the workspace root:

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-emit-spirv
```

Consume verified lowering products and emit SPIR-V artifacts through the shared writer.

The crate lives at `vyre-emit-spirv`. The `spirv-emitter` owner maintains its
`emitter` testing contract.

## Commands

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-emit-spirv
```

## Feature sets

This crate declares no Cargo features.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `example` | `vyre_emit_spirv_release_surface` | `vyre-emit-spirv/examples/vyre_emit_spirv_release_surface.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-emit-spirv --example vyre_emit_spirv_release_surface` |
| `lib` | `vyre_emit_spirv` | `vyre-emit-spirv/src/lib.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-emit-spirv` |
| `test` | `adversarial_emit_program_matrix` | `vyre-emit-spirv/tests/adversarial_emit_program_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-emit-spirv --test adversarial_emit_program_matrix` |
| `test` | `cross_emitter_parity` | `vyre-emit-spirv/tests/cross_emitter_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-emit-spirv --test cross_emitter_parity` |
| `test` | `emitted_artifact_byte_stability` | `vyre-emit-spirv/tests/emitted_artifact_byte_stability.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-emit-spirv --test emitted_artifact_byte_stability` |
| `test` | `generated_emit_descriptor_matrix` | `vyre-emit-spirv/tests/generated_emit_descriptor_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-emit-spirv --test generated_emit_descriptor_matrix` |
| `test` | `target_capabilities` | `vyre-emit-spirv/tests/target_capabilities.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-emit-spirv --test target_capabilities` |

## Test classes

- Target artifact emission
- Instruction and layout lowering
- Determinism and unsupported-operation diagnostics

## Hardware requirements

No accelerator is required for the default suite.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
