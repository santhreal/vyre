# Testing `vyre-emit-metal`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-emit-metal
```

Consume verified lowering products and emit native Apple shader source through the shared emitter path.

The crate lives at `vyre-emit-metal`. The `metal-emitter` owner maintains its
`emitter` testing contract.

## Commands

```console
./cargo_full test -p vyre-emit-metal
```

## Feature sets

This crate declares no Cargo features.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `example` | `vyre_emit_metal_release_surface` | `vyre-emit-metal/examples/vyre_emit_metal_release_surface.rs` | None | `./cargo_full test -p vyre-emit-metal --example vyre_emit_metal_release_surface` |
| `lib` | `vyre_emit_metal` | `vyre-emit-metal/src/lib.rs` | None | `./cargo_full test -p vyre-emit-metal` |
| `test` | `adversarial_emit_program_matrix` | `vyre-emit-metal/tests/adversarial_emit_program_matrix.rs` | None | `./cargo_full test -p vyre-emit-metal --test adversarial_emit_program_matrix` |

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
