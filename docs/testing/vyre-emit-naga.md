# Testing `vyre-emit-naga`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-emit-naga
```

Consume verified lowering products and emit the primary text representation and related binary targets.

The crate lives at `vyre-emit-naga`. The `primary-text-emitter` owner maintains its
`emitter` testing contract.

## Commands

```console
./cargo_full test -p vyre-emit-naga
```

## Feature sets

This crate declares no Cargo features.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `example` | `vyre_emit_naga_release_surface` | `vyre-emit-naga/examples/vyre_emit_naga_release_surface.rs` | None | `./cargo_full test -p vyre-emit-naga --example vyre_emit_naga_release_surface` |
| `lib` | `vyre_emit_naga` | `vyre-emit-naga/src/lib.rs` | None | `./cargo_full test -p vyre-emit-naga` |
| `test` | `adversarial_emit_program_matrix` | `vyre-emit-naga/tests/adversarial_emit_program_matrix.rs` | None | `./cargo_full test -p vyre-emit-naga --test adversarial_emit_program_matrix` |
| `test` | `carrier_scope_regression` | `vyre-emit-naga/tests/carrier_scope_regression.rs` | None | `./cargo_full test -p vyre-emit-naga --test carrier_scope_regression` |
| `test` | `emitted_artifact_byte_stability` | `vyre-emit-naga/tests/emitted_artifact_byte_stability.rs` | None | `./cargo_full test -p vyre-emit-naga --test emitted_artifact_byte_stability` |
| `test` | `target_capabilities` | `vyre-emit-naga/tests/target_capabilities.rs` | None | `./cargo_full test -p vyre-emit-naga --test target_capabilities` |

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
