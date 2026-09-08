# Testing `vyre-emit-spirv`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-emit-spirv
```

Consume verified lowering products and emit SPIR-V artifacts through the shared writer.

The crate lives at `vyre-emit-spirv`. The `spirv-emitter` owner maintains its
`emitter` testing contract.

## Commands

```console
./cargo_full test -p vyre-emit-spirv
```

## Feature sets

This crate declares no Cargo features.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `example` | `vyre_emit_spirv_release_surface` | `vyre-emit-spirv/examples/vyre_emit_spirv_release_surface.rs` | None | `./cargo_full test -p vyre-emit-spirv --example vyre_emit_spirv_release_surface` |
| `lib` | `vyre_emit_spirv` | `vyre-emit-spirv/src/lib.rs` | None | `./cargo_full test -p vyre-emit-spirv` |
| `test` | `all_tests` | `vyre-emit-spirv/tests/all_tests.rs` | None | `./cargo_full test -p vyre-emit-spirv --test all_tests` |

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
