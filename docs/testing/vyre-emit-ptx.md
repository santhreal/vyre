# Testing `vyre-emit-ptx`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-emit-ptx
```

Consume verified lowering products and emit the primary binary backend text artifact.

The crate lives at `vyre-emit-ptx` and owns the `primary-binary-emitter` seam in the `emitter` layer.

## Commands

```console
./cargo_full test -p vyre-emit-ptx
```

```console
./cargo_full test -p vyre-emit-ptx --all-features
```

## Feature sets

- Default feature members: None
- Available manifest features: `default`, `nvrtc`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `example` | `vyre_emit_ptx_release_surface` | `vyre-emit-ptx/examples/vyre_emit_ptx_release_surface.rs` | None | `./cargo_full test -p vyre-emit-ptx --example vyre_emit_ptx_release_surface` |
| `lib` | `vyre_emit_ptx` | `vyre-emit-ptx/src/lib.rs` | None | `./cargo_full test -p vyre-emit-ptx` |
| `test` | `all_tests` | `vyre-emit-ptx/tests/all_tests.rs` | None | `./cargo_full test -p vyre-emit-ptx --test all_tests` |

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
