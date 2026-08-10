# Testing `vyre-emit-ptx`

Run the default crate suite from the workspace root:

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-emit-ptx
```

Consume verified lowering products and emit the primary binary backend text artifact.

The crate lives at `vyre-emit-ptx`. The `primary-binary-emitter` owner maintains its
`emitter` testing contract.

## Commands

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-emit-ptx
```

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-emit-ptx --all-features
```

## Feature sets

- Default feature members: None
- Available manifest features: `default`, `nvrtc`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `example` | `vyre_emit_ptx_release_surface` | `vyre-emit-ptx/examples/vyre_emit_ptx_release_surface.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-emit-ptx --example vyre_emit_ptx_release_surface` |
| `lib` | `vyre_emit_ptx` | `vyre-emit-ptx/src/lib.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-emit-ptx` |
| `test` | `adversarial_emit_program_matrix` | `vyre-emit-ptx/tests/adversarial_emit_program_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-emit-ptx --test adversarial_emit_program_matrix` |
| `test` | `cross_emitter_parity` | `vyre-emit-ptx/tests/cross_emitter_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-emit-ptx --test cross_emitter_parity` |
| `test` | `grid_sync_loop_refusal` | `vyre-emit-ptx/tests/grid_sync_loop_refusal.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-emit-ptx --test grid_sync_loop_refusal` |
| `test` | `nested_return_branch` | `vyre-emit-ptx/tests/nested_return_branch.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-emit-ptx --test nested_return_branch` |
| `test` | `nvrtc_compile_gate` | `vyre-emit-ptx/tests/nvrtc_compile_gate.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-emit-ptx --test nvrtc_compile_gate` |
| `test` | `regression_emit_fixes` | `vyre-emit-ptx/tests/regression_emit_fixes.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-emit-ptx --test regression_emit_fixes` |

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
