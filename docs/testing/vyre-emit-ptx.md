# Testing `vyre-emit-ptx`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-emit-ptx
```

Consume verified lowering products and emit the primary binary backend text artifact.

The crate lives at `vyre-emit-ptx`. The `primary-binary-emitter` owner maintains its
`emitter` testing contract.

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
| `test` | `adversarial_emit_program_matrix` | `vyre-emit-ptx/tests/adversarial_emit_program_matrix.rs` | None | `./cargo_full test -p vyre-emit-ptx --test adversarial_emit_program_matrix` |
| `test` | `barrier_scope_parity` | `vyre-emit-ptx/tests/barrier_scope_parity.rs` | None | `./cargo_full test -p vyre-emit-ptx --test barrier_scope_parity` |
| `test` | `cross_emitter_parity` | `vyre-emit-ptx/tests/cross_emitter_parity.rs` | None | `./cargo_full test -p vyre-emit-ptx --test cross_emitter_parity` |
| `test` | `divergent_trap_and_grid_barrier` | `vyre-emit-ptx/tests/divergent_trap_and_grid_barrier.rs` | None | `./cargo_full test -p vyre-emit-ptx --test divergent_trap_and_grid_barrier` |
| `test` | `emit_contracts` | `vyre-emit-ptx/tests/emit_contracts/mod.rs` | None | `./cargo_full test -p vyre-emit-ptx --test emit_contracts` |
| `test` | `emitted_artifact_byte_stability` | `vyre-emit-ptx/tests/emitted_artifact_byte_stability.rs` | None | `./cargo_full test -p vyre-emit-ptx --test emitted_artifact_byte_stability` |
| `test` | `grid_sync_loop_refusal` | `vyre-emit-ptx/tests/grid_sync_loop_refusal.rs` | None | `./cargo_full test -p vyre-emit-ptx --test grid_sync_loop_refusal` |
| `test` | `nested_return_branch` | `vyre-emit-ptx/tests/nested_return_branch.rs` | None | `./cargo_full test -p vyre-emit-ptx --test nested_return_branch` |
| `test` | `nvrtc_compile_gate` | `vyre-emit-ptx/tests/nvrtc_compile_gate/mod.rs` | None | `./cargo_full test -p vyre-emit-ptx --test nvrtc_compile_gate` |
| `test` | `pattern_analysis_contracts` | `vyre-emit-ptx/tests/pattern_analysis_contracts/mod.rs` | None | `./cargo_full test -p vyre-emit-ptx --test pattern_analysis_contracts` |
| `test` | `ping_pong_schedule_contracts` | `vyre-emit-ptx/tests/ping_pong_schedule_contracts.rs` | None | `./cargo_full test -p vyre-emit-ptx --test ping_pong_schedule_contracts` |
| `test` | `regression_emit_fixes` | `vyre-emit-ptx/tests/regression_emit_fixes.rs` | None | `./cargo_full test -p vyre-emit-ptx --test regression_emit_fixes` |
| `test` | `shared_branch_walk_equality` | `vyre-emit-ptx/tests/shared_branch_walk_equality.rs` | None | `./cargo_full test -p vyre-emit-ptx --test shared_branch_walk_equality` |
| `test` | `ulp_budget_is_not_an_admission_gate` | `vyre-emit-ptx/tests/ulp_budget_is_not_an_admission_gate.rs` | None | `./cargo_full test -p vyre-emit-ptx --test ulp_budget_is_not_an_admission_gate` |

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
