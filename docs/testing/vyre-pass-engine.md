# Testing `vyre-pass-engine`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-pass-engine
```

Execute the optimizer's own passes as Vyre Programs, dispatched through the ProgramDispatcher seam.

The crate lives at `vyre-pass-engine`. The `pass-engine` owner maintains its
`pass-engine` testing contract.

## Commands

```console
./cargo_full test -p vyre-pass-engine
```

```console
./cargo_full test -p vyre-pass-engine --all-features
```

## Feature sets

- Default feature members: `optimizer`
- Available manifest features: `all-solvers`, `cpu-parity`, `default`, `optimizer`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `example` | `vyre_pass_engine_release_surface` | `vyre-pass-engine/examples/vyre_pass_engine_release_surface.rs` | None | `./cargo_full test -p vyre-pass-engine --example vyre_pass_engine_release_surface` |
| `lib` | `vyre_pass_engine` | `vyre-pass-engine/src/lib.rs` | None | `./cargo_full test -p vyre-pass-engine` |
| `test` | `consumer_boundary` | `vyre-pass-engine/tests/consumer_boundary.rs` | None | `./cargo_full test -p vyre-pass-engine --test consumer_boundary` |
| `test` | `cross_scope_cse_still_fires` | `vyre-pass-engine/tests/cross_scope_cse_still_fires.rs` | None | `./cargo_full test -p vyre-pass-engine --test cross_scope_cse_still_fires` |
| `test` | `cross_scope_cse_still_fires` | `vyre-pass-engine/tests/cross_scope_cse_still_fires.rs` | `optimizer` | `./cargo_full test -p vyre-pass-engine --test cross_scope_cse_still_fires` |
| `test` | `dce_dispatch_binding_contract` | `vyre-pass-engine/tests/dce_dispatch_binding_contract.rs` | None | `./cargo_full test -p vyre-pass-engine --test dce_dispatch_binding_contract` |
| `test` | `dce_dispatch_binding_contract` | `vyre-pass-engine/tests/dce_dispatch_binding_contract.rs` | `optimizer` | `./cargo_full test -p vyre-pass-engine --test dce_dispatch_binding_contract` |
| `test` | `dce_program_back_edge_contract` | `vyre-pass-engine/tests/dce_program_back_edge_contract.rs` | None | `./cargo_full test -p vyre-pass-engine --test dce_program_back_edge_contract` |
| `test` | `dce_program_back_edge_contract` | `vyre-pass-engine/tests/dce_program_back_edge_contract.rs` | `optimizer` | `./cargo_full test -p vyre-pass-engine --test dce_program_back_edge_contract` |
| `test` | `encoded_rewrite_walk_contract` | `vyre-pass-engine/tests/encoded_rewrite_walk_contract.rs` | None | `./cargo_full test -p vyre-pass-engine --test encoded_rewrite_walk_contract` |
| `test` | `feature_boundaries` | `vyre-pass-engine/tests/feature_boundaries.rs` | None | `./cargo_full test -p vyre-pass-engine --test feature_boundaries` |
| `test` | `optimizer_bfs_and_softmax_parity` | `vyre-pass-engine/tests/optimizer_bfs_and_softmax_parity.rs` | None | `./cargo_full test -p vyre-pass-engine --test optimizer_bfs_and_softmax_parity` |
| `test` | `optimizer_bfs_and_softmax_parity` | `vyre-pass-engine/tests/optimizer_bfs_and_softmax_parity.rs` | `cpu-parity` | `./cargo_full test -p vyre-pass-engine --test optimizer_bfs_and_softmax_parity` |

## Test classes

- Encoded-pass Program semantics
- Reference-parity of every dispatched pass
- Determinism and boundary tests

## Hardware requirements

No accelerator is required for the default suite.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
