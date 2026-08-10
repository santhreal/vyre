# Testing `vyre-lower`

Run the default crate suite from the workspace root:

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-lower
```

Consume verified semantic programs and own the single backend-neutral lowering boundary and pre-emission transforms.

The crate lives at `vyre-lower`. The `lowering` owner maintains its
`lowering` testing contract.

## Commands

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-lower
```

## Feature sets

This crate declares no Cargo features.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `example` | `optimize` | `vyre-lower/examples/optimize.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-lower --example optimize` |
| `lib` | `vyre_lower` | `vyre-lower/src/lib.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-lower` |
| `test` | `adversarial_shared_mem_nonuniform_cf` | `vyre-lower/tests/adversarial_shared_mem_nonuniform_cf.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-lower --test adversarial_shared_mem_nonuniform_cf` |
| `test` | `analysis_fixture_corpuses` | `vyre-lower/tests/analysis_fixture_corpuses.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-lower --test analysis_fixture_corpuses` |
| `test` | `branch_collapse_backend_differential` | `vyre-lower/tests/branch_collapse_backend_differential.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-lower --test branch_collapse_backend_differential` |
| `test` | `branch_collapse_nested_assign_miscompile` | `vyre-lower/tests/branch_collapse_nested_assign_miscompile.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-lower --test branch_collapse_nested_assign_miscompile` |
| `test` | `dataflow_analysis_loop_rewrites` | `vyre-lower/tests/dataflow_analysis_loop_rewrites.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-lower --test dataflow_analysis_loop_rewrites` |
| `test` | `dataflow_loop_rewrites` | `vyre-lower/tests/dataflow_loop_rewrites.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-lower --test dataflow_loop_rewrites` |
| `test` | `dataflow_loop_support` | `vyre-lower/tests/dataflow_loop_support.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-lower --test dataflow_loop_support` |
| `test` | `dataflow_rewrite_api_contracts` | `vyre-lower/tests/dataflow_rewrite_api_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-lower --test dataflow_rewrite_api_contracts` |
| `test` | `dead_store_dataflow` | `vyre-lower/tests/dead_store_dataflow.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-lower --test dead_store_dataflow` |
| `test` | `dead_store_dataflow_analysis` | `vyre-lower/tests/dead_store_dataflow_analysis.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-lower --test dead_store_dataflow_analysis` |
| `test` | `egraph_pipeline_integration` | `vyre-lower/tests/egraph_pipeline_integration.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-lower --test egraph_pipeline_integration` |
| `test` | `kitchen_sink_snapshot` | `vyre-lower/tests/kitchen_sink_snapshot.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-lower --test kitchen_sink_snapshot` |
| `test` | `no_duplicate_result_ids_after_rewrites` | `vyre-lower/tests/no_duplicate_result_ids_after_rewrites.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-lower --test no_duplicate_result_ids_after_rewrites` |
| `test` | `optimization_corpus_contracts` | `vyre-lower/tests/optimization_corpus_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-lower --test optimization_corpus_contracts` |
| `test` | `rewrite_layer_contract` | `vyre-lower/tests/rewrite_layer_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-lower --test rewrite_layer_contract` |
| `test` | `rewrite_soundness_fuzz` | `vyre-lower/tests/rewrite_soundness_fuzz.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-lower --test rewrite_soundness_fuzz` |
| `test` | `target_capabilities` | `vyre-lower/tests/target_capabilities.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-lower --test target_capabilities` |
| `test` | `verify_result_id_uniqueness` | `vyre-lower/tests/verify_result_id_uniqueness.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-lower --test verify_result_id_uniqueness` |

## Test classes

- Backend-neutral lowering transforms
- Pre-emission invariants
- Invalid IR and target rejection

## Hardware requirements

No accelerator is required for the default suite.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
