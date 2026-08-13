# Testing `vyre-self-substrate`

Run the default crate suite from the workspace root:

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate
```

Use Vyre primitives to implement scheduler, graph, coverage, and optimization support.

The crate lives at `vyre-self-substrate`. The `self-substrate` owner maintains its
`scheduler` testing contract.

## Commands

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate
```

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --all-features
```

## Feature sets

- Default feature members: `optimizer`
- Available manifest features: `all-solvers`, `analysis`, `cpu-parity`, `data`, `default`, `graph-solvers`, `logic`, `math-solvers`, `optimizer`, `scheduling`, `telemetry`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `example` | `vyre_self_substrate_release_surface` | `vyre-self-substrate/examples/vyre_self_substrate_release_surface.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --example vyre_self_substrate_release_surface` |
| `lib` | `vyre_self_substrate` | `vyre-self-substrate/src/lib.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate` |
| `test` | `bellman_shortest_path_via_reference_parity` | `vyre-self-substrate/tests/bellman_shortest_path_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test bellman_shortest_path_via_reference_parity` |
| `test` | `bellman_shortest_path_via_reference_parity` | `vyre-self-substrate/tests/bellman_shortest_path_via_reference_parity.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test bellman_shortest_path_via_reference_parity` |
| `test` | `bitset_dense_matvec_pipeline_generated` | `vyre-self-substrate/tests/bitset_dense_matvec_pipeline_generated.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test bitset_dense_matvec_pipeline_generated` |
| `test` | `bitset_dense_matvec_pipeline_generated` | `vyre-self-substrate/tests/bitset_dense_matvec_pipeline_generated.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test bitset_dense_matvec_pipeline_generated` |
| `test` | `bitset_mask_algebra_via_reference_parity` | `vyre-self-substrate/tests/bitset_mask_algebra_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test bitset_mask_algebra_via_reference_parity` |
| `test` | `bitset_mask_algebra_via_reference_parity` | `vyre-self-substrate/tests/bitset_mask_algebra_via_reference_parity.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test bitset_mask_algebra_via_reference_parity` |
| `test` | `bitset_summary_via_reference_parity` | `vyre-self-substrate/tests/bitset_summary_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test bitset_summary_via_reference_parity` |
| `test` | `bitset_summary_via_reference_parity` | `vyre-self-substrate/tests/bitset_summary_via_reference_parity.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test bitset_summary_via_reference_parity` |
| `test` | `categorical_laws_proptest` | `vyre-self-substrate/tests/categorical_laws_proptest.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test categorical_laws_proptest` |
| `test` | `categorical_laws_proptest` | `vyre-self-substrate/tests/categorical_laws_proptest.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test categorical_laws_proptest` |
| `test` | `consumer_boundary` | `vyre-self-substrate/tests/consumer_boundary.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test consumer_boundary` |
| `test` | `cost_model_predict_runtime_via_reference_parity` | `vyre-self-substrate/tests/cost_model_predict_runtime_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test cost_model_predict_runtime_via_reference_parity` |
| `test` | `cost_model_predict_runtime_via_reference_parity` | `vyre-self-substrate/tests/cost_model_predict_runtime_via_reference_parity.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test cost_model_predict_runtime_via_reference_parity` |
| `test` | `dce_dispatch_binding_contract` | `vyre-self-substrate/tests/dce_dispatch_binding_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test dce_dispatch_binding_contract` |
| `test` | `dce_program_back_edge_contract` | `vyre-self-substrate/tests/dce_program_back_edge_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test dce_program_back_edge_contract` |
| `test` | `do_calculus_surgery_via_reference_parity` | `vyre-self-substrate/tests/do_calculus_surgery_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test do_calculus_surgery_via_reference_parity` |
| `test` | `do_calculus_surgery_via_reference_parity` | `vyre-self-substrate/tests/do_calculus_surgery_via_reference_parity.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test do_calculus_surgery_via_reference_parity` |
| `test` | `feature_boundaries` | `vyre-self-substrate/tests/feature_boundaries.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test feature_boundaries` |
| `test` | `fmm_compress_pairwise_via_reference_parity` | `vyre-self-substrate/tests/fmm_compress_pairwise_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test fmm_compress_pairwise_via_reference_parity` |
| `test` | `fmm_compress_pairwise_via_reference_parity` | `vyre-self-substrate/tests/fmm_compress_pairwise_via_reference_parity.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test fmm_compress_pairwise_via_reference_parity` |
| `test` | `fmm_polyhedral_via_reference_parity` | `vyre-self-substrate/tests/fmm_polyhedral_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test fmm_polyhedral_via_reference_parity` |
| `test` | `fmm_polyhedral_via_reference_parity` | `vyre-self-substrate/tests/fmm_polyhedral_via_reference_parity.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test fmm_polyhedral_via_reference_parity` |
| `test` | `functor_apply_via_reference_parity` | `vyre-self-substrate/tests/functor_apply_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test functor_apply_via_reference_parity` |
| `test` | `functor_apply_via_reference_parity` | `vyre-self-substrate/tests/functor_apply_via_reference_parity.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test functor_apply_via_reference_parity` |
| `test` | `fusion_scores_via_reference_parity` | `vyre-self-substrate/tests/fusion_scores_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test fusion_scores_via_reference_parity` |
| `test` | `fusion_scores_via_reference_parity` | `vyre-self-substrate/tests/fusion_scores_via_reference_parity.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test fusion_scores_via_reference_parity` |
| `test` | `graph_single_source_contracts` | `vyre-self-substrate/tests/graph_single_source_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test graph_single_source_contracts` |
| `test` | `kfac_via_reference_parity` | `vyre-self-substrate/tests/kfac_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test kfac_via_reference_parity` |
| `test` | `kfac_via_reference_parity` | `vyre-self-substrate/tests/kfac_via_reference_parity.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test kfac_via_reference_parity` |
| `test` | `match_motif_via_reference_parity` | `vyre-self-substrate/tests/match_motif_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test match_motif_via_reference_parity` |
| `test` | `match_motif_via_reference_parity` | `vyre-self-substrate/tests/match_motif_via_reference_parity.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test match_motif_via_reference_parity` |
| `test` | `matching_diagnostic_via_reference_parity` | `vyre-self-substrate/tests/matching_diagnostic_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test matching_diagnostic_via_reference_parity` |
| `test` | `matching_diagnostic_via_reference_parity` | `vyre-self-substrate/tests/matching_diagnostic_via_reference_parity.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test matching_diagnostic_via_reference_parity` |
| `test` | `matroid_exact_subset_via_reference_parity` | `vyre-self-substrate/tests/matroid_exact_subset_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test matroid_exact_subset_via_reference_parity` |
| `test` | `matroid_exact_subset_via_reference_parity` | `vyre-self-substrate/tests/matroid_exact_subset_via_reference_parity.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test matroid_exact_subset_via_reference_parity` |
| `test` | `multigrid_matroid_via_reference_parity` | `vyre-self-substrate/tests/multigrid_matroid_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test multigrid_matroid_via_reference_parity` |
| `test` | `multigrid_matroid_via_reference_parity` | `vyre-self-substrate/tests/multigrid_matroid_via_reference_parity.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test multigrid_matroid_via_reference_parity` |
| `test` | `mz_project_via_reference_parity` | `vyre-self-substrate/tests/mz_project_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test mz_project_via_reference_parity` |
| `test` | `mz_project_via_reference_parity` | `vyre-self-substrate/tests/mz_project_via_reference_parity.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test mz_project_via_reference_parity` |
| `test` | `natural_config_gradient_via_reference_parity` | `vyre-self-substrate/tests/natural_config_gradient_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test natural_config_gradient_via_reference_parity` |
| `test` | `natural_config_gradient_via_reference_parity` | `vyre-self-substrate/tests/natural_config_gradient_via_reference_parity.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test natural_config_gradient_via_reference_parity` |
| `test` | `natural_gradient_via_reference_parity` | `vyre-self-substrate/tests/natural_gradient_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test natural_gradient_via_reference_parity` |
| `test` | `natural_gradient_via_reference_parity` | `vyre-self-substrate/tests/natural_gradient_via_reference_parity.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test natural_gradient_via_reference_parity` |
| `test` | `optimizer_bfs_and_softmax_parity` | `vyre-self-substrate/tests/optimizer_bfs_and_softmax_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test optimizer_bfs_and_softmax_parity` |
| `test` | `optimizer_bfs_and_softmax_parity` | `vyre-self-substrate/tests/optimizer_bfs_and_softmax_parity.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test optimizer_bfs_and_softmax_parity` |
| `test` | `planar_rewrite_via_reference_parity` | `vyre-self-substrate/tests/planar_rewrite_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test planar_rewrite_via_reference_parity` |
| `test` | `planar_rewrite_via_reference_parity` | `vyre-self-substrate/tests/planar_rewrite_via_reference_parity.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test planar_rewrite_via_reference_parity` |
| `test` | `platform_doc_consumer_boundary` | `vyre-self-substrate/tests/platform_doc_consumer_boundary.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test platform_doc_consumer_boundary` |
| `test` | `predict_impact_via_reference_parity` | `vyre-self-substrate/tests/predict_impact_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test predict_impact_via_reference_parity` |
| `test` | `predict_impact_via_reference_parity` | `vyre-self-substrate/tests/predict_impact_via_reference_parity.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test predict_impact_via_reference_parity` |
| `test` | `primitive_vs_consumer` | `vyre-self-substrate/tests/primitive_vs_consumer.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test primitive_vs_consumer` |
| `test` | `primitive_vs_consumer` | `vyre-self-substrate/tests/primitive_vs_consumer.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test primitive_vs_consumer` |
| `test` | `provenance_closure` | `vyre-self-substrate/tests/provenance_closure.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test provenance_closure` |
| `test` | `provenance_closure` | `vyre-self-substrate/tests/provenance_closure.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test provenance_closure` |
| `test` | `quantized_via_reference_parity` | `vyre-self-substrate/tests/quantized_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test quantized_via_reference_parity` |
| `test` | `quantized_via_reference_parity` | `vyre-self-substrate/tests/quantized_via_reference_parity.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test quantized_via_reference_parity` |
| `test` | `reconstruct_path_via_reference_parity` | `vyre-self-substrate/tests/reconstruct_path_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test reconstruct_path_via_reference_parity` |
| `test` | `reconstruct_path_via_reference_parity` | `vyre-self-substrate/tests/reconstruct_path_via_reference_parity.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test reconstruct_path_via_reference_parity` |
| `test` | `reduction_metrics_via_reference_parity` | `vyre-self-substrate/tests/reduction_metrics_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test reduction_metrics_via_reference_parity` |
| `test` | `reduction_metrics_via_reference_parity` | `vyre-self-substrate/tests/reduction_metrics_via_reference_parity.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test reduction_metrics_via_reference_parity` |
| `test` | `release_evidence_path_contract` | `vyre-self-substrate/tests/release_evidence_path_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test release_evidence_path_contract` |
| `test` | `scallop_provenance_via_reference_parity` | `vyre-self-substrate/tests/scallop_provenance_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test scallop_provenance_via_reference_parity` |
| `test` | `scallop_provenance_via_reference_parity` | `vyre-self-substrate/tests/scallop_provenance_via_reference_parity.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test scallop_provenance_via_reference_parity` |
| `test` | `self_consumer_conform` | `vyre-self-substrate/tests/self_consumer_conform.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test self_consumer_conform` |
| `test` | `self_consumer_conform` | `vyre-self-substrate/tests/self_consumer_conform.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test self_consumer_conform` |
| `test` | `semiring_gemm_via_reference_parity` | `vyre-self-substrate/tests/semiring_gemm_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test semiring_gemm_via_reference_parity` |
| `test` | `semiring_gemm_via_reference_parity` | `vyre-self-substrate/tests/semiring_gemm_via_reference_parity.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test semiring_gemm_via_reference_parity` |
| `test` | `shape_spectrum_via_reference_parity` | `vyre-self-substrate/tests/shape_spectrum_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test shape_spectrum_via_reference_parity` |
| `test` | `shape_spectrum_via_reference_parity` | `vyre-self-substrate/tests/shape_spectrum_via_reference_parity.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test shape_spectrum_via_reference_parity` |
| `test` | `sheaf_heterophilic_via_reference_parity` | `vyre-self-substrate/tests/sheaf_heterophilic_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test sheaf_heterophilic_via_reference_parity` |
| `test` | `sheaf_heterophilic_via_reference_parity` | `vyre-self-substrate/tests/sheaf_heterophilic_via_reference_parity.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test sheaf_heterophilic_via_reference_parity` |
| `test` | `sheaf_spectrum_via_reference_parity` | `vyre-self-substrate/tests/sheaf_spectrum_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test sheaf_spectrum_via_reference_parity` |
| `test` | `sheaf_spectrum_via_reference_parity` | `vyre-self-substrate/tests/sheaf_spectrum_via_reference_parity.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test sheaf_spectrum_via_reference_parity` |
| `test` | `sinkhorn_via_reference_parity` | `vyre-self-substrate/tests/sinkhorn_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test sinkhorn_via_reference_parity` |
| `test` | `sinkhorn_via_reference_parity` | `vyre-self-substrate/tests/sinkhorn_via_reference_parity.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test sinkhorn_via_reference_parity` |
| `test` | `smooth_latency_trace_via_reference_parity` | `vyre-self-substrate/tests/smooth_latency_trace_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test smooth_latency_trace_via_reference_parity` |
| `test` | `smooth_latency_trace_via_reference_parity` | `vyre-self-substrate/tests/smooth_latency_trace_via_reference_parity.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test smooth_latency_trace_via_reference_parity` |
| `test` | `smooth_matroid_flow_via_reference_parity` | `vyre-self-substrate/tests/smooth_matroid_flow_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test smooth_matroid_flow_via_reference_parity` |
| `test` | `smooth_matroid_flow_via_reference_parity` | `vyre-self-substrate/tests/smooth_matroid_flow_via_reference_parity.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test smooth_matroid_flow_via_reference_parity` |
| `test` | `softmax_pick_config_via_reference_parity` | `vyre-self-substrate/tests/softmax_pick_config_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test softmax_pick_config_via_reference_parity` |
| `test` | `softmax_pick_config_via_reference_parity` | `vyre-self-substrate/tests/softmax_pick_config_via_reference_parity.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test softmax_pick_config_via_reference_parity` |
| `test` | `string_diagram_via_reference_parity` | `vyre-self-substrate/tests/string_diagram_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test string_diagram_via_reference_parity` |
| `test` | `string_diagram_via_reference_parity` | `vyre-self-substrate/tests/string_diagram_via_reference_parity.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test string_diagram_via_reference_parity` |
| `test` | `submodular_retention_via_reference_parity` | `vyre-self-substrate/tests/submodular_retention_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test submodular_retention_via_reference_parity` |
| `test` | `submodular_retention_via_reference_parity` | `vyre-self-substrate/tests/submodular_retention_via_reference_parity.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test submodular_retention_via_reference_parity` |
| `test` | `sweep_graph_cpu_oracle_matrix` | `vyre-self-substrate/tests/sweep_graph_cpu_oracle_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test sweep_graph_cpu_oracle_matrix` |
| `test` | `sweep_graph_cpu_oracle_matrix` | `vyre-self-substrate/tests/sweep_graph_cpu_oracle_matrix.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test sweep_graph_cpu_oracle_matrix` |
| `test` | `tensor_train_chain_fusion_via_reference_parity` | `vyre-self-substrate/tests/tensor_train_chain_fusion_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test tensor_train_chain_fusion_via_reference_parity` |
| `test` | `tensor_train_chain_fusion_via_reference_parity` | `vyre-self-substrate/tests/tensor_train_chain_fusion_via_reference_parity.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test tensor_train_chain_fusion_via_reference_parity` |
| `test` | `tensor_train_compress_via_reference_parity` | `vyre-self-substrate/tests/tensor_train_compress_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test tensor_train_compress_via_reference_parity` |
| `test` | `tensor_train_compress_via_reference_parity` | `vyre-self-substrate/tests/tensor_train_compress_via_reference_parity.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test tensor_train_compress_via_reference_parity` |
| `test` | `transport_residual_via_reference_parity` | `vyre-self-substrate/tests/transport_residual_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test transport_residual_via_reference_parity` |
| `test` | `transport_residual_via_reference_parity` | `vyre-self-substrate/tests/transport_residual_via_reference_parity.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test transport_residual_via_reference_parity` |
| `test` | `union_find_alias_via_reference_parity` | `vyre-self-substrate/tests/union_find_alias_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test union_find_alias_via_reference_parity` |
| `test` | `union_find_alias_via_reference_parity` | `vyre-self-substrate/tests/union_find_alias_via_reference_parity.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test union_find_alias_via_reference_parity` |
| `test` | `vietoris_rips_via_reference_parity` | `vyre-self-substrate/tests/vietoris_rips_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test vietoris_rips_via_reference_parity` |
| `test` | `vietoris_rips_via_reference_parity` | `vyre-self-substrate/tests/vietoris_rips_via_reference_parity.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test vietoris_rips_via_reference_parity` |
| `test` | `vsa_fingerprint_via_reference_parity` | `vyre-self-substrate/tests/vsa_fingerprint_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test vsa_fingerprint_via_reference_parity` |
| `test` | `vsa_fingerprint_via_reference_parity` | `vyre-self-substrate/tests/vsa_fingerprint_via_reference_parity.rs` | `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-self-substrate --test vsa_fingerprint_via_reference_parity` |

## Test classes

- Scheduler and graph semantics
- Self-consumer composition contracts
- Determinism and boundary tests

## Hardware requirements

No accelerator is required for the default suite.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
