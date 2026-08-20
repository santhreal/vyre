# cross-backend comparison

Produced by `./cargo_full run --bin xtask -- bench-crossback --write`. Every row is a
wall-clock reading a release benchmark suite recorded under
`release/evidence/benchmarks/`, with the commit, source-tree fingerprint and
device signature it was taken under. `ratio` is the case wall time over the
fastest backend measured for that case.

| case | backend | ms | ratio | commit | source tree | device | artifact |
|------|---------|----|-------|--------|-------------|--------|----------|
| `callgraph.reachability.step.262k` | cuda | 0.035 | 1.000 | 1eead476cab0 | source-tree-v1:a525e214856d | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-13-callgraph-reachability.json` |
| `callgraph.reachability.step.262k` | wgpu | 0.087 | 2.465 | 1eead476cab0 | source-tree-v1:a525e214856d | device-profile-v1:79429d7cc184 | `release/evidence/benchmarks/wgpu-workload-13-callgraph-reachability.json` |
| `compound.pipeline.fused_filter.1m` | cuda | 0.888 | 1.000 | 1eead476cab0 | source-tree-v1:a525e214856d | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-14-compound-fused-filter.json` |
| `compound.pipeline.fused_filter.1m` | wgpu | 0.920 | 1.036 | 1eead476cab0 | source-tree-v1:a525e214856d | device-profile-v1:79429d7cc184 | `release/evidence/benchmarks/wgpu-workload-14-compound-fused-filter.json` |
| `cuda.ptx.patterns.release.corpus` | cuda | 0.063 | 1.000 | a8cdc3985324 | source-tree-v1:35eddeba1522 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/cuda-ptx-patterns.json` |
| `foundation.optimizer.impact` | cuda | 1.464 | 1.000 | 1eead476cab0 | source-tree-v1:a525e214856d | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-11-semantic-optimizer-impact.json` |
| `foundation.optimizer.impact` | wgpu | 1.465 | 1.001 | 1eead476cab0 | source-tree-v1:a525e214856d | device-profile-v1:79429d7cc184 | `release/evidence/benchmarks/wgpu-workload-11-semantic-optimizer-impact.json` |
| `foundation.reduce.sum.crossover` | cuda | 1.856 | 1.278 | 1eead476cab0 | source-tree-v1:a525e214856d | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-15-adaptive-routing.json` |
| `foundation.reduce.sum.crossover` | wgpu | 1.453 | 1.000 | 1eead476cab0 | source-tree-v1:a525e214856d | device-profile-v1:79429d7cc184 | `release/evidence/benchmarks/wgpu-workload-15-adaptive-routing.json` |
| `metadata.condition.filesize_header.1m` | cuda | 0.020 | 1.000 | 1eead476cab0 | source-tree-v1:a525e214856d | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-04-metadata-conditions.json` |
| `metadata.condition.filesize_header.1m` | wgpu | 0.070 | 3.400 | 1eead476cab0 | source-tree-v1:a525e214856d | device-profile-v1:79429d7cc184 | `release/evidence/benchmarks/wgpu-workload-04-metadata-conditions.json` |
| `nn.linear_4bit_affine_grouped.1m` | cuda | 0.005 | 1.000 | 1eead476cab0 | source-tree-v1:a525e214856d | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-16-quantized-linear.json` |
| `nn.linear_4bit_affine_grouped.1m` | wgpu | 0.006 | 1.274 | 1eead476cab0 | source-tree-v1:a525e214856d | device-profile-v1:79429d7cc184 | `release/evidence/benchmarks/wgpu-workload-16-quantized-linear.json` |
| `release.alias_reaching_def.1m` | cuda | 0.021 | 1.000 | 1eead476cab0 | source-tree-v1:a525e214856d | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-07-alias-reaching-def.json` |
| `release.alias_reaching_def.1m` | wgpu | 0.211 | 9.874 | 1eead476cab0 | source-tree-v1:a525e214856d | device-profile-v1:79429d7cc184 | `release/evidence/benchmarks/wgpu-workload-07-alias-reaching-def.json` |
| `release.ast_motif_traversal.1m` | cuda | 0.021 | 1.000 | 1eead476cab0 | source-tree-v1:a525e214856d | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-09-ast-motif-traversal.json` |
| `release.ast_motif_traversal.1m` | wgpu | 0.279 | 13.053 | 1eead476cab0 | source-tree-v1:a525e214856d | device-profile-v1:79429d7cc184 | `release/evidence/benchmarks/wgpu-workload-09-ast-motif-traversal.json` |
| `release.condition_eval.1m` | cuda | 0.021 | 1.000 | 1eead476cab0 | source-tree-v1:a525e214856d | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-01-condition-eval.json` |
| `release.condition_eval.1m` | wgpu | 0.271 | 12.804 | 1eead476cab0 | source-tree-v1:a525e214856d | device-profile-v1:79429d7cc184 | `release/evidence/benchmarks/wgpu-workload-01-condition-eval.json` |
| `release.egraph_saturation.1m` | cuda | 0.022 | 1.000 | 1eead476cab0 | source-tree-v1:a525e214856d | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-17-egraph-saturation.json` |
| `release.egraph_saturation.1m` | wgpu | 0.247 | 11.346 | 1eead476cab0 | source-tree-v1:a525e214856d | device-profile-v1:79429d7cc184 | `release/evidence/benchmarks/wgpu-workload-17-egraph-saturation.json` |
| `release.entropy_window.1m` | cuda | 0.021 | 1.000 | 1eead476cab0 | source-tree-v1:a525e214856d | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-05-entropy-window.json` |
| `release.entropy_window.1m` | wgpu | 0.272 | 13.243 | 1eead476cab0 | source-tree-v1:a525e214856d | device-profile-v1:79429d7cc184 | `release/evidence/benchmarks/wgpu-workload-05-entropy-window.json` |
| `release.ifds_witness.1m` | cuda | 0.021 | 1.000 | 1eead476cab0 | source-tree-v1:a525e214856d | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-08-ifds-witness.json` |
| `release.ifds_witness.1m` | wgpu | 0.271 | 12.756 | 1eead476cab0 | source-tree-v1:a525e214856d | device-profile-v1:79429d7cc184 | `release/evidence/benchmarks/wgpu-workload-08-ifds-witness.json` |
| `release.megakernel_queue.1m` | cuda | 0.021 | 1.000 | 1eead476cab0 | source-tree-v1:a525e214856d | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-10-megakernel-queued-batches.json` |
| `release.megakernel_queue.1m` | wgpu | 0.180 | 8.457 | 1eead476cab0 | source-tree-v1:a525e214856d | device-profile-v1:79429d7cc184 | `release/evidence/benchmarks/wgpu-workload-10-megakernel-queued-batches.json` |
| `release.offset_count_aggregation.1m` | cuda | 0.021 | 1.000 | 1eead476cab0 | source-tree-v1:a525e214856d | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-03-offset-count-aggregation.json` |
| `release.offset_count_aggregation.1m` | wgpu | 0.126 | 5.996 | 1eead476cab0 | source-tree-v1:a525e214856d | device-profile-v1:79429d7cc184 | `release/evidence/benchmarks/wgpu-workload-03-offset-count-aggregation.json` |
| `release.quantified_condition_loops.1m` | cuda | 0.021 | 1.000 | 1eead476cab0 | source-tree-v1:a525e214856d | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-06-quantified-condition-loops.json` |
| `release.quantified_condition_loops.1m` | wgpu | 0.137 | 6.453 | 1eead476cab0 | source-tree-v1:a525e214856d | device-profile-v1:79429d7cc184 | `release/evidence/benchmarks/wgpu-workload-06-quantified-condition-loops.json` |
| `release.string_bitmap_scatter.1m` | cuda | 0.002 | 1.000 | 1eead476cab0 | source-tree-v1:a525e214856d | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-02-string-bitmap-scatter.json` |
| `release.string_bitmap_scatter.1m` | wgpu | 0.006 | 2.696 | 1eead476cab0 | source-tree-v1:a525e214856d | device-profile-v1:79429d7cc184 | `release/evidence/benchmarks/wgpu-workload-02-string-bitmap-scatter.json` |
| `sparse.compaction.count.1m` | cuda | 0.020 | 1.000 | 1eead476cab0 | source-tree-v1:a525e214856d | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-12-sparse-output-compaction.json` |
| `sparse.compaction.count.1m` | wgpu | 0.076 | 3.847 | 1eead476cab0 | source-tree-v1:a525e214856d | device-profile-v1:79429d7cc184 | `release/evidence/benchmarks/wgpu-workload-12-sparse-output-compaction.json` |

## declared without a measurement

Every backend a case contract declares carries a measurement.
