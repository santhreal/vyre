# cross-backend comparison

Produced by `./cargo_full run --bin xtask -- bench-crossback --write`. Every row is a
wall-clock reading a release benchmark suite recorded under
`release/evidence/benchmarks/`, with the commit, source-tree fingerprint and
device signature it was taken under. `ratio` is the case wall time over the
fastest backend measured for that case.

| case | backend | ms | ratio | commit | source tree | device | artifact |
|------|---------|----|-------|--------|-------------|--------|----------|
| `callgraph.reachability.step.262k` | cuda | 0.034 | 1.000 | a8cdc3985324 | source-tree-v1:35eddeba1522 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-13-callgraph-reachability.json` |
| `callgraph.reachability.step.262k` | wgpu | 0.135 | 3.980 | a8cdc3985324 | source-tree-v1:35eddeba1522 | device-profile-v1:79429d7cc184 | `release/evidence/benchmarks/wgpu-workload-13-callgraph-reachability.json` |
| `compound.pipeline.fused_filter.1m` | cuda | 1.083 | 1.000 | a8cdc3985324 | source-tree-v1:35eddeba1522 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-14-compound-fused-filter.json` |
| `compound.pipeline.fused_filter.1m` | wgpu | 1.111 | 1.027 | a8cdc3985324 | source-tree-v1:35eddeba1522 | device-profile-v1:79429d7cc184 | `release/evidence/benchmarks/wgpu-workload-14-compound-fused-filter.json` |
| `cuda.ptx.patterns.release.corpus` | cuda | 0.063 | 1.000 | a8cdc3985324 | source-tree-v1:35eddeba1522 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/cuda-ptx-patterns.json` |
| `foundation.optimizer.impact` | cuda | 1.838 | 1.000 | a8cdc3985324 | source-tree-v1:35eddeba1522 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-11-semantic-optimizer-impact.json` |
| `foundation.optimizer.impact` | wgpu | 2.169 | 1.180 | a8cdc3985324 | source-tree-v1:35eddeba1522 | device-profile-v1:79429d7cc184 | `release/evidence/benchmarks/wgpu-workload-11-semantic-optimizer-impact.json` |
| `foundation.reduce.sum.crossover` | cuda | 2.412 | 1.515 | a8cdc3985324 | source-tree-v1:35eddeba1522 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-15-adaptive-routing.json` |
| `foundation.reduce.sum.crossover` | wgpu | 1.592 | 1.000 | a8cdc3985324 | source-tree-v1:35eddeba1522 | device-profile-v1:79429d7cc184 | `release/evidence/benchmarks/wgpu-workload-15-adaptive-routing.json` |
| `metadata.condition.filesize_header.1m` | cuda | 0.027 | 1.000 | a8cdc3985324 | source-tree-v1:35eddeba1522 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-04-metadata-conditions.json` |
| `metadata.condition.filesize_header.1m` | wgpu | 0.115 | 4.276 | a8cdc3985324 | source-tree-v1:35eddeba1522 | device-profile-v1:79429d7cc184 | `release/evidence/benchmarks/wgpu-workload-04-metadata-conditions.json` |
| `nn.linear_4bit_affine_grouped.1m` | cuda | 0.006 | 1.000 | a8cdc3985324 | source-tree-v1:35eddeba1522 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-16-quantized-linear.json` |
| `nn.linear_4bit_affine_grouped.1m` | wgpu | 0.006 | 1.165 | a8cdc3985324 | source-tree-v1:35eddeba1522 | device-profile-v1:79429d7cc184 | `release/evidence/benchmarks/wgpu-workload-16-quantized-linear.json` |
| `release.alias_reaching_def.1m` | cuda | 0.027 | 1.000 | a8cdc3985324 | source-tree-v1:35eddeba1522 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-07-alias-reaching-def.json` |
| `release.alias_reaching_def.1m` | wgpu | 0.179 | 6.614 | a8cdc3985324 | source-tree-v1:35eddeba1522 | device-profile-v1:79429d7cc184 | `release/evidence/benchmarks/wgpu-workload-07-alias-reaching-def.json` |
| `release.ast_motif_traversal.1m` | cuda | 0.030 | 1.000 | a8cdc3985324 | source-tree-v1:35eddeba1522 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-09-ast-motif-traversal.json` |
| `release.ast_motif_traversal.1m` | wgpu | 0.202 | 6.745 | a8cdc3985324 | source-tree-v1:35eddeba1522 | device-profile-v1:79429d7cc184 | `release/evidence/benchmarks/wgpu-workload-09-ast-motif-traversal.json` |
| `release.condition_eval.1m` | cuda | 0.030 | 1.000 | a8cdc3985324 | source-tree-v1:35eddeba1522 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-01-condition-eval.json` |
| `release.condition_eval.1m` | wgpu | 0.182 | 6.150 | a8cdc3985324 | source-tree-v1:35eddeba1522 | device-profile-v1:79429d7cc184 | `release/evidence/benchmarks/wgpu-workload-01-condition-eval.json` |
| `release.egraph_saturation.1m` | cuda | 0.030 | 1.000 | a8cdc3985324 | source-tree-v1:35eddeba1522 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-17-egraph-saturation.json` |
| `release.egraph_saturation.1m` | wgpu | 0.187 | 6.335 | a8cdc3985324 | source-tree-v1:35eddeba1522 | device-profile-v1:79429d7cc184 | `release/evidence/benchmarks/wgpu-workload-17-egraph-saturation.json` |
| `release.entropy_window.1m` | cuda | 0.027 | 1.000 | a8cdc3985324 | source-tree-v1:35eddeba1522 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-05-entropy-window.json` |
| `release.entropy_window.1m` | wgpu | 0.216 | 7.875 | a8cdc3985324 | source-tree-v1:35eddeba1522 | device-profile-v1:79429d7cc184 | `release/evidence/benchmarks/wgpu-workload-05-entropy-window.json` |
| `release.ifds_witness.1m` | cuda | 0.030 | 1.000 | a8cdc3985324 | source-tree-v1:35eddeba1522 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-08-ifds-witness.json` |
| `release.ifds_witness.1m` | wgpu | 0.171 | 5.790 | a8cdc3985324 | source-tree-v1:35eddeba1522 | device-profile-v1:79429d7cc184 | `release/evidence/benchmarks/wgpu-workload-08-ifds-witness.json` |
| `release.megakernel_queue.1m` | cuda | 0.030 | 1.000 | a8cdc3985324 | source-tree-v1:35eddeba1522 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-10-megakernel-queued-batches.json` |
| `release.megakernel_queue.1m` | wgpu | 0.119 | 4.025 | a8cdc3985324 | source-tree-v1:35eddeba1522 | device-profile-v1:79429d7cc184 | `release/evidence/benchmarks/wgpu-workload-10-megakernel-queued-batches.json` |
| `release.offset_count_aggregation.1m` | cuda | 0.030 | 1.000 | a8cdc3985324 | source-tree-v1:35eddeba1522 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-03-offset-count-aggregation.json` |
| `release.offset_count_aggregation.1m` | wgpu | 0.130 | 4.340 | a8cdc3985324 | source-tree-v1:35eddeba1522 | device-profile-v1:79429d7cc184 | `release/evidence/benchmarks/wgpu-workload-03-offset-count-aggregation.json` |
| `release.quantified_condition_loops.1m` | cuda | 0.027 | 1.000 | a8cdc3985324 | source-tree-v1:35eddeba1522 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-06-quantified-condition-loops.json` |
| `release.quantified_condition_loops.1m` | wgpu | 0.194 | 7.191 | a8cdc3985324 | source-tree-v1:35eddeba1522 | device-profile-v1:79429d7cc184 | `release/evidence/benchmarks/wgpu-workload-06-quantified-condition-loops.json` |
| `release.string_bitmap_scatter.1m` | cuda | 0.003 | 1.000 | a8cdc3985324 | source-tree-v1:35eddeba1522 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-02-string-bitmap-scatter.json` |
| `release.string_bitmap_scatter.1m` | wgpu | 0.008 | 2.576 | a8cdc3985324 | source-tree-v1:35eddeba1522 | device-profile-v1:79429d7cc184 | `release/evidence/benchmarks/wgpu-workload-02-string-bitmap-scatter.json` |
| `sparse.compaction.count.1m` | cuda | 0.027 | 1.000 | a8cdc3985324 | source-tree-v1:35eddeba1522 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-12-sparse-output-compaction.json` |
| `sparse.compaction.count.1m` | wgpu | 0.135 | 4.994 | a8cdc3985324 | source-tree-v1:35eddeba1522 | device-profile-v1:79429d7cc184 | `release/evidence/benchmarks/wgpu-workload-12-sparse-output-compaction.json` |

## declared without a measurement

Every backend a case contract declares carries a measurement.
