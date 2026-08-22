# cross-backend comparison

Produced by `./cargo_full run --bin xtask -- bench-crossback --write`. Every row is a
wall-clock reading a release benchmark suite recorded under
`release/evidence/benchmarks/`, with the commit, source-tree fingerprint and
device signature it was taken under. `ratio` is the case wall time over the
fastest backend measured for that case.

| case | backend | ms | ratio | commit | source tree | device | artifact |
|------|---------|----|-------|--------|-------------|--------|----------|
| `callgraph.reachability.step.262k` | cuda | 0.040 | 1.000 | 8d6478ebb2e8 | source-tree-v1:66806e66e236 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-13-callgraph-reachability.json` |
| `callgraph.reachability.step.262k` | wgpu | 0.088 | 2.227 | 8d6478ebb2e8 | source-tree-v1:66806e66e236 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-13-callgraph-reachability.json` |
| `compound.pipeline.fused_filter.1m` | cuda | 0.903 | 1.012 | 8d6478ebb2e8 | source-tree-v1:66806e66e236 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-14-compound-fused-filter.json` |
| `compound.pipeline.fused_filter.1m` | wgpu | 0.892 | 1.000 | 8d6478ebb2e8 | source-tree-v1:66806e66e236 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-14-compound-fused-filter.json` |
| `cuda.ptx.patterns.release.corpus` | cuda | 0.048 | 1.000 | 8d6478ebb2e8 | source-tree-v1:66806e66e236 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/cuda-ptx-patterns.json` |
| `foundation.optimizer.impact` | cuda | 1.453 | 1.002 | 8d6478ebb2e8 | source-tree-v1:66806e66e236 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-11-semantic-optimizer-impact.json` |
| `foundation.optimizer.impact` | wgpu | 1.450 | 1.000 | 8d6478ebb2e8 | source-tree-v1:66806e66e236 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-11-semantic-optimizer-impact.json` |
| `foundation.reduce.sum.crossover` | cuda | 0.820 | 1.000 | 8d6478ebb2e8 | source-tree-v1:66806e66e236 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-15-adaptive-routing.json` |
| `foundation.reduce.sum.crossover` | wgpu | 1.173 | 1.430 | 8d6478ebb2e8 | source-tree-v1:66806e66e236 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-15-adaptive-routing.json` |
| `metadata.condition.filesize_header.1m` | cuda | 0.021 | 1.000 | 8d6478ebb2e8 | source-tree-v1:66806e66e236 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-04-metadata-conditions.json` |
| `metadata.condition.filesize_header.1m` | wgpu | 0.069 | 3.312 | 8d6478ebb2e8 | source-tree-v1:66806e66e236 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-04-metadata-conditions.json` |
| `nn.linear_4bit_affine_grouped.1m` | cuda | 0.005 | 1.000 | 8d6478ebb2e8 | source-tree-v1:66806e66e236 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-16-quantized-linear.json` |
| `nn.linear_4bit_affine_grouped.1m` | wgpu | 0.005 | 1.138 | 8d6478ebb2e8 | source-tree-v1:66806e66e236 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-16-quantized-linear.json` |
| `release.alias_reaching_def.1m` | cuda | 0.032 | 1.000 | 8d6478ebb2e8 | source-tree-v1:66806e66e236 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-07-alias-reaching-def.json` |
| `release.alias_reaching_def.1m` | wgpu | 0.083 | 2.638 | 8d6478ebb2e8 | source-tree-v1:66806e66e236 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-07-alias-reaching-def.json` |
| `release.ast_motif_traversal.1m` | cuda | 0.032 | 1.000 | 8d6478ebb2e8 | source-tree-v1:66806e66e236 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-09-ast-motif-traversal.json` |
| `release.ast_motif_traversal.1m` | wgpu | 0.087 | 2.679 | 8d6478ebb2e8 | source-tree-v1:66806e66e236 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-09-ast-motif-traversal.json` |
| `release.condition_eval.1m` | cuda | 0.032 | 1.000 | 8d6478ebb2e8 | source-tree-v1:66806e66e236 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-01-condition-eval.json` |
| `release.condition_eval.1m` | wgpu | 0.086 | 2.725 | 8d6478ebb2e8 | source-tree-v1:66806e66e236 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-01-condition-eval.json` |
| `release.egraph_saturation.1m` | cuda | 0.032 | 1.000 | 8d6478ebb2e8 | source-tree-v1:66806e66e236 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-17-egraph-saturation.json` |
| `release.egraph_saturation.1m` | wgpu | 0.086 | 2.646 | 8d6478ebb2e8 | source-tree-v1:66806e66e236 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-17-egraph-saturation.json` |
| `release.entropy_window.1m` | cuda | 0.032 | 1.000 | 8d6478ebb2e8 | source-tree-v1:66806e66e236 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-05-entropy-window.json` |
| `release.entropy_window.1m` | wgpu | 0.088 | 2.775 | 8d6478ebb2e8 | source-tree-v1:66806e66e236 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-05-entropy-window.json` |
| `release.ifds_witness.1m` | cuda | 0.032 | 1.000 | 8d6478ebb2e8 | source-tree-v1:66806e66e236 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-08-ifds-witness.json` |
| `release.ifds_witness.1m` | wgpu | 0.085 | 2.678 | 8d6478ebb2e8 | source-tree-v1:66806e66e236 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-08-ifds-witness.json` |
| `release.megakernel_queue.1m` | cuda | 0.032 | 1.000 | 8d6478ebb2e8 | source-tree-v1:66806e66e236 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-10-megakernel-queued-batches.json` |
| `release.megakernel_queue.1m` | wgpu | 0.086 | 2.669 | 8d6478ebb2e8 | source-tree-v1:66806e66e236 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-10-megakernel-queued-batches.json` |
| `release.offset_count_aggregation.1m` | cuda | 0.032 | 1.000 | 8d6478ebb2e8 | source-tree-v1:66806e66e236 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-03-offset-count-aggregation.json` |
| `release.offset_count_aggregation.1m` | wgpu | 0.086 | 2.686 | 8d6478ebb2e8 | source-tree-v1:66806e66e236 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-03-offset-count-aggregation.json` |
| `release.quantified_condition_loops.1m` | cuda | 0.032 | 1.000 | 8d6478ebb2e8 | source-tree-v1:66806e66e236 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-06-quantified-condition-loops.json` |
| `release.quantified_condition_loops.1m` | wgpu | 0.087 | 2.697 | 8d6478ebb2e8 | source-tree-v1:66806e66e236 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-06-quantified-condition-loops.json` |
| `release.string_bitmap_scatter.1m` | cuda | 0.088 | 1.000 | 8d6478ebb2e8 | source-tree-v1:66806e66e236 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-02-string-bitmap-scatter.json` |
| `release.string_bitmap_scatter.1m` | wgpu | 0.097 | 1.097 | 8d6478ebb2e8 | source-tree-v1:66806e66e236 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-02-string-bitmap-scatter.json` |
| `sparse.compaction.count.1m` | cuda | 0.020 | 1.000 | 8d6478ebb2e8 | source-tree-v1:66806e66e236 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-12-sparse-output-compaction.json` |
| `sparse.compaction.count.1m` | wgpu | 0.077 | 3.955 | 8d6478ebb2e8 | source-tree-v1:66806e66e236 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-12-sparse-output-compaction.json` |

## declared without a measurement

Every backend a case contract declares carries a measurement.
