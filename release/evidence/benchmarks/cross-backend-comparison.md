# cross-backend comparison

Produced by `./cargo_full run --bin xtask -- bench-crossback --write`. Every row is a
wall-clock reading a release benchmark suite recorded under
`release/evidence/benchmarks/`, with the commit, source-tree fingerprint and
device signature it was taken under. `ratio` is the case wall time over the
fastest backend measured for that case.

| case | backend | ms | ratio | commit | source tree | device | artifact |
|------|---------|----|-------|--------|-------------|--------|----------|
| `callgraph.reachability.step.262k` | cuda | 0.032 | 1.000 | 560d6f1796fd | source-tree-v1:80dab105d967 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-13-callgraph-reachability.json` |
| `callgraph.reachability.step.262k` | wgpu | 0.086 | 2.716 | 560d6f1796fd | source-tree-v1:80dab105d967 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-13-callgraph-reachability.json` |
| `compound.pipeline.fused_filter.1m` | cuda | 1.691 | 1.840 | 560d6f1796fd | source-tree-v1:80dab105d967 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-14-compound-fused-filter.json` |
| `compound.pipeline.fused_filter.1m` | wgpu | 0.919 | 1.000 | 560d6f1796fd | source-tree-v1:80dab105d967 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-14-compound-fused-filter.json` |
| `cuda.ptx.patterns.release.corpus` | cuda | 0.101 | 1.000 | 560d6f1796fd | source-tree-v1:80dab105d967 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/cuda-ptx-patterns.json` |
| `foundation.optimizer.impact` | cuda | 1.853 | 2.645 | 560d6f1796fd | source-tree-v1:80dab105d967 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-11-semantic-optimizer-impact.json` |
| `foundation.optimizer.impact` | wgpu | 0.701 | 1.000 | 560d6f1796fd | source-tree-v1:80dab105d967 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-11-semantic-optimizer-impact.json` |
| `foundation.reduce.sum.crossover` | cuda | 0.255 | 1.033 | 560d6f1796fd | source-tree-v1:80dab105d967 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-15-adaptive-routing.json` |
| `foundation.reduce.sum.crossover` | wgpu | 0.247 | 1.000 | 560d6f1796fd | source-tree-v1:80dab105d967 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-15-adaptive-routing.json` |
| `metadata.condition.filesize_header.1m` | cuda | 0.020 | 1.000 | 560d6f1796fd | source-tree-v1:80dab105d967 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-04-metadata-conditions.json` |
| `metadata.condition.filesize_header.1m` | wgpu | 0.072 | 3.614 | 560d6f1796fd | source-tree-v1:80dab105d967 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-04-metadata-conditions.json` |
| `nn.linear_4bit_affine_grouped.1m` | cuda | 0.006 | 1.002 | 560d6f1796fd | source-tree-v1:80dab105d967 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-16-quantized-linear.json` |
| `nn.linear_4bit_affine_grouped.1m` | wgpu | 0.006 | 1.000 | 560d6f1796fd | source-tree-v1:80dab105d967 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-16-quantized-linear.json` |
| `release.alias_reaching_def.1m` | cuda | 0.028 | 1.000 | 560d6f1796fd | source-tree-v1:80dab105d967 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-07-alias-reaching-def.json` |
| `release.alias_reaching_def.1m` | wgpu | 0.085 | 3.071 | 560d6f1796fd | source-tree-v1:80dab105d967 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-07-alias-reaching-def.json` |
| `release.ast_motif_traversal.1m` | cuda | 0.027 | 1.000 | 560d6f1796fd | source-tree-v1:80dab105d967 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-09-ast-motif-traversal.json` |
| `release.ast_motif_traversal.1m` | wgpu | 0.088 | 3.254 | 560d6f1796fd | source-tree-v1:80dab105d967 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-09-ast-motif-traversal.json` |
| `release.condition_eval.1m` | cuda | 0.030 | 1.000 | 560d6f1796fd | source-tree-v1:80dab105d967 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-01-condition-eval.json` |
| `release.condition_eval.1m` | wgpu | 0.089 | 2.998 | 560d6f1796fd | source-tree-v1:80dab105d967 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-01-condition-eval.json` |
| `release.egraph_saturation.1m` | cuda | 0.030 | 1.000 | 560d6f1796fd | source-tree-v1:80dab105d967 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-17-egraph-saturation.json` |
| `release.egraph_saturation.1m` | wgpu | 0.094 | 3.170 | 560d6f1796fd | source-tree-v1:80dab105d967 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-17-egraph-saturation.json` |
| `release.entropy_window.1m` | cuda | 0.030 | 1.000 | 560d6f1796fd | source-tree-v1:80dab105d967 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-05-entropy-window.json` |
| `release.entropy_window.1m` | wgpu | 0.082 | 2.746 | 560d6f1796fd | source-tree-v1:80dab105d967 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-05-entropy-window.json` |
| `release.ifds_witness.1m` | cuda | 0.031 | 1.000 | 560d6f1796fd | source-tree-v1:80dab105d967 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-08-ifds-witness.json` |
| `release.ifds_witness.1m` | wgpu | 0.088 | 2.821 | 560d6f1796fd | source-tree-v1:80dab105d967 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-08-ifds-witness.json` |
| `release.megakernel_queue.1m` | cuda | 0.029 | 1.000 | 560d6f1796fd | source-tree-v1:80dab105d967 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-10-megakernel-queued-batches.json` |
| `release.megakernel_queue.1m` | wgpu | 0.088 | 3.072 | 560d6f1796fd | source-tree-v1:80dab105d967 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-10-megakernel-queued-batches.json` |
| `release.offset_count_aggregation.1m` | cuda | 0.029 | 1.000 | 560d6f1796fd | source-tree-v1:80dab105d967 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-03-offset-count-aggregation.json` |
| `release.offset_count_aggregation.1m` | wgpu | 0.087 | 2.985 | 560d6f1796fd | source-tree-v1:80dab105d967 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-03-offset-count-aggregation.json` |
| `release.optimizer.resident_pipeline` | cuda | 668.803 | 1.000 | 560d6f1796fd | source-tree-v1:80dab105d967 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/resident-optimizer-pipeline.json` |
| `release.quantified_condition_loops.1m` | cuda | 0.027 | 1.000 | 560d6f1796fd | source-tree-v1:80dab105d967 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-06-quantified-condition-loops.json` |
| `release.quantified_condition_loops.1m` | wgpu | 0.089 | 3.235 | 560d6f1796fd | source-tree-v1:80dab105d967 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-06-quantified-condition-loops.json` |
| `release.string_bitmap_scatter.1m` | cuda | 0.066 | 1.000 | 560d6f1796fd | source-tree-v1:80dab105d967 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-02-string-bitmap-scatter.json` |
| `release.string_bitmap_scatter.1m` | wgpu | 0.096 | 1.450 | 560d6f1796fd | source-tree-v1:80dab105d967 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-02-string-bitmap-scatter.json` |
| `sparse.compaction.count.1m` | cuda | 0.018 | 1.000 | 560d6f1796fd | source-tree-v1:80dab105d967 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/workload-12-sparse-output-compaction.json` |
| `sparse.compaction.count.1m` | wgpu | 0.078 | 4.228 | 560d6f1796fd | source-tree-v1:80dab105d967 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-12-sparse-output-compaction.json` |

## declared without a measurement

| case | backend | declared by |
|------|---------|-------------|
| `release.optimizer.resident_pipeline` | wgpu | `release/evidence/benchmarks/resident-optimizer-pipeline.json` |
