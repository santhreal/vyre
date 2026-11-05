# cross-backend comparison

Produced by `./cargo_full run --bin xtask -- bench-crossback --write`. Every row is a
wall-clock reading a release benchmark suite recorded under
`release/evidence/benchmarks/`, with the commit, source-tree fingerprint and
device signature it was taken under. `ratio` is the case wall time over the
fastest backend measured for that case.

| case | backend | ms | ratio | commit | source tree | device | artifact |
|------|---------|----|-------|--------|-------------|--------|----------|
| `callgraph.reachability.step.262k` | cuda | 0.023 | 1.000 | 4b1c1f824d11 | source-tree-v1:d931dcfdec77 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-13-callgraph-reachability.json` |
| `callgraph.reachability.step.262k` | wgpu | 0.062 | 2.683 | 4b1c1f824d11 | source-tree-v1:d931dcfdec77 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-13-callgraph-reachability.json` |
| `compound.pipeline.fused_filter.1m` | cuda | 0.397 | 1.000 | 4b1c1f824d11 | source-tree-v1:d931dcfdec77 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-14-compound-fused-filter.json` |
| `compound.pipeline.fused_filter.1m` | wgpu | 0.428 | 1.078 | 4b1c1f824d11 | source-tree-v1:d931dcfdec77 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-14-compound-fused-filter.json` |
| `cuda.ptx.patterns.release.corpus` | cuda | 0.034 | 1.000 | 4b1c1f824d11 | source-tree-v1:d931dcfdec77 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/cuda-ptx-patterns.json` |
| `foundation.optimizer.impact` | cuda | 0.327 | 1.000 | 4b1c1f824d11 | source-tree-v1:d931dcfdec77 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-11-semantic-optimizer-impact.json` |
| `foundation.optimizer.impact` | wgpu | 0.361 | 1.102 | 4b1c1f824d11 | source-tree-v1:d931dcfdec77 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-11-semantic-optimizer-impact.json` |
| `foundation.reduce.sum.crossover` | cuda | 0.137 | 1.000 | 4b1c1f824d11 | source-tree-v1:d931dcfdec77 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-15-adaptive-routing.json` |
| `foundation.reduce.sum.crossover` | wgpu | 0.202 | 1.472 | 4b1c1f824d11 | source-tree-v1:d931dcfdec77 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-15-adaptive-routing.json` |
| `metadata.condition.filesize_header.1m` | cuda | 0.011 | 1.000 | 4b1c1f824d11 | source-tree-v1:d931dcfdec77 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-04-metadata-conditions.json` |
| `metadata.condition.filesize_header.1m` | wgpu | 0.055 | 5.114 | 4b1c1f824d11 | source-tree-v1:d931dcfdec77 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-04-metadata-conditions.json` |
| `nn.linear_4bit_affine_grouped.1m` | cuda | 0.003 | 1.000 | 4b1c1f824d11 | source-tree-v1:d931dcfdec77 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-16-quantized-linear.json` |
| `nn.linear_4bit_affine_grouped.1m` | wgpu | 0.003 | 1.103 | 4b1c1f824d11 | source-tree-v1:d931dcfdec77 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-16-quantized-linear.json` |
| `release.alias_reaching_def.1m` | cuda | 0.017 | 1.000 | 4b1c1f824d11 | source-tree-v1:d931dcfdec77 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-07-alias-reaching-def.json` |
| `release.alias_reaching_def.1m` | wgpu | 0.064 | 3.875 | 4b1c1f824d11 | source-tree-v1:d931dcfdec77 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-07-alias-reaching-def.json` |
| `release.ast_motif_traversal.1m` | cuda | 0.017 | 1.000 | 4b1c1f824d11 | source-tree-v1:d931dcfdec77 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-09-ast-motif-traversal.json` |
| `release.ast_motif_traversal.1m` | wgpu | 0.061 | 3.581 | 4b1c1f824d11 | source-tree-v1:d931dcfdec77 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-09-ast-motif-traversal.json` |
| `release.condition_eval.1m` | cuda | 0.017 | 1.000 | 4b1c1f824d11 | source-tree-v1:d931dcfdec77 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-01-condition-eval.json` |
| `release.condition_eval.1m` | wgpu | 0.062 | 3.670 | 4b1c1f824d11 | source-tree-v1:d931dcfdec77 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-01-condition-eval.json` |
| `release.egraph_saturation.1m` | cuda | 0.017 | 1.000 | 4b1c1f824d11 | source-tree-v1:d931dcfdec77 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-17-egraph-saturation.json` |
| `release.egraph_saturation.1m` | wgpu | 0.067 | 4.052 | 4b1c1f824d11 | source-tree-v1:d931dcfdec77 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-17-egraph-saturation.json` |
| `release.entropy_window.1m` | cuda | 0.018 | 1.000 | 4b1c1f824d11 | source-tree-v1:d931dcfdec77 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-05-entropy-window.json` |
| `release.entropy_window.1m` | wgpu | 0.060 | 3.270 | 4b1c1f824d11 | source-tree-v1:d931dcfdec77 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-05-entropy-window.json` |
| `release.ifds_witness.1m` | cuda | 0.017 | 1.000 | 4b1c1f824d11 | source-tree-v1:d931dcfdec77 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-08-ifds-witness.json` |
| `release.ifds_witness.1m` | wgpu | 0.066 | 3.941 | 4b1c1f824d11 | source-tree-v1:d931dcfdec77 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-08-ifds-witness.json` |
| `release.megakernel_queue.1m` | cuda | 0.017 | 1.000 | 4b1c1f824d11 | source-tree-v1:d931dcfdec77 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-10-megakernel-queued-batches.json` |
| `release.megakernel_queue.1m` | wgpu | 0.060 | 3.545 | 4b1c1f824d11 | source-tree-v1:d931dcfdec77 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-10-megakernel-queued-batches.json` |
| `release.offset_count_aggregation.1m` | cuda | 0.017 | 1.000 | 4b1c1f824d11 | source-tree-v1:d931dcfdec77 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-03-offset-count-aggregation.json` |
| `release.offset_count_aggregation.1m` | wgpu | 0.068 | 4.134 | 4b1c1f824d11 | source-tree-v1:d931dcfdec77 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-03-offset-count-aggregation.json` |
| `release.optimizer.resident_pipeline` | cuda | 557.258 | 1.000 | 4b1c1f824d11 | source-tree-v1:d931dcfdec77 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/resident-optimizer-pipeline.json` |
| `release.quantified_condition_loops.1m` | cuda | 0.017 | 1.000 | 4b1c1f824d11 | source-tree-v1:d931dcfdec77 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-06-quantified-condition-loops.json` |
| `release.quantified_condition_loops.1m` | wgpu | 0.061 | 3.630 | 4b1c1f824d11 | source-tree-v1:d931dcfdec77 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-06-quantified-condition-loops.json` |
| `release.string_bitmap_scatter.1m` | cuda | 0.028 | 1.000 | 4b1c1f824d11 | source-tree-v1:d931dcfdec77 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-02-string-bitmap-scatter.json` |
| `release.string_bitmap_scatter.1m` | wgpu | 0.080 | 2.920 | 4b1c1f824d11 | source-tree-v1:d931dcfdec77 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-02-string-bitmap-scatter.json` |
| `sparse.compaction.count.1m` | cuda | 0.016 | 1.000 | 4b1c1f824d11 | source-tree-v1:d931dcfdec77 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-12-sparse-output-compaction.json` |
| `sparse.compaction.count.1m` | wgpu | 0.062 | 3.913 | 4b1c1f824d11 | source-tree-v1:d931dcfdec77 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-12-sparse-output-compaction.json` |

## declared without a measurement

| case | backend | declared by |
|------|---------|-------------|
| `release.optimizer.resident_pipeline` | wgpu | `release/evidence/benchmarks/resident-optimizer-pipeline.json` |
