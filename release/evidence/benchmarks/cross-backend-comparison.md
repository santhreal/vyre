# cross-backend comparison

Produced by `./cargo_full run --bin xtask -- bench-crossback --write`. Every row is a
wall-clock reading a release benchmark suite recorded under
`release/evidence/benchmarks/`, with the commit, source-tree fingerprint and
device signature it was taken under. `ratio` is the case wall time over the
fastest backend measured for that case.

| case | backend | ms | ratio | commit | source tree | device | artifact |
|------|---------|----|-------|--------|-------------|--------|----------|
| `callgraph.reachability.step.262k` | cuda | 0.025 | 1.000 | 732351b4874e | source-tree-v1:e9a16de66a92 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-13-callgraph-reachability.json` |
| `callgraph.reachability.step.262k` | wgpu | 0.085 | 3.450 | 732351b4874e | source-tree-v1:e9a16de66a92 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-13-callgraph-reachability.json` |
| `compound.pipeline.fused_filter.1m` | cuda | 0.434 | 1.000 | 732351b4874e | source-tree-v1:e9a16de66a92 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-14-compound-fused-filter.json` |
| `compound.pipeline.fused_filter.1m` | wgpu | 0.486 | 1.119 | 732351b4874e | source-tree-v1:e9a16de66a92 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-14-compound-fused-filter.json` |
| `cuda.ptx.patterns.release.corpus` | cuda | 0.035 | 1.000 | 732351b4874e | source-tree-v1:e9a16de66a92 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/cuda-ptx-patterns.json` |
| `foundation.optimizer.impact` | cuda | 0.409 | 1.015 | 732351b4874e | source-tree-v1:e9a16de66a92 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-11-semantic-optimizer-impact.json` |
| `foundation.optimizer.impact` | wgpu | 0.403 | 1.000 | 732351b4874e | source-tree-v1:e9a16de66a92 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-11-semantic-optimizer-impact.json` |
| `foundation.reduce.sum.crossover` | cuda | 0.150 | 1.000 | 732351b4874e | source-tree-v1:e9a16de66a92 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-15-adaptive-routing.json` |
| `foundation.reduce.sum.crossover` | wgpu | 0.413 | 2.745 | 732351b4874e | source-tree-v1:e9a16de66a92 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-15-adaptive-routing.json` |
| `metadata.condition.filesize_header.1m` | cuda | 0.012 | 1.000 | 732351b4874e | source-tree-v1:e9a16de66a92 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-04-metadata-conditions.json` |
| `metadata.condition.filesize_header.1m` | wgpu | 0.078 | 6.565 | 732351b4874e | source-tree-v1:e9a16de66a92 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-04-metadata-conditions.json` |
| `nn.linear_4bit_affine_grouped.1m` | cuda | 0.004 | 1.000 | 732351b4874e | source-tree-v1:e9a16de66a92 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-16-quantized-linear.json` |
| `nn.linear_4bit_affine_grouped.1m` | wgpu | 0.004 | 1.093 | 732351b4874e | source-tree-v1:e9a16de66a92 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-16-quantized-linear.json` |
| `release.alias_reaching_def.1m` | cuda | 0.019 | 1.000 | 732351b4874e | source-tree-v1:e9a16de66a92 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-07-alias-reaching-def.json` |
| `release.alias_reaching_def.1m` | wgpu | 0.087 | 4.660 | 732351b4874e | source-tree-v1:e9a16de66a92 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-07-alias-reaching-def.json` |
| `release.ast_motif_traversal.1m` | cuda | 0.018 | 1.000 | 732351b4874e | source-tree-v1:e9a16de66a92 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-09-ast-motif-traversal.json` |
| `release.ast_motif_traversal.1m` | wgpu | 0.087 | 4.991 | 732351b4874e | source-tree-v1:e9a16de66a92 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-09-ast-motif-traversal.json` |
| `release.condition_eval.1m` | cuda | 0.018 | 1.000 | 732351b4874e | source-tree-v1:e9a16de66a92 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-01-condition-eval.json` |
| `release.condition_eval.1m` | wgpu | 0.086 | 4.641 | 732351b4874e | source-tree-v1:e9a16de66a92 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-01-condition-eval.json` |
| `release.egraph_saturation.1m` | cuda | 0.020 | 1.000 | 732351b4874e | source-tree-v1:e9a16de66a92 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-17-egraph-saturation.json` |
| `release.egraph_saturation.1m` | wgpu | 0.090 | 4.571 | 732351b4874e | source-tree-v1:e9a16de66a92 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-17-egraph-saturation.json` |
| `release.entropy_window.1m` | cuda | 0.018 | 1.000 | 732351b4874e | source-tree-v1:e9a16de66a92 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-05-entropy-window.json` |
| `release.entropy_window.1m` | wgpu | 0.088 | 4.877 | 732351b4874e | source-tree-v1:e9a16de66a92 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-05-entropy-window.json` |
| `release.ifds_witness.1m` | cuda | 0.018 | 1.000 | 732351b4874e | source-tree-v1:e9a16de66a92 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-08-ifds-witness.json` |
| `release.ifds_witness.1m` | wgpu | 0.093 | 5.299 | 732351b4874e | source-tree-v1:e9a16de66a92 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-08-ifds-witness.json` |
| `release.megakernel_queue.1m` | cuda | 0.021 | 1.000 | 732351b4874e | source-tree-v1:e9a16de66a92 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-10-megakernel-queued-batches.json` |
| `release.megakernel_queue.1m` | wgpu | 0.089 | 4.267 | 732351b4874e | source-tree-v1:e9a16de66a92 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-10-megakernel-queued-batches.json` |
| `release.offset_count_aggregation.1m` | cuda | 0.017 | 1.000 | 732351b4874e | source-tree-v1:e9a16de66a92 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-03-offset-count-aggregation.json` |
| `release.offset_count_aggregation.1m` | wgpu | 0.086 | 4.971 | 732351b4874e | source-tree-v1:e9a16de66a92 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-03-offset-count-aggregation.json` |
| `release.optimizer.resident_pipeline` | cuda | 112.930 | 1.000 | 732351b4874e | source-tree-v1:e9a16de66a92 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/resident-optimizer-pipeline.json` |
| `release.quantified_condition_loops.1m` | cuda | 0.021 | 1.000 | 732351b4874e | source-tree-v1:e9a16de66a92 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-06-quantified-condition-loops.json` |
| `release.quantified_condition_loops.1m` | wgpu | 0.092 | 4.382 | 732351b4874e | source-tree-v1:e9a16de66a92 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-06-quantified-condition-loops.json` |
| `release.string_bitmap_scatter.1m` | cuda | 0.033 | 1.000 | 732351b4874e | source-tree-v1:e9a16de66a92 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-02-string-bitmap-scatter.json` |
| `release.string_bitmap_scatter.1m` | wgpu | 0.090 | 2.695 | 732351b4874e | source-tree-v1:e9a16de66a92 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-02-string-bitmap-scatter.json` |
| `sparse.compaction.count.1m` | cuda | 0.016 | 1.000 | 732351b4874e | source-tree-v1:e9a16de66a92 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-12-sparse-output-compaction.json` |
| `sparse.compaction.count.1m` | wgpu | 0.089 | 5.498 | 732351b4874e | source-tree-v1:e9a16de66a92 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-12-sparse-output-compaction.json` |

## declared without a measurement

| case | backend | declared by |
|------|---------|-------------|
| `release.optimizer.resident_pipeline` | wgpu | `release/evidence/benchmarks/resident-optimizer-pipeline.json` |
