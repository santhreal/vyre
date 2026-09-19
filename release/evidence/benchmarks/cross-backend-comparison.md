# cross-backend comparison

Produced by `./cargo_full run --bin xtask -- bench-crossback --write`. Every row is a
wall-clock reading a release benchmark suite recorded under
`release/evidence/benchmarks/`, with the commit, source-tree fingerprint and
device signature it was taken under. `ratio` is the case wall time over the
fastest backend measured for that case.

| case | backend | ms | ratio | commit | source tree | device | artifact |
|------|---------|----|-------|--------|-------------|--------|----------|
| `callgraph.reachability.step.262k` | cuda | 0.021 | 1.000 | afc3515fbbb0 | source-tree-v1:4cbb2e8de845 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-13-callgraph-reachability.json` |
| `callgraph.reachability.step.262k` | wgpu | 0.076 | 3.649 | afc3515fbbb0 | source-tree-v1:4cbb2e8de845 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-13-callgraph-reachability.json` |
| `compound.pipeline.fused_filter.1m` | cuda | 0.403 | 1.000 | afc3515fbbb0 | source-tree-v1:4cbb2e8de845 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-14-compound-fused-filter.json` |
| `compound.pipeline.fused_filter.1m` | wgpu | 0.460 | 1.143 | afc3515fbbb0 | source-tree-v1:4cbb2e8de845 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-14-compound-fused-filter.json` |
| `cuda.ptx.patterns.release.corpus` | cuda | 0.036 | 1.000 | afc3515fbbb0 | source-tree-v1:4cbb2e8de845 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/cuda-ptx-patterns.json` |
| `foundation.optimizer.impact` | cuda | 0.716 | 2.010 | afc3515fbbb0 | source-tree-v1:4cbb2e8de845 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-11-semantic-optimizer-impact.json` |
| `foundation.optimizer.impact` | wgpu | 0.356 | 1.000 | afc3515fbbb0 | source-tree-v1:4cbb2e8de845 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-11-semantic-optimizer-impact.json` |
| `foundation.reduce.sum.crossover` | cuda | 0.131 | 1.000 | afc3515fbbb0 | source-tree-v1:4cbb2e8de845 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-15-adaptive-routing.json` |
| `foundation.reduce.sum.crossover` | wgpu | 0.243 | 1.848 | afc3515fbbb0 | source-tree-v1:4cbb2e8de845 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-15-adaptive-routing.json` |
| `metadata.condition.filesize_header.1m` | cuda | 0.012 | 1.000 | afc3515fbbb0 | source-tree-v1:4cbb2e8de845 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-04-metadata-conditions.json` |
| `metadata.condition.filesize_header.1m` | wgpu | 0.066 | 5.689 | afc3515fbbb0 | source-tree-v1:4cbb2e8de845 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-04-metadata-conditions.json` |
| `nn.linear_4bit_affine_grouped.1m` | cuda | 0.003 | 1.000 | afc3515fbbb0 | source-tree-v1:4cbb2e8de845 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-16-quantized-linear.json` |
| `nn.linear_4bit_affine_grouped.1m` | wgpu | 0.004 | 1.158 | afc3515fbbb0 | source-tree-v1:4cbb2e8de845 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-16-quantized-linear.json` |
| `release.alias_reaching_def.1m` | cuda | 0.017 | 1.000 | afc3515fbbb0 | source-tree-v1:4cbb2e8de845 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-07-alias-reaching-def.json` |
| `release.alias_reaching_def.1m` | wgpu | 0.078 | 4.563 | afc3515fbbb0 | source-tree-v1:4cbb2e8de845 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-07-alias-reaching-def.json` |
| `release.ast_motif_traversal.1m` | cuda | 0.018 | 1.000 | afc3515fbbb0 | source-tree-v1:4cbb2e8de845 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-09-ast-motif-traversal.json` |
| `release.ast_motif_traversal.1m` | wgpu | 0.076 | 4.230 | afc3515fbbb0 | source-tree-v1:4cbb2e8de845 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-09-ast-motif-traversal.json` |
| `release.condition_eval.1m` | cuda | 0.017 | 1.000 | afc3515fbbb0 | source-tree-v1:4cbb2e8de845 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-01-condition-eval.json` |
| `release.condition_eval.1m` | wgpu | 0.076 | 4.418 | afc3515fbbb0 | source-tree-v1:4cbb2e8de845 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-01-condition-eval.json` |
| `release.egraph_saturation.1m` | cuda | 0.020 | 1.000 | afc3515fbbb0 | source-tree-v1:4cbb2e8de845 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-17-egraph-saturation.json` |
| `release.egraph_saturation.1m` | wgpu | 0.082 | 4.180 | afc3515fbbb0 | source-tree-v1:4cbb2e8de845 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-17-egraph-saturation.json` |
| `release.entropy_window.1m` | cuda | 0.017 | 1.000 | afc3515fbbb0 | source-tree-v1:4cbb2e8de845 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-05-entropy-window.json` |
| `release.entropy_window.1m` | wgpu | 0.081 | 4.748 | afc3515fbbb0 | source-tree-v1:4cbb2e8de845 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-05-entropy-window.json` |
| `release.ifds_witness.1m` | cuda | 0.017 | 1.000 | afc3515fbbb0 | source-tree-v1:4cbb2e8de845 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-08-ifds-witness.json` |
| `release.ifds_witness.1m` | wgpu | 0.073 | 4.324 | afc3515fbbb0 | source-tree-v1:4cbb2e8de845 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-08-ifds-witness.json` |
| `release.megakernel_queue.1m` | cuda | 0.017 | 1.000 | afc3515fbbb0 | source-tree-v1:4cbb2e8de845 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-10-megakernel-queued-batches.json` |
| `release.megakernel_queue.1m` | wgpu | 0.075 | 4.480 | afc3515fbbb0 | source-tree-v1:4cbb2e8de845 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-10-megakernel-queued-batches.json` |
| `release.offset_count_aggregation.1m` | cuda | 0.017 | 1.000 | afc3515fbbb0 | source-tree-v1:4cbb2e8de845 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-03-offset-count-aggregation.json` |
| `release.offset_count_aggregation.1m` | wgpu | 0.081 | 4.795 | afc3515fbbb0 | source-tree-v1:4cbb2e8de845 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-03-offset-count-aggregation.json` |
| `release.optimizer.resident_pipeline` | cuda | 106.643 | 1.000 | afc3515fbbb0 | source-tree-v1:4cbb2e8de845 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/resident-optimizer-pipeline.json` |
| `release.quantified_condition_loops.1m` | cuda | 0.017 | 1.000 | afc3515fbbb0 | source-tree-v1:4cbb2e8de845 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-06-quantified-condition-loops.json` |
| `release.quantified_condition_loops.1m` | wgpu | 0.070 | 4.164 | afc3515fbbb0 | source-tree-v1:4cbb2e8de845 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-06-quantified-condition-loops.json` |
| `release.string_bitmap_scatter.1m` | cuda | 0.028 | 1.000 | afc3515fbbb0 | source-tree-v1:4cbb2e8de845 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-02-string-bitmap-scatter.json` |
| `release.string_bitmap_scatter.1m` | wgpu | 0.084 | 2.941 | afc3515fbbb0 | source-tree-v1:4cbb2e8de845 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-02-string-bitmap-scatter.json` |
| `sparse.compaction.count.1m` | cuda | 0.016 | 1.000 | afc3515fbbb0 | source-tree-v1:4cbb2e8de845 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-12-sparse-output-compaction.json` |
| `sparse.compaction.count.1m` | wgpu | 0.070 | 4.472 | afc3515fbbb0 | source-tree-v1:4cbb2e8de845 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-12-sparse-output-compaction.json` |

## declared without a measurement

| case | backend | declared by |
|------|---------|-------------|
| `release.optimizer.resident_pipeline` | wgpu | `release/evidence/benchmarks/resident-optimizer-pipeline.json` |
