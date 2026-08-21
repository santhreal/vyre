# cross-backend comparison

Produced by `./cargo_full run --bin xtask -- bench-crossback --write`. Every row is a
wall-clock reading a release benchmark suite recorded under
`release/evidence/benchmarks/`, with the commit, source-tree fingerprint and
device signature it was taken under. `ratio` is the case wall time over the
fastest backend measured for that case.

| case | backend | ms | ratio | commit | source tree | device | artifact |
|------|---------|----|-------|--------|-------------|--------|----------|
| `callgraph.reachability.step.262k` | cuda | 0.065 | 1.000 | 492d2958eeae | source-tree-v1:5ff3fb0bd9db | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-13-callgraph-reachability.json` |
| `callgraph.reachability.step.262k` | wgpu | 0.124 | 1.906 | 492d2958eeae | source-tree-v1:5ff3fb0bd9db | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-13-callgraph-reachability.json` |
| `compound.pipeline.fused_filter.1m` | cuda | 1.661 | 2.944 | 492d2958eeae | source-tree-v1:5ff3fb0bd9db | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-14-compound-fused-filter.json` |
| `compound.pipeline.fused_filter.1m` | wgpu | 0.564 | 1.000 | 492d2958eeae | source-tree-v1:5ff3fb0bd9db | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-14-compound-fused-filter.json` |
| `cuda.ptx.patterns.release.corpus` | cuda | 0.031 | 1.000 | 492d2958eeae | source-tree-v1:5ff3fb0bd9db | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/cuda-ptx-patterns.json` |
| `foundation.optimizer.impact` | cuda | 0.441 | 1.000 | 492d2958eeae | source-tree-v1:5ff3fb0bd9db | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-11-semantic-optimizer-impact.json` |
| `foundation.optimizer.impact` | wgpu | 0.965 | 2.190 | 492d2958eeae | source-tree-v1:5ff3fb0bd9db | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-11-semantic-optimizer-impact.json` |
| `foundation.reduce.sum.crossover` | cuda | 1.053 | 1.000 | 492d2958eeae | source-tree-v1:5ff3fb0bd9db | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-15-adaptive-routing.json` |
| `foundation.reduce.sum.crossover` | wgpu | 1.200 | 1.140 | 492d2958eeae | source-tree-v1:5ff3fb0bd9db | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-15-adaptive-routing.json` |
| `metadata.condition.filesize_header.1m` | cuda | 0.036 | 1.000 | 492d2958eeae | source-tree-v1:5ff3fb0bd9db | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-04-metadata-conditions.json` |
| `metadata.condition.filesize_header.1m` | wgpu | 0.109 | 3.007 | 492d2958eeae | source-tree-v1:5ff3fb0bd9db | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-04-metadata-conditions.json` |
| `nn.linear_4bit_affine_grouped.1m` | cuda | 0.004 | 1.000 | 492d2958eeae | source-tree-v1:5ff3fb0bd9db | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-16-quantized-linear.json` |
| `nn.linear_4bit_affine_grouped.1m` | wgpu | 0.004 | 1.064 | 492d2958eeae | source-tree-v1:5ff3fb0bd9db | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-16-quantized-linear.json` |
| `release.alias_reaching_def.1m` | cuda | 0.041 | 1.000 | 492d2958eeae | source-tree-v1:5ff3fb0bd9db | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-07-alias-reaching-def.json` |
| `release.alias_reaching_def.1m` | wgpu | 0.110 | 2.655 | 492d2958eeae | source-tree-v1:5ff3fb0bd9db | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-07-alias-reaching-def.json` |
| `release.ast_motif_traversal.1m` | cuda | 0.048 | 1.000 | 492d2958eeae | source-tree-v1:5ff3fb0bd9db | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-09-ast-motif-traversal.json` |
| `release.ast_motif_traversal.1m` | wgpu | 0.115 | 2.399 | 492d2958eeae | source-tree-v1:5ff3fb0bd9db | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-09-ast-motif-traversal.json` |
| `release.condition_eval.1m` | cuda | 0.045 | 1.000 | 492d2958eeae | source-tree-v1:5ff3fb0bd9db | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-01-condition-eval.json` |
| `release.condition_eval.1m` | wgpu | 0.098 | 2.182 | 492d2958eeae | source-tree-v1:5ff3fb0bd9db | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-01-condition-eval.json` |
| `release.egraph_saturation.1m` | cuda | 0.062 | 1.000 | 492d2958eeae | source-tree-v1:5ff3fb0bd9db | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-17-egraph-saturation.json` |
| `release.egraph_saturation.1m` | wgpu | 0.121 | 1.951 | 492d2958eeae | source-tree-v1:5ff3fb0bd9db | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-17-egraph-saturation.json` |
| `release.entropy_window.1m` | cuda | 0.043 | 1.000 | 492d2958eeae | source-tree-v1:5ff3fb0bd9db | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-05-entropy-window.json` |
| `release.entropy_window.1m` | wgpu | 0.122 | 2.817 | 492d2958eeae | source-tree-v1:5ff3fb0bd9db | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-05-entropy-window.json` |
| `release.ifds_witness.1m` | cuda | 0.038 | 1.000 | 492d2958eeae | source-tree-v1:5ff3fb0bd9db | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-08-ifds-witness.json` |
| `release.ifds_witness.1m` | wgpu | 0.109 | 2.842 | 492d2958eeae | source-tree-v1:5ff3fb0bd9db | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-08-ifds-witness.json` |
| `release.megakernel_queue.1m` | cuda | 0.045 | 1.000 | 492d2958eeae | source-tree-v1:5ff3fb0bd9db | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-10-megakernel-queued-batches.json` |
| `release.megakernel_queue.1m` | wgpu | 0.117 | 2.621 | 492d2958eeae | source-tree-v1:5ff3fb0bd9db | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-10-megakernel-queued-batches.json` |
| `release.offset_count_aggregation.1m` | cuda | 0.069 | 1.000 | 492d2958eeae | source-tree-v1:5ff3fb0bd9db | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-03-offset-count-aggregation.json` |
| `release.offset_count_aggregation.1m` | wgpu | 0.102 | 1.496 | 492d2958eeae | source-tree-v1:5ff3fb0bd9db | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-03-offset-count-aggregation.json` |
| `release.quantified_condition_loops.1m` | cuda | 0.043 | 1.000 | 492d2958eeae | source-tree-v1:5ff3fb0bd9db | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-06-quantified-condition-loops.json` |
| `release.quantified_condition_loops.1m` | wgpu | 0.119 | 2.768 | 492d2958eeae | source-tree-v1:5ff3fb0bd9db | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-06-quantified-condition-loops.json` |
| `release.string_bitmap_scatter.1m` | cuda | 0.106 | 1.011 | 492d2958eeae | source-tree-v1:5ff3fb0bd9db | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-02-string-bitmap-scatter.json` |
| `release.string_bitmap_scatter.1m` | wgpu | 0.105 | 1.000 | 492d2958eeae | source-tree-v1:5ff3fb0bd9db | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-02-string-bitmap-scatter.json` |
| `sparse.compaction.count.1m` | cuda | 0.033 | 1.000 | 492d2958eeae | source-tree-v1:5ff3fb0bd9db | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-12-sparse-output-compaction.json` |
| `sparse.compaction.count.1m` | wgpu | 0.090 | 2.754 | 492d2958eeae | source-tree-v1:5ff3fb0bd9db | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-12-sparse-output-compaction.json` |

## declared without a measurement

Every backend a case contract declares carries a measurement.
