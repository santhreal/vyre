# cross-backend comparison

Produced by `./cargo_full run --bin xtask -- bench-crossback --write`. Every row is a
wall-clock reading a release benchmark suite recorded under
`release/evidence/benchmarks/`, with the commit, source-tree fingerprint and
device signature it was taken under. `ratio` is the case wall time over the
fastest backend measured for that case.

| case | backend | ms | ratio | commit | source tree | device | artifact |
|------|---------|----|-------|--------|-------------|--------|----------|
| `callgraph.reachability.step.262k` | cuda | 0.052 | 1.000 | 0156a229d252 | source-tree-v1:a18936040707 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-13-callgraph-reachability.json` |
| `callgraph.reachability.step.262k` | wgpu | 0.160 | 3.102 | 0156a229d252 | source-tree-v1:a18936040707 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-13-callgraph-reachability.json` |
| `compound.pipeline.fused_filter.1m` | cuda | 0.482 | 1.000 | 0156a229d252 | source-tree-v1:a18936040707 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-14-compound-fused-filter.json` |
| `compound.pipeline.fused_filter.1m` | wgpu | 0.660 | 1.371 | 0156a229d252 | source-tree-v1:a18936040707 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-14-compound-fused-filter.json` |
| `cuda.ptx.patterns.release.corpus` | cuda | 0.037 | 1.000 | 0156a229d252 | source-tree-v1:a18936040707 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/cuda-ptx-patterns.json` |
| `foundation.optimizer.impact` | cuda | 0.596 | 1.334 | 0156a229d252 | source-tree-v1:a18936040707 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-11-semantic-optimizer-impact.json` |
| `foundation.optimizer.impact` | wgpu | 0.447 | 1.000 | 0156a229d252 | source-tree-v1:a18936040707 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-11-semantic-optimizer-impact.json` |
| `foundation.reduce.sum.crossover` | cuda | 0.207 | 1.000 | 0156a229d252 | source-tree-v1:a18936040707 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-15-adaptive-routing.json` |
| `foundation.reduce.sum.crossover` | wgpu | 0.246 | 1.185 | 0156a229d252 | source-tree-v1:a18936040707 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-15-adaptive-routing.json` |
| `metadata.condition.filesize_header.1m` | cuda | 0.015 | 1.000 | 0156a229d252 | source-tree-v1:a18936040707 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-04-metadata-conditions.json` |
| `metadata.condition.filesize_header.1m` | wgpu | 0.106 | 7.129 | 0156a229d252 | source-tree-v1:a18936040707 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-04-metadata-conditions.json` |
| `nn.linear_4bit_affine_grouped.1m` | cuda | 0.003 | 1.000 | 0156a229d252 | source-tree-v1:a18936040707 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-16-quantized-linear.json` |
| `nn.linear_4bit_affine_grouped.1m` | wgpu | 0.005 | 1.608 | 0156a229d252 | source-tree-v1:a18936040707 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-16-quantized-linear.json` |
| `release.alias_reaching_def.1m` | cuda | 0.020 | 1.000 | 0156a229d252 | source-tree-v1:a18936040707 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-07-alias-reaching-def.json` |
| `release.alias_reaching_def.1m` | wgpu | 0.138 | 7.018 | 0156a229d252 | source-tree-v1:a18936040707 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-07-alias-reaching-def.json` |
| `release.ast_motif_traversal.1m` | cuda | 0.018 | 1.000 | 0156a229d252 | source-tree-v1:a18936040707 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-09-ast-motif-traversal.json` |
| `release.ast_motif_traversal.1m` | wgpu | 0.141 | 8.029 | 0156a229d252 | source-tree-v1:a18936040707 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-09-ast-motif-traversal.json` |
| `release.condition_eval.1m` | cuda | 0.024 | 1.000 | 0156a229d252 | source-tree-v1:a18936040707 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-01-condition-eval.json` |
| `release.condition_eval.1m` | wgpu | 0.145 | 6.054 | 0156a229d252 | source-tree-v1:a18936040707 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-01-condition-eval.json` |
| `release.egraph_saturation.1m` | cuda | 0.024 | 1.000 | 0156a229d252 | source-tree-v1:a18936040707 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-17-egraph-saturation.json` |
| `release.egraph_saturation.1m` | wgpu | 0.131 | 5.443 | 0156a229d252 | source-tree-v1:a18936040707 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-17-egraph-saturation.json` |
| `release.entropy_window.1m` | cuda | 0.024 | 1.000 | 0156a229d252 | source-tree-v1:a18936040707 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-05-entropy-window.json` |
| `release.entropy_window.1m` | wgpu | 0.121 | 5.005 | 0156a229d252 | source-tree-v1:a18936040707 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-05-entropy-window.json` |
| `release.ifds_witness.1m` | cuda | 0.018 | 1.000 | 0156a229d252 | source-tree-v1:a18936040707 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-08-ifds-witness.json` |
| `release.ifds_witness.1m` | wgpu | 0.157 | 8.552 | 0156a229d252 | source-tree-v1:a18936040707 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-08-ifds-witness.json` |
| `release.megakernel_queue.1m` | cuda | 0.022 | 1.000 | 0156a229d252 | source-tree-v1:a18936040707 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-10-megakernel-queued-batches.json` |
| `release.megakernel_queue.1m` | wgpu | 0.149 | 6.822 | 0156a229d252 | source-tree-v1:a18936040707 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-10-megakernel-queued-batches.json` |
| `release.offset_count_aggregation.1m` | cuda | 0.021 | 1.000 | 0156a229d252 | source-tree-v1:a18936040707 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-03-offset-count-aggregation.json` |
| `release.offset_count_aggregation.1m` | wgpu | 0.118 | 5.646 | 0156a229d252 | source-tree-v1:a18936040707 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-03-offset-count-aggregation.json` |
| `release.optimizer.resident_pipeline` | cuda | 112.429 | 1.000 | 0156a229d252 | source-tree-v1:a18936040707 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/resident-optimizer-pipeline.json` |
| `release.quantified_condition_loops.1m` | cuda | 0.017 | 1.000 | 0156a229d252 | source-tree-v1:a18936040707 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-06-quantified-condition-loops.json` |
| `release.quantified_condition_loops.1m` | wgpu | 0.138 | 8.048 | 0156a229d252 | source-tree-v1:a18936040707 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-06-quantified-condition-loops.json` |
| `release.string_bitmap_scatter.1m` | cuda | 0.040 | 1.000 | 0156a229d252 | source-tree-v1:a18936040707 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-02-string-bitmap-scatter.json` |
| `release.string_bitmap_scatter.1m` | wgpu | 0.126 | 3.139 | 0156a229d252 | source-tree-v1:a18936040707 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-02-string-bitmap-scatter.json` |
| `sparse.compaction.count.1m` | cuda | 0.022 | 1.000 | 0156a229d252 | source-tree-v1:a18936040707 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-12-sparse-output-compaction.json` |
| `sparse.compaction.count.1m` | wgpu | 0.126 | 5.842 | 0156a229d252 | source-tree-v1:a18936040707 | device-profile-v1:4f188825b2ad | `release/evidence/benchmarks/wgpu-workload-12-sparse-output-compaction.json` |

## declared without a measurement

| case | backend | declared by |
|------|---------|-------------|
| `release.optimizer.resident_pipeline` | wgpu | `release/evidence/benchmarks/resident-optimizer-pipeline.json` |
