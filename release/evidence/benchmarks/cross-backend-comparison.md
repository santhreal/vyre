# cross-backend comparison

Produced by `./cargo_full run --bin xtask -- bench-crossback --write`. Every row is a
wall-clock reading a release benchmark suite recorded under
`release/evidence/benchmarks/`, with the commit, source-tree fingerprint and
device signature it was taken under. `ratio` is the case wall time over the
fastest backend measured for that case.

| case | backend | ms | ratio | commit | source tree | device | artifact |
|------|---------|----|-------|--------|-------------|--------|----------|
| `callgraph.reachability.step.262k` | cuda | 0.087 | 1.000 | d2cddddd1fd9 | source-tree-v1:4dd843377040 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-13-callgraph-reachability.json` |
| `callgraph.reachability.step.262k` | wgpu | 0.088 | 1.014 | 24b6787218f5 | source-tree-v1:c23ec3b17257 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-13-callgraph-reachability.json` |
| `compound.pipeline.fused_filter.1m` | cuda | 1.969 | 2.217 | d2cddddd1fd9 | source-tree-v1:4dd843377040 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-14-compound-fused-filter.json` |
| `compound.pipeline.fused_filter.1m` | wgpu | 0.888 | 1.000 | 24b6787218f5 | source-tree-v1:c23ec3b17257 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-14-compound-fused-filter.json` |
| `cuda.ptx.patterns.release.corpus` | cuda | 0.049 | 1.000 | 24b6787218f5 | source-tree-v1:c23ec3b17257 | device-profile-v1:9311a5c6da66 | `release/evidence/benchmarks/cuda-ptx-patterns.json` |
| `foundation.optimizer.impact` | cuda | 0.497 | 1.000 | d2cddddd1fd9 | source-tree-v1:298740d7e903 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-11-semantic-optimizer-impact.json` |
| `foundation.optimizer.impact` | wgpu | 1.451 | 2.918 | 24b6787218f5 | source-tree-v1:c23ec3b17257 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-11-semantic-optimizer-impact.json` |
| `foundation.reduce.sum.crossover` | wgpu | 1.210 | 1.000 | 24b6787218f5 | source-tree-v1:c23ec3b17257 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-15-adaptive-routing.json` |
| `metadata.condition.filesize_header.1m` | cuda | 0.021 | 1.000 | d2cddddd1fd9 | source-tree-v1:76769df45f2d | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-04-metadata-conditions.json` |
| `metadata.condition.filesize_header.1m` | wgpu | 0.070 | 3.274 | 24b6787218f5 | source-tree-v1:c23ec3b17257 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-04-metadata-conditions.json` |
| `nn.linear_4bit_affine_grouped.1m` | cuda | 0.004 | 1.000 | d2cddddd1fd9 | source-tree-v1:4dd843377040 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-16-quantized-linear.json` |
| `nn.linear_4bit_affine_grouped.1m` | wgpu | 0.005 | 1.516 | 24b6787218f5 | source-tree-v1:c23ec3b17257 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-16-quantized-linear.json` |
| `release.alias_reaching_def.1m` | cuda | 0.036 | 1.000 | d2cddddd1fd9 | source-tree-v1:76769df45f2d | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-07-alias-reaching-def.json` |
| `release.alias_reaching_def.1m` | wgpu | 0.087 | 2.413 | 24b6787218f5 | source-tree-v1:c23ec3b17257 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-07-alias-reaching-def.json` |
| `release.ast_motif_traversal.1m` | cuda | 0.043 | 1.000 | d2cddddd1fd9 | source-tree-v1:76769df45f2d | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-09-ast-motif-traversal.json` |
| `release.ast_motif_traversal.1m` | wgpu | 0.084 | 1.947 | 24b6787218f5 | source-tree-v1:c23ec3b17257 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-09-ast-motif-traversal.json` |
| `release.condition_eval.1m` | cuda | 0.044 | 1.000 | d2cddddd1fd9 | source-tree-v1:6fe360c4d0bd | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-01-condition-eval.json` |
| `release.condition_eval.1m` | wgpu | 0.086 | 1.936 | 24b6787218f5 | source-tree-v1:c23ec3b17257 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-01-condition-eval.json` |
| `release.egraph_saturation.1m` | cuda | 0.041 | 1.000 | d2cddddd1fd9 | source-tree-v1:4dd843377040 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-17-egraph-saturation.json` |
| `release.egraph_saturation.1m` | wgpu | 0.085 | 2.062 | 24b6787218f5 | source-tree-v1:c23ec3b17257 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-17-egraph-saturation.json` |
| `release.entropy_window.1m` | cuda | 0.032 | 1.000 | d2cddddd1fd9 | source-tree-v1:76769df45f2d | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-05-entropy-window.json` |
| `release.entropy_window.1m` | wgpu | 0.084 | 2.629 | 24b6787218f5 | source-tree-v1:c23ec3b17257 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-05-entropy-window.json` |
| `release.ifds_witness.1m` | cuda | 0.036 | 1.000 | d2cddddd1fd9 | source-tree-v1:76769df45f2d | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-08-ifds-witness.json` |
| `release.ifds_witness.1m` | wgpu | 0.085 | 2.352 | 24b6787218f5 | source-tree-v1:c23ec3b17257 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-08-ifds-witness.json` |
| `release.megakernel_queue.1m` | cuda | 0.046 | 1.000 | d2cddddd1fd9 | source-tree-v1:5fc119af36c9 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-10-megakernel-queued-batches.json` |
| `release.megakernel_queue.1m` | wgpu | 0.084 | 1.832 | 24b6787218f5 | source-tree-v1:c23ec3b17257 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-10-megakernel-queued-batches.json` |
| `release.offset_count_aggregation.1m` | cuda | 0.046 | 1.000 | d2cddddd1fd9 | source-tree-v1:76769df45f2d | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-03-offset-count-aggregation.json` |
| `release.offset_count_aggregation.1m` | wgpu | 0.094 | 2.029 | 24b6787218f5 | source-tree-v1:c23ec3b17257 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-03-offset-count-aggregation.json` |
| `release.optimizer.resident_pipeline` | cuda | 526.014 | 1.000 | a08d4b239de9 | source-tree-v1:1e1880d56007 | device-profile-v1:f04bd6c0b35f | `release/evidence/benchmarks/resident-optimizer-pipeline.json` |
| `release.quantified_condition_loops.1m` | cuda | 0.042 | 1.000 | d2cddddd1fd9 | source-tree-v1:76769df45f2d | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-06-quantified-condition-loops.json` |
| `release.quantified_condition_loops.1m` | wgpu | 0.086 | 2.062 | 24b6787218f5 | source-tree-v1:c23ec3b17257 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-06-quantified-condition-loops.json` |
| `release.string_bitmap_scatter.1m` | cuda | 0.118 | 1.187 | d2cddddd1fd9 | source-tree-v1:76769df45f2d | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-02-string-bitmap-scatter.json` |
| `release.string_bitmap_scatter.1m` | wgpu | 0.099 | 1.000 | 24b6787218f5 | source-tree-v1:c23ec3b17257 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-02-string-bitmap-scatter.json` |
| `runtime.adaptive_routing.gpu_resident.1m` | cuda | 1.578 | 1.000 | b2b94c9fc2db | source-tree-v1:7836a93bd759 | device-profile-v1:f04bd6c0b35f | `release/evidence/benchmarks/workload-15-adaptive-routing.json` |
| `sparse.compaction.count.1m` | cuda | 0.036 | 1.000 | d2cddddd1fd9 | source-tree-v1:4dd843377040 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-12-sparse-output-compaction.json` |
| `sparse.compaction.count.1m` | wgpu | 0.076 | 2.092 | 24b6787218f5 | source-tree-v1:c23ec3b17257 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-12-sparse-output-compaction.json` |

## declared without a measurement

| case | backend | declared by |
|------|---------|-------------|
| `foundation.reduce.sum.crossover` | cuda | `release/evidence/benchmarks/wgpu-workload-15-adaptive-routing.json` |
| `release.optimizer.resident_pipeline` | wgpu | `release/evidence/benchmarks/resident-optimizer-pipeline.json` |
| `runtime.adaptive_routing.gpu_resident.1m` | wgpu | `release/evidence/benchmarks/workload-15-adaptive-routing.json` |
