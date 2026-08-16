# cross-backend comparison

Produced by `./cargo_full run --bin xtask -- bench-crossback --write`. Every row is a
wall-clock reading a release benchmark suite recorded under
`release/evidence/benchmarks/`, with the commit, source-tree fingerprint and
device signature it was taken under. `ratio` is the case wall time over the
fastest backend measured for that case.

| case | backend | ms | ratio | commit | source tree | device | artifact |
|------|---------|----|-------|--------|-------------|--------|----------|
| `callgraph.reachability.step.262k` | cuda | 0.129 | 1.242 | 7a82c191b179 | source-tree-v1:5ced8470d96a | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-13-callgraph-reachability.json` |
| `callgraph.reachability.step.262k` | wgpu | 0.104 | 1.000 | 7a82c191b179 | source-tree-v1:5ced8470d96a | device-profile-v1:27e1acef5b70 | `release/evidence/benchmarks/wgpu-workload-13-callgraph-reachability.json` |
| `compound.pipeline.fused_filter.1m` | cuda | 0.956 | 1.209 | 7a82c191b179 | source-tree-v1:5ced8470d96a | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-14-compound-fused-filter.json` |
| `compound.pipeline.fused_filter.1m` | wgpu | 0.791 | 1.000 | 7a82c191b179 | source-tree-v1:5ced8470d96a | device-profile-v1:27e1acef5b70 | `release/evidence/benchmarks/wgpu-workload-14-compound-fused-filter.json` |
| `cuda.ptx.patterns.release.corpus` | cuda | 0.252 | 1.000 | 56d01f8fe77b | source-tree-v1:f42685f0dc36 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/cuda-ptx-patterns.json` |
| `foundation.optimizer.impact` | cuda | 15.350 | 5.561 | 7a82c191b179 | source-tree-v1:5ced8470d96a | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-11-semantic-optimizer-impact.json` |
| `foundation.optimizer.impact` | wgpu | 2.760 | 1.000 | 7a82c191b179 | source-tree-v1:5ced8470d96a | device-profile-v1:27e1acef5b70 | `release/evidence/benchmarks/wgpu-workload-11-semantic-optimizer-impact.json` |
| `foundation.reduce.sum.crossover` | cuda | 1.512 | 1.058 | 7a82c191b179 | source-tree-v1:5ced8470d96a | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-15-adaptive-routing.json` |
| `foundation.reduce.sum.crossover` | wgpu | 1.429 | 1.000 | 7a82c191b179 | source-tree-v1:5ced8470d96a | device-profile-v1:27e1acef5b70 | `release/evidence/benchmarks/wgpu-workload-15-adaptive-routing.json` |
| `metadata.condition.filesize_header.1m` | cuda | 0.090 | 1.000 | 7a82c191b179 | source-tree-v1:5ced8470d96a | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-04-metadata-conditions.json` |
| `metadata.condition.filesize_header.1m` | wgpu | 0.103 | 1.148 | 7a82c191b179 | source-tree-v1:5ced8470d96a | device-profile-v1:27e1acef5b70 | `release/evidence/benchmarks/wgpu-workload-04-metadata-conditions.json` |
| `nn.linear_4bit_affine_grouped.1m` | cuda | 0.007 | 1.000 | 7a82c191b179 | source-tree-v1:5ced8470d96a | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-16-quantized-linear.json` |
| `nn.linear_4bit_affine_grouped.1m` | wgpu | 0.335 | 49.684 | 764d17a039f3 | source-tree-v1:ab696e1a2bcc | device-profile-v1:596d0bef41ed | `release/evidence/benchmarks/wgpu-workload-16-quantized-linear.json` |
| `release.alias_reaching_def.1m` | cuda | 0.192 | 1.000 | 7a82c191b179 | source-tree-v1:5ced8470d96a | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-07-alias-reaching-def.json` |
| `release.alias_reaching_def.1m` | wgpu | 0.313 | 1.629 | 7a82c191b179 | source-tree-v1:5ced8470d96a | device-profile-v1:27e1acef5b70 | `release/evidence/benchmarks/wgpu-workload-07-alias-reaching-def.json` |
| `release.ast_motif_traversal.1m` | cuda | 0.178 | 1.000 | 7a82c191b179 | source-tree-v1:5ced8470d96a | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-09-ast-motif-traversal.json` |
| `release.ast_motif_traversal.1m` | wgpu | 0.341 | 1.914 | 7a82c191b179 | source-tree-v1:5ced8470d96a | device-profile-v1:27e1acef5b70 | `release/evidence/benchmarks/wgpu-workload-09-ast-motif-traversal.json` |
| `release.condition_eval.1m` | cuda | 0.205 | 1.000 | 7a82c191b179 | source-tree-v1:5ced8470d96a | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-01-condition-eval.json` |
| `release.condition_eval.1m` | wgpu | 0.533 | 2.597 | 7a82c191b179 | source-tree-v1:5ced8470d96a | device-profile-v1:27e1acef5b70 | `release/evidence/benchmarks/wgpu-workload-01-condition-eval.json` |
| `release.egraph_saturation.1m` | cuda | 0.226 | 1.000 | 7a82c191b179 | source-tree-v1:5ced8470d96a | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-17-egraph-saturation.json` |
| `release.egraph_saturation.1m` | wgpu | 0.272 | 1.205 | 7a82c191b179 | source-tree-v1:5ced8470d96a | device-profile-v1:27e1acef5b70 | `release/evidence/benchmarks/wgpu-workload-17-egraph-saturation.json` |
| `release.entropy_window.1m` | cuda | 0.121 | 1.000 | 7a82c191b179 | source-tree-v1:5ced8470d96a | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-05-entropy-window.json` |
| `release.entropy_window.1m` | wgpu | 0.327 | 2.712 | 7a82c191b179 | source-tree-v1:5ced8470d96a | device-profile-v1:27e1acef5b70 | `release/evidence/benchmarks/wgpu-workload-05-entropy-window.json` |
| `release.ifds_witness.1m` | cuda | 0.095 | 1.000 | 7a82c191b179 | source-tree-v1:5ced8470d96a | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-08-ifds-witness.json` |
| `release.ifds_witness.1m` | wgpu | 0.220 | 2.330 | 7a82c191b179 | source-tree-v1:5ced8470d96a | device-profile-v1:27e1acef5b70 | `release/evidence/benchmarks/wgpu-workload-08-ifds-witness.json` |
| `release.megakernel_queue.1m` | cuda | 0.265 | 1.000 | 7a82c191b179 | source-tree-v1:5ced8470d96a | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-10-megakernel-queued-batches.json` |
| `release.megakernel_queue.1m` | wgpu | 0.282 | 1.065 | 7a82c191b179 | source-tree-v1:5ced8470d96a | device-profile-v1:27e1acef5b70 | `release/evidence/benchmarks/wgpu-workload-10-megakernel-queued-batches.json` |
| `release.offset_count_aggregation.1m` | cuda | 0.209 | 1.000 | 7a82c191b179 | source-tree-v1:5ced8470d96a | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-03-offset-count-aggregation.json` |
| `release.offset_count_aggregation.1m` | wgpu | 0.378 | 1.809 | 7a82c191b179 | source-tree-v1:5ced8470d96a | device-profile-v1:27e1acef5b70 | `release/evidence/benchmarks/wgpu-workload-03-offset-count-aggregation.json` |
| `release.quantified_condition_loops.1m` | cuda | 0.096 | 1.000 | 7a82c191b179 | source-tree-v1:5ced8470d96a | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-06-quantified-condition-loops.json` |
| `release.quantified_condition_loops.1m` | wgpu | 0.409 | 4.244 | 7a82c191b179 | source-tree-v1:5ced8470d96a | device-profile-v1:27e1acef5b70 | `release/evidence/benchmarks/wgpu-workload-06-quantified-condition-loops.json` |
| `release.string_bitmap_scatter.1m` | cuda | 0.025 | 1.000 | 7a82c191b179 | source-tree-v1:5ced8470d96a | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-02-string-bitmap-scatter.json` |
| `release.string_bitmap_scatter.1m` | wgpu | 0.025 | 1.029 | 7a82c191b179 | source-tree-v1:5ced8470d96a | device-profile-v1:27e1acef5b70 | `release/evidence/benchmarks/wgpu-workload-02-string-bitmap-scatter.json` |
| `sparse.compaction.count.1m` | cuda | 0.046 | 1.000 | 7a82c191b179 | source-tree-v1:5ced8470d96a | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-12-sparse-output-compaction.json` |
| `sparse.compaction.count.1m` | wgpu | 0.086 | 1.861 | 7a82c191b179 | source-tree-v1:5ced8470d96a | device-profile-v1:27e1acef5b70 | `release/evidence/benchmarks/wgpu-workload-12-sparse-output-compaction.json` |

## declared without a measurement

Every backend a case contract declares carries a measurement.
