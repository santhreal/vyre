# cross-backend comparison

Produced by `./cargo_full run --bin xtask -- bench-crossback --write`. Every row is a
wall-clock reading a release benchmark suite recorded under
`release/evidence/benchmarks/`, with the commit, source-tree fingerprint and
device signature it was taken under. `ratio` is the case wall time over the
fastest backend measured for that case.

| case | backend | ms | ratio | commit | source tree | device | artifact |
|------|---------|----|-------|--------|-------------|--------|----------|
| `callgraph.reachability.step.262k` | cuda | 0.068 | 1.000 | 56d01f8fe77b | source-tree-v1:f42685f0dc36 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-13-callgraph-reachability.json` |
| `callgraph.reachability.step.262k` | wgpu | 1.702 | 25.080 | 764d17a039f3 | source-tree-v1:ab696e1a2bcc | device-profile-v1:596d0bef41ed | `release/evidence/benchmarks/wgpu-workload-13-callgraph-reachability.json` |
| `compound.pipeline.fused_filter.1m` | cuda | 0.636 | 1.086 | 56d01f8fe77b | source-tree-v1:f42685f0dc36 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-14-compound-fused-filter.json` |
| `compound.pipeline.fused_filter.1m` | wgpu | 0.585 | 1.000 | 764d17a039f3 | source-tree-v1:ab696e1a2bcc | device-profile-v1:596d0bef41ed | `release/evidence/benchmarks/wgpu-workload-14-compound-fused-filter.json` |
| `cuda.ptx.patterns.release.corpus` | cuda | 0.252 | 1.000 | 56d01f8fe77b | source-tree-v1:f42685f0dc36 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/cuda-ptx-patterns.json` |
| `foundation.optimizer.impact` | cuda | 5.516 | 1.000 | af0cdacc30b9 | source-tree-v1:4e7226af4a99 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-11-semantic-optimizer-impact.json` |
| `foundation.reduce.sum.crossover` | cuda | 0.685 | 1.000 | 56d01f8fe77b | source-tree-v1:f42685f0dc36 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-15-adaptive-routing.json` |
| `metadata.condition.filesize_header.1m` | cuda | 0.061 | 1.000 | 56d01f8fe77b | source-tree-v1:f42685f0dc36 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-04-metadata-conditions.json` |
| `metadata.condition.filesize_header.1m` | wgpu | 0.213 | 3.488 | 764d17a039f3 | source-tree-v1:ab696e1a2bcc | device-profile-v1:596d0bef41ed | `release/evidence/benchmarks/wgpu-workload-04-metadata-conditions.json` |
| `nn.linear_4bit_affine_grouped.1m` | cuda | 0.010 | 1.000 | a14f1b979f30 | source-tree-v1:e351693246ba | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-16-quantized-linear.json` |
| `nn.linear_4bit_affine_grouped.1m` | wgpu | 0.335 | 35.076 | 764d17a039f3 | source-tree-v1:ab696e1a2bcc | device-profile-v1:596d0bef41ed | `release/evidence/benchmarks/wgpu-workload-16-quantized-linear.json` |
| `release.alias_reaching_def.1m` | cuda | 0.326 | 1.000 | 56d01f8fe77b | source-tree-v1:f42685f0dc36 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-07-alias-reaching-def.json` |
| `release.alias_reaching_def.1m` | wgpu | 1.139 | 3.497 | 59a49f86983c | source-tree-v1:562590e78936 | device-profile-v1:596d0bef41ed | `release/evidence/benchmarks/wgpu-workload-07-alias-reaching-def.json` |
| `release.c_ast_traversal.1m` | cuda | 0.278 | 1.000 | 56d01f8fe77b | source-tree-v1:f42685f0dc36 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-09-c-ast-traversal.json` |
| `release.c_ast_traversal.1m` | wgpu | 0.964 | 3.465 | 764d17a039f3 | source-tree-v1:ab696e1a2bcc | device-profile-v1:596d0bef41ed | `release/evidence/benchmarks/wgpu-workload-09-c-ast-traversal.json` |
| `release.condition_eval.1m` | cuda | 0.281 | 1.000 | 56d01f8fe77b | source-tree-v1:f42685f0dc36 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-01-condition-eval.json` |
| `release.condition_eval.1m` | wgpu | 0.819 | 2.921 | 764d17a039f3 | source-tree-v1:ab696e1a2bcc | device-profile-v1:596d0bef41ed | `release/evidence/benchmarks/wgpu-workload-01-condition-eval.json` |
| `release.egraph_saturation.1m` | cuda | 0.310 | 1.000 | 56d01f8fe77b | source-tree-v1:f42685f0dc36 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-11-egraph-saturation.json` |
| `release.egraph_saturation.1m` | wgpu | 0.854 | 2.750 | 764d17a039f3 | source-tree-v1:ab696e1a2bcc | device-profile-v1:596d0bef41ed | `release/evidence/benchmarks/wgpu-workload-11-egraph-saturation.json` |
| `release.entropy_window.1m` | cuda | 0.271 | 1.000 | 56d01f8fe77b | source-tree-v1:f42685f0dc36 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-05-entropy-window.json` |
| `release.entropy_window.1m` | wgpu | 1.390 | 5.138 | 764d17a039f3 | source-tree-v1:ab696e1a2bcc | device-profile-v1:596d0bef41ed | `release/evidence/benchmarks/wgpu-workload-05-entropy-window.json` |
| `release.ifds_witness.1m` | cuda | 0.272 | 1.000 | 56d01f8fe77b | source-tree-v1:f42685f0dc36 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-08-ifds-witness.json` |
| `release.ifds_witness.1m` | wgpu | 1.092 | 4.023 | 59a49f86983c | source-tree-v1:562590e78936 | device-profile-v1:596d0bef41ed | `release/evidence/benchmarks/wgpu-workload-08-ifds-witness.json` |
| `release.megakernel_queue.1m` | cuda | 0.282 | 1.000 | 56d01f8fe77b | source-tree-v1:f42685f0dc36 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-10-megakernel-queued-batches.json` |
| `release.megakernel_queue.1m` | wgpu | 0.860 | 3.048 | 764d17a039f3 | source-tree-v1:ab696e1a2bcc | device-profile-v1:596d0bef41ed | `release/evidence/benchmarks/wgpu-workload-10-megakernel-queued-batches.json` |
| `release.offset_count_aggregation.1m` | cuda | 0.382 | 1.000 | 56d01f8fe77b | source-tree-v1:f42685f0dc36 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-03-offset-count-aggregation.json` |
| `release.offset_count_aggregation.1m` | wgpu | 0.844 | 2.210 | 764d17a039f3 | source-tree-v1:ab696e1a2bcc | device-profile-v1:596d0bef41ed | `release/evidence/benchmarks/wgpu-workload-03-offset-count-aggregation.json` |
| `release.quantified_condition_loops.1m` | cuda | 0.287 | 1.000 | 56d01f8fe77b | source-tree-v1:f42685f0dc36 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-06-quantified-condition-loops.json` |
| `release.quantified_condition_loops.1m` | wgpu | 1.323 | 4.603 | 764d17a039f3 | source-tree-v1:ab696e1a2bcc | device-profile-v1:596d0bef41ed | `release/evidence/benchmarks/wgpu-workload-06-quantified-condition-loops.json` |
| `release.string_bitmap_scatter.1m` | cuda | 0.038 | 1.000 | a14f1b979f30 | source-tree-v1:e351693246ba | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-02-string-bitmap-scatter.json` |
| `release.string_bitmap_scatter.1m` | wgpu | 0.388 | 10.193 | 764d17a039f3 | source-tree-v1:ab696e1a2bcc | device-profile-v1:596d0bef41ed | `release/evidence/benchmarks/wgpu-workload-02-string-bitmap-scatter.json` |
| `runtime.adaptive_routing.gpu_resident.1m` | wgpu | 0.573 | 1.000 | 764d17a039f3 | source-tree-v1:ab696e1a2bcc | device-profile-v1:596d0bef41ed | `release/evidence/benchmarks/wgpu-workload-15-adaptive-routing.json` |
| `sparse.compaction.count.1m` | cuda | 0.058 | 1.000 | 56d01f8fe77b | source-tree-v1:f42685f0dc36 | device-profile-v1:d0842449f73e | `release/evidence/benchmarks/workload-12-sparse-output-compaction.json` |
| `sparse.compaction.count.1m` | wgpu | 1.568 | 27.146 | 764d17a039f3 | source-tree-v1:ab696e1a2bcc | device-profile-v1:596d0bef41ed | `release/evidence/benchmarks/wgpu-workload-12-sparse-output-compaction.json` |

## declared without a measurement

| case | backend | declared by |
|------|---------|-------------|
| `foundation.reduce.sum.crossover` | wgpu | `release/evidence/benchmarks/workload-15-adaptive-routing.json` |
| `runtime.adaptive_routing.gpu_resident.1m` | cuda | `release/evidence/benchmarks/wgpu-workload-15-adaptive-routing.json` |
