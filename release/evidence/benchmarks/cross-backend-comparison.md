# cross-backend comparison

Produced by `./cargo_full run --bin xtask -- bench-crossback --write`. Every row is a
wall-clock reading a release benchmark suite recorded under
`release/evidence/benchmarks/`, with the commit, source-tree fingerprint and
device signature it was taken under. `ratio` is the case wall time over the
fastest backend measured for that case.

| case | backend | ms | ratio | commit | source tree | device | artifact |
|------|---------|----|-------|--------|-------------|--------|----------|
| `callgraph.reachability.step.262k` | cuda | 0.048 | 1.000 | 6e8ce09485b3 | source-tree-v1:2727ecd1e1c8 | device-profile-v1:f04bd6c0b35f | `release/evidence/benchmarks/workload-13-callgraph-reachability.json` |
| `callgraph.reachability.step.262k` | wgpu | 0.123 | 2.591 | 6e8ce09485b3 | source-tree-v1:2727ecd1e1c8 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-13-callgraph-reachability.json` |
| `compound.pipeline.fused_filter.1m` | cuda | 1.424 | 1.102 | 6e8ce09485b3 | source-tree-v1:2727ecd1e1c8 | device-profile-v1:f04bd6c0b35f | `release/evidence/benchmarks/workload-14-compound-fused-filter.json` |
| `compound.pipeline.fused_filter.1m` | wgpu | 1.292 | 1.000 | 6e8ce09485b3 | source-tree-v1:2727ecd1e1c8 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-14-compound-fused-filter.json` |
| `cuda.ptx.patterns.release.corpus` | cuda | 0.055 | 1.000 | 6e8ce09485b3 | source-tree-v1:2727ecd1e1c8 | device-profile-v1:f04bd6c0b35f | `release/evidence/benchmarks/cuda-ptx-patterns.json` |
| `foundation.optimizer.impact` | cuda | 1.279 | 1.146 | 6e8ce09485b3 | source-tree-v1:2727ecd1e1c8 | device-profile-v1:f04bd6c0b35f | `release/evidence/benchmarks/workload-11-semantic-optimizer-impact.json` |
| `foundation.optimizer.impact` | wgpu | 1.116 | 1.000 | 6e8ce09485b3 | source-tree-v1:2727ecd1e1c8 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-11-semantic-optimizer-impact.json` |
| `foundation.reduce.sum.crossover` | cuda | 0.487 | 1.469 | 6e8ce09485b3 | source-tree-v1:2727ecd1e1c8 | device-profile-v1:f04bd6c0b35f | `release/evidence/benchmarks/workload-15-adaptive-routing.json` |
| `foundation.reduce.sum.crossover` | wgpu | 0.331 | 1.000 | 6e8ce09485b3 | source-tree-v1:2727ecd1e1c8 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-15-adaptive-routing.json` |
| `metadata.condition.filesize_header.1m` | cuda | 0.025 | 1.000 | 6e8ce09485b3 | source-tree-v1:2727ecd1e1c8 | device-profile-v1:f04bd6c0b35f | `release/evidence/benchmarks/workload-04-metadata-conditions.json` |
| `metadata.condition.filesize_header.1m` | wgpu | 0.107 | 4.204 | 6e8ce09485b3 | source-tree-v1:2727ecd1e1c8 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-04-metadata-conditions.json` |
| `nn.linear_4bit_affine_grouped.1m` | cuda | 0.009 | 1.037 | 6e8ce09485b3 | source-tree-v1:2727ecd1e1c8 | device-profile-v1:f04bd6c0b35f | `release/evidence/benchmarks/workload-16-quantized-linear.json` |
| `nn.linear_4bit_affine_grouped.1m` | wgpu | 0.009 | 1.000 | 6e8ce09485b3 | source-tree-v1:2727ecd1e1c8 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-16-quantized-linear.json` |
| `release.alias_reaching_def.1m` | cuda | 0.027 | 1.000 | 6e8ce09485b3 | source-tree-v1:2727ecd1e1c8 | device-profile-v1:f04bd6c0b35f | `release/evidence/benchmarks/workload-07-alias-reaching-def.json` |
| `release.alias_reaching_def.1m` | wgpu | 0.117 | 4.363 | 6e8ce09485b3 | source-tree-v1:2727ecd1e1c8 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-07-alias-reaching-def.json` |
| `release.ast_motif_traversal.1m` | cuda | 0.027 | 1.000 | 6e8ce09485b3 | source-tree-v1:2727ecd1e1c8 | device-profile-v1:f04bd6c0b35f | `release/evidence/benchmarks/workload-09-ast-motif-traversal.json` |
| `release.ast_motif_traversal.1m` | wgpu | 0.116 | 4.295 | 6e8ce09485b3 | source-tree-v1:2727ecd1e1c8 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-09-ast-motif-traversal.json` |
| `release.condition_eval.1m` | cuda | 0.027 | 1.000 | 6e8ce09485b3 | source-tree-v1:2727ecd1e1c8 | device-profile-v1:f04bd6c0b35f | `release/evidence/benchmarks/workload-01-condition-eval.json` |
| `release.condition_eval.1m` | wgpu | 0.127 | 4.792 | 6e8ce09485b3 | source-tree-v1:2727ecd1e1c8 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-01-condition-eval.json` |
| `release.egraph_saturation.1m` | cuda | 0.028 | 1.000 | 6e8ce09485b3 | source-tree-v1:2727ecd1e1c8 | device-profile-v1:f04bd6c0b35f | `release/evidence/benchmarks/workload-17-egraph-saturation.json` |
| `release.egraph_saturation.1m` | wgpu | 0.118 | 4.163 | 6e8ce09485b3 | source-tree-v1:2727ecd1e1c8 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-17-egraph-saturation.json` |
| `release.entropy_window.1m` | cuda | 0.027 | 1.000 | 6e8ce09485b3 | source-tree-v1:2727ecd1e1c8 | device-profile-v1:f04bd6c0b35f | `release/evidence/benchmarks/workload-05-entropy-window.json` |
| `release.entropy_window.1m` | wgpu | 0.113 | 4.197 | 6e8ce09485b3 | source-tree-v1:2727ecd1e1c8 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-05-entropy-window.json` |
| `release.ifds_witness.1m` | cuda | 0.027 | 1.000 | 6e8ce09485b3 | source-tree-v1:2727ecd1e1c8 | device-profile-v1:f04bd6c0b35f | `release/evidence/benchmarks/workload-08-ifds-witness.json` |
| `release.ifds_witness.1m` | wgpu | 0.131 | 4.781 | 6e8ce09485b3 | source-tree-v1:2727ecd1e1c8 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-08-ifds-witness.json` |
| `release.megakernel_queue.1m` | cuda | 0.028 | 1.000 | 6e8ce09485b3 | source-tree-v1:2727ecd1e1c8 | device-profile-v1:f04bd6c0b35f | `release/evidence/benchmarks/workload-10-megakernel-queued-batches.json` |
| `release.megakernel_queue.1m` | wgpu | 0.113 | 4.104 | 6e8ce09485b3 | source-tree-v1:2727ecd1e1c8 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-10-megakernel-queued-batches.json` |
| `release.offset_count_aggregation.1m` | cuda | 0.027 | 1.000 | 6e8ce09485b3 | source-tree-v1:2727ecd1e1c8 | device-profile-v1:f04bd6c0b35f | `release/evidence/benchmarks/workload-03-offset-count-aggregation.json` |
| `release.offset_count_aggregation.1m` | wgpu | 0.111 | 4.038 | 6e8ce09485b3 | source-tree-v1:2727ecd1e1c8 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-03-offset-count-aggregation.json` |
| `release.optimizer.resident_pipeline` | cuda | 549.992 | 1.000 | 6e8ce09485b3 | source-tree-v1:2727ecd1e1c8 | device-profile-v1:f04bd6c0b35f | `release/evidence/benchmarks/resident-optimizer-pipeline.json` |
| `release.quantified_condition_loops.1m` | cuda | 0.027 | 1.000 | 6e8ce09485b3 | source-tree-v1:2727ecd1e1c8 | device-profile-v1:f04bd6c0b35f | `release/evidence/benchmarks/workload-06-quantified-condition-loops.json` |
| `release.quantified_condition_loops.1m` | wgpu | 0.114 | 4.136 | 6e8ce09485b3 | source-tree-v1:2727ecd1e1c8 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-06-quantified-condition-loops.json` |
| `release.string_bitmap_scatter.1m` | cuda | 0.061 | 1.000 | 6e8ce09485b3 | source-tree-v1:2727ecd1e1c8 | device-profile-v1:f04bd6c0b35f | `release/evidence/benchmarks/workload-02-string-bitmap-scatter.json` |
| `release.string_bitmap_scatter.1m` | wgpu | 0.133 | 2.170 | 6e8ce09485b3 | source-tree-v1:2727ecd1e1c8 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-02-string-bitmap-scatter.json` |
| `sparse.compaction.count.1m` | cuda | 0.019 | 1.000 | 6e8ce09485b3 | source-tree-v1:2727ecd1e1c8 | device-profile-v1:f04bd6c0b35f | `release/evidence/benchmarks/workload-12-sparse-output-compaction.json` |
| `sparse.compaction.count.1m` | wgpu | 0.115 | 6.105 | 6e8ce09485b3 | source-tree-v1:2727ecd1e1c8 | device-profile-v1:7e14ee791134 | `release/evidence/benchmarks/wgpu-workload-12-sparse-output-compaction.json` |

## declared without a measurement

| case | backend | declared by |
|------|---------|-------------|
| `release.optimizer.resident_pipeline` | wgpu | `release/evidence/benchmarks/resident-optimizer-pipeline.json` |
