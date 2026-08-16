# Criterion baselines

Every crate with a bench target has a section below. The `bench-baselines` gate
reads each member manifest for a bench target and fails when one of those
packages has no section.

Each number is the criterion median of the run named under the section, on the
machine recorded here. Recapture a section by rerunning its command and
replacing the table; do not edit a number without rerunning.

machine: desktop workstation, Linux 6.17.0-19-generic, 92 GiB RAM
cpu: AMD Ryzen 9 9950X, 16 cores / 32 threads
gpu: NVIDIA GeForce RTX 5090, 32607 MiB, driver 580.178.04
rustc: 1.86.0 (05f9846f8 2025-03-31)
commit: dfe6ac654c1876b1ea262dee2a1f73d23a20da31
date: 2026-08-14

### vyre-primitives

`cargo bench -p vyre-primitives --bench wire_throughput`

Wire pack, `wire` is `pack_u32_slice`, `naive_flat_map` is per-word `to_le_bytes`.

| words | wire | naive_flat_map |
| --- | --- | --- |
| 256 | 11.784 ns | 13.418 ns |
| 262144 | 14.303 us | 14.215 us |
| 26214400 | 65.460 ms | 63.738 ms |

`pack_u32_slice_into` over 262144 words, reusing caller storage: 14.055 us.

Wire unpack, `wire` is `unpack_u32_slice_into`, `wire_decode_all` is
`decode_u32_le_bytes_all`, `naive_chunks_exact` is per-word `from_le_bytes`.

| words | wire | wire_decode_all | naive_chunks_exact |
| --- | --- | --- | --- |
| 256 | 8.2043 ns | 12.875 ns | 10.194 ns |
| 262144 | 14.638 us | 14.971 us | 14.736 us |
| 26214400 | 51.661 ms | 80.850 ms | 78.990 ms |

The pack fast path wins at 1 KiB and ties the naive path from 1 MiB up, where
both are bandwidth bound. The unpack fast path wins at every size and is 1.53x
the naive path at 100 MiB.

### vyre-bench

`cargo bench -p vyre-bench --bench release`

| bench | median |
| --- | --- |
| cold_setup_registry_inventory_collection | 1.7143 us |
| primitive_cpu_ref/bitset_and/32 | 16.903 ns |
| primitive_cpu_ref/bitset_and/256 | 25.377 ns |
| primitive_cpu_ref/bitset_and/2048 | 150.40 ns |
| primitive_cpu_ref/bitset_and/16384 | 1.1775 us |
| primitive_cpu_ref/bitset_and/131072 | 11.726 us |
| primitive_cpu_ref/bitset_and_into/32 | 5.0740 ns |
| primitive_cpu_ref/bitset_and_into/256 | 18.880 ns |
| primitive_cpu_ref/bitset_and_into/2048 | 147.18 ns |
| primitive_cpu_ref/bitset_and_into/16384 | 1.1607 us |
| primitive_cpu_ref/bitset_and_into/131072 | 10.972 us |
| primitive_cpu_ref/dominator_tree/linear_chain/1000 | 97.007 us |
| primitive_cpu_ref/dominator_tree/linear_chain/10000 | 1.2656 ms |
| primitive_cpu_ref/dominator_tree/linear_chain/100000 | 46.148 ms |
| primitive_cpu_ref/dominator_tree/linear_chain/1000000 | 621.37 ms |
| primitive_cpu_ref/dominator_tree/fanout_tree/1000 | 80.446 us |
| primitive_cpu_ref/dominator_tree/fanout_tree/10000 | 767.94 us |
| primitive_cpu_ref/dominator_tree/fanout_tree/100000 | 26.941 ms |
| primitive_cpu_ref/dominator_tree/fanout_tree/1000000 | 547.06 ms |
| primitive_program_build/dominator_tree/program/1000 | 3.3140 us |
| primitive_program_build/dominator_tree/program/10000 | 3.1870 us |
| primitive_program_build/dominator_tree/program/100000 | 3.1843 us |
| primitive_program_build/dominator_tree/program/1000000 | 3.2041 us |
| primitive_program_build/dominator_tree/vram_bytes/1000 | 223.89 ps |
| primitive_program_build/dominator_tree/vram_bytes/10000 | 218.50 ps |
| primitive_program_build/dominator_tree/vram_bytes/100000 | 226.78 ps |
| primitive_program_build/dominator_tree/vram_bytes/1000000 | 230.36 ps |
| compiler_grade_release/program_build/macro/release.condition_eval.1m | 7.2659 us |
| compiler_grade_release/program_build/macro/release.string_bitmap_scatter.1m | 4.6778 us |
| compiler_grade_release/program_build/macro/release.offset_count_aggregation.1m | 7.2920 us |
| compiler_grade_release/program_build/macro/release.entropy_window.1m | 7.2866 us |
| compiler_grade_release/program_build/macro/release.quantified_condition_loops.1m | 8.7927 us |
| compiler_grade_release/program_build/macro/release.alias_reaching_def.1m | 7.3963 us |
| compiler_grade_release/program_build/macro/release.ifds_witness.1m | 7.4797 us |
| compiler_grade_release/program_build/macro/release.c_ast_traversal.1m | 7.2086 us |
| compiler_grade_release/program_build/macro/release.megakernel_queue.1m | 8.1397 us |
| compiler_grade_release/program_build/macro/release.egraph_saturation.1m | 7.3335 us |
| runtime_io/nvme_gpu_ingest_telemetry/registered_mapped_read/4294967296 | 197.67 ns |
| runtime_io/nvme_gpu_ingest_telemetry/gpudirect_nvme_passthrough/68719476736 | 187.88 ns |

`program_build` is host-side IR construction, not device execution. Device
timings for the same macro cases are published as release evidence under
`release/evidence/benchmarks/`. The `runtime_io` rows time telemetry record
construction for the stated byte counts, not the transfer.

### vyre-foundation

`cargo bench -p vyre-foundation --bench optimizer_pipeline`

Measured at commit 4cb6a3a37d5db12159dd52ca6103d122c959ea47 on the machine in
the header. `optimize()` is the host IR pipeline every compile runs before any
backend lowering, so these are compile latency, not device time.

| bench | median |
| --- | --- |
| optimizer/pipeline/release_corpus_families | 240.21 us |
| optimizer/pipeline/kernel_wide/16 | 242.40 us |
| optimizer/pipeline/kernel_wide/64 | 886.14 us |
| optimizer/pipeline/kernel_loop_nest/4x8 | 70.902 us |

`release_corpus_families` optimizes one program per semantic family of the
shipped release corpus per iteration. The two `kernel_wide` rows scale straight
line arithmetic depth, so the difference between them is rewrite work rather
than per-pass fixed cost: 4x the depth costs 3.65x the time.
