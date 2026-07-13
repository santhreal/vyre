# Changelog

All notable changes to vyre are documented here. Follows Keep a Changelog.

## [Unreleased]

## [0.6.5]  -  2026-07-13

### Added: C-frontend visible-type precompute wiring (`vyre-frontend-c`, `vyre-libs`)

- Complete the visible-type precompute path so the precomputed-context typedef annotator no longer drops the ordinary declarator flag for `T x;` where `T` is a typedef-name. `c11_precompute_vast_visible_type` resolves the per-node visible-typedef-name table once (after the decl-context table settles) and the annotate pass reads the bit; `c11_annotate_typedef_names_precomputed_context[_packed_haystack]` now take the table as a ReadOnly buffer at binding 3. The vyre-frontend-c pipeline gained a `vast_pg/visible_type.rs` stage (stage-pipeline cached) that feeds both the fused and unfused annotate dispatches, failing closed if the table is absent on the non-global path.

### Added: IR-parity + regression coverage sweep (`vyre-primitives`, `vyre-self-substrate`, `vyre-foundation`, `vyre-libs`)

- Add and extend reference_eval GPU-IR-vs-cpu_ref parity proptests and regression tests across graph/nfa/math/decode primitives, including signed fixed-point negative-intermediate coverage and sharding-decomposition property gates. Boundary anchors assert real values, not shape.

### Changed: signed fixed-point correctness + ONE-PLACE dedup (`vyre-primitives`, `vyre-libs`)

- Route weighted-Jacobi / AMG divides through `fixed_sdiv_by_positive_expr` so negative 16.16 intermediates no longer corrupt (validated by the new parity tests). Replace inline masked 256-table lookups with the canonical `crate::ir_safe::byte_table_lookup`. Add `dfa_compile_case_insensitive[_with_budget]`. New dev-only `vyre-test-support` crate holding the canonical registry/coverage closure gate.

### Added: interpreter op-counting + roofline operating point (`vyre-reference`, `vyre-bench`)

- Added `vyre_reference::count_ops`: a thread-local scope that counts the arithmetic IR operations (`BinOp`/`UnOp`/`Fma`) the reference interpreter executes during a closure, a backend-agnostic dynamic operation count for roofline / complexity analysis. Because the interpreter runs the same vyre IR with the same data-dependent control flow any backend does, its count for a `(program, inputs)` pair equals the GPU's dynamic IR-op count for those inputs (at vyre-IR granularity, coarser than hardware SASS). Counting is opt-in, a no-op thread-local read outside a `count_ops` scope, so ordinary interpreter use (all in tests) is unaffected (vyre-reference and vyre-primitives suites green). This closes the last non-root piece of the W3-6 roofline: the new `scan_roofline_operating_point_cuda` test measures the literal scan's operational intensity via `count_ops` on the CPU reference backend and its achieved bandwidth on the RTX 5090, placing the operating point on the roofline, intensity 13.77 IR-ops/byte, left of the 29.23 ops/byte ridge (memory-bound side confirmed), achieved compute ≈3.2 T-IR-ops/s under the 52-TOPS ceiling. The full roofline, both ceilings, ridge, both measured axes, and the bound verdict, is now complete and honest without root. A finer SASS-granularity count (`sm__inst_executed`) via Nsight-Compute would only refine the granularity and remains the optional root-gated step.

### Added: property gates for sharding decompositions (`vyre-primitives`, `vyre-libs`)

- Added 10k-case property tests hardening the two sharding decompositions shipped this cycle (Testing Contract: proptest per feature). `proptest_csr_frontier_shard` (vyre-primitives, 3×10k cases) proves the graph-frontier device-sharding invariant over random graphs, frontiers, and shard counts: sharded expansion always equals single-device expansion, the vertex partition is always disjoint+complete, and the OR-merge is order-independent and round-trips the frontier. `shard_assignment_is_a_valid_total_partition` (vyre-libs, 4k cases) proves the scan-sharding load balancer always produces a valid total partition for any window sizes / shard count / weights, one shard per window, all in range, byte-work conserved (nothing dropped or double-counted), and exact round-robin unweighted. These are the invariants the parallel sharded scan + graph frontier rely on to stay byte-identical to single-device regardless of work distribution.

### Added: device-sharded graph frontier expansion (`vyre-primitives`)

- Added `vyre_primitives::graph::csr_frontier_shard`: the W3-5 `graph-frontier-device-shards` decomposition. A forward `csr_frontier_step` expands only the vertices set in `frontier_in`, so the active frontier can be partitioned across device shards by vertex ownership (`partition_frontier_by_vertex`: disjoint, complete, contiguous vertex ranges) and the per-shard `frontier_out` bitsets OR-merged back together (`merge_frontier_out`: the cross-shard visited/frontier merge, a peer-transfer reduce on real multi-GPU, a host OR here). `frontier_step_sharded` runs one sharded expansion level given a per-shard `expand` closure (each shard dispatched on its own device), and fails closed on a zero shard count, a mis-sized frontier, or a wrong-sized shard output. Because per-vertex expansions are independent and the partition is disjoint and complete, the merged result equals a single-device expansion exactly, proven three ways: a hand oracle over a graph with cross-shard edges, a pure-Rust expansion oracle across 1–5 shard counts, and (the load-bearing proof) the real `csr_frontier_step` GPU program driven through the reference interpreter across 1–4 shards versus the single-device run. Device sharding therefore changes no reachability bit. Per-device concurrent dispatch reuses the `std::thread::scope` pattern already proven for byte-range scan sharding; only wall-clock multi-GPU speedup and the on-device peer-transfer merge need a second physical GPU.

### Added: roofline COMPUTE ceiling + full model (`vyre-driver-cuda`, `vyre-bench`)

- Added `CudaDeviceCaps::peak_compute_ops_per_sec()`: the compute ceiling of the W3-6 roofline, alongside the existing `memory_bandwidth_gbps()` memory ceiling. It is `SM_count × 4 warp-schedulers × warp_size × core_clock`, backed by a new `core_clock_rate_khz` device attribute (`CU_DEVICE_ATTRIBUTE_CLOCK_RATE`) joining the existing memory clock. The "4 warp schedulers per SM" factor is a universal published NVIDIA architectural constant (every SM from Volta through Blackwell is four processing sub-partitions, each issuing one warp-wide instruction per cycle), not a fabricated per-generation cores-per-SM table, so the ceiling is an honest analytical value with no invented device numbers. With both ceilings the roofline now has a **ridge point** (operational intensity where memory-bound flips to compute-bound). The new `scan_roofline_model_cuda` test assembles the full model on the RTX 5090: peak memory 1792 GB/s + peak compute ~52 TOPS + ridge 29227 ops/KiB + the scan's measured memory-axis point (218 GB/s achieved, 12% util) → the bound verdict (memory-side, launch/latency-bound, not compute-bound). A pure unit test locks the peak-compute formula (`170×4×32×2.41 GHz` ≈ 52 TOPS, asserted in the sane 40–80 TOPS Blackwell range). Only the scan's achieved *compute* operating point (executed op-count → arithmetic intensity) still needs Nsight-Compute instruction counters (admin-only here); both ceilings, the ridge, the measured memory-axis point, and the bound verdict are complete without root.

### Changed: cross-device sharded scan now dispatches in PARALLEL (`vyre-libs`, `vyre-driver-cuda`)

- `scan_sharded_core` (behind `scan_sharded_fused`/`_weighted`/`_timed`) now runs each device shard on its **own OS thread** via `std::thread::scope`, every shard prepares its own resident session and dispatches its assigned windows **concurrently** with the other devices, replacing the previous sequential shard loop. This is W3-5's "true cross-device PARALLEL dispatch (spawn per-device threads)." Aggregation stays deterministic despite the nondeterministic thread interleave: each thread globalizes into owned per-window blocks tagged with the global window index; the parent re-sorts by window index and concatenates presence in window order (byte-identical presence layout) while matches are gathered and canonically sorted by `finish_result` (order-free). It fails closed on a shard-thread panic (no partial cross-device result), each thread frees its own session before surfacing any error (one free path per thread), and `scope` guarantees all threads join so none leaks on the error path. The globalization logic is now shared between the sequential single-device paged driver and the parallel sharded core via extracted `window_presence_words` + `map_window_matches` helpers (ONE PLACE). As part of this, the CUDA resident-scan launch path (`dispatch_resident_via_borrowed_into`) now binds the device context on the calling thread (`warmup()`), it was the one resident entry point missing the bind its `batch`/`async`/`sequence` siblings already had, a latent foreign-thread `CUDA_ERROR_INVALID_CONTEXT` that per-device threading would otherwise trigger. Proven on the RTX 5090 (`parallel_sharded_dispatch_across_four_concurrent_handles_equals_single_shot_on_gpu`): a 32-file, ≥8-window corpus sharded across a four-handle set (four concurrent threads/sessions) is byte-identical to the single-device paged scan, with honest per-shard timing showing the work spread across all four shards. The existing 1-/3-device and throughput-weighted parity tests now also exercise the parallel path. True multi-GPU wall-clock speedup + peer-transfer aggregation remains gated on a second physical GPU; the parallel dispatch and deterministic aggregation are proven correct on one device.

### Added: stream-ordered `cuMemPool` device allocator (`vyre-driver-cuda`)

- Added `CudaStreamOrderedPool` (`backend/stream_ordered_pool.rs`), the stream-ordered device allocator half of W3-4. Where the synchronous bucketed `DeviceAllocationPool` recycles raw `cuMemAlloc_v2` blocks behind a host free-list (every acquire/release ordered by hand), this binds the device's **default** CUDA memory pool via `cuDeviceGetDefaultMemPool` (no private pool to create/destroy, no `Drop` hazard against context teardown) and drives it with the driver's stream-ordered allocator: `alloc_async`/`free_async` take a caller stream so an allocation and its free ride the same stream as the dispatch that consumes them, and the driver reuses a freed block for the next same-stream allocation with no host round-trip. Construction sets `RELEASE_THRESHOLD=u64::MAX` so freed physical memory stays **reserved** for reuse (the default 0 releases it on every sync, which would defeat a re-dispatch loop); `reserved_bytes()`/`used_bytes()` expose `RESERVED_MEM_CURRENT`/`USED_MEM_CURRENT`, and `trim(min_keep)` hands the reservation back to the OS. Proven on the RTX 5090 (`stream_ordered_pool_serves_usable_memory_and_reuses_reserved_blocks_on_gpu`): (a) a `memset(0xABCD1234)`→DtoH readback confirms the pool serves *usable* device memory; (b) freeing a block then re-allocating the same size leaves `reserved_bytes` **exactly unchanged**, the freed block is reused, not re-faulted; (c) `trim(0)` strictly *drops* the reservation. Hot-path integration (threading a stream through `DeviceAllocationPool::acquire`) is the tracked follow-up; this lands and proves the allocator primitive first.

### Added: roofline achieved-bandwidth evidence (`vyre-bench`)

- Added `scan_roofline_bandwidth_cuda`: the memory-bandwidth axis of the W3-6 roofline, sourced from vyre's own timing (no Nsight-Compute, which is admin-only here). A resident fused scan's achieved read bandwidth is `haystack_bytes / device_ns` (1 byte/ns == 1 GB/s), compared against the device peak from `CudaDeviceCaps::memory_bandwidth_gbps()` to place the scan on the roofline and state its bound. Measured on the RTX 5090: a 32 MiB scan runs at 235 GB/s against a 1792 GB/s peak (13% utilization → not-bandwidth-bound; this literal-set scan is latency/compute-bound with large DRAM headroom). The sanity ceiling allows for legitimate L2 over-DRAM-peak effects. An honest timing-sourced datum, explicitly not presented as Nsight counters.

### Added: scan-counter proxy capture (`vyre-bench`)

- Proved the `SCAN_COUNTER_EVIDENCE.toml` proxy counters are actually SOURCED from runtime telemetry (not just schema-declared) for the cuda backend, with a real-GPU `scan_counter_proxy_capture_cuda` test: it runs a live `GpuLiteralSet` scan through `CudaBackendRegistration` and captures `memory_bytes` (host↔device bytes), `occupancy_proxy` (the new `mean_occupancy_bps()`), `branch_divergence_proxy` (`logical_thread_waste_bps`), and `candidate_count` (match count, asserted against the planted total of 5). Measured: `memory_bytes=177612 occupancy_bps=10000 branch_divergence_bps=0 candidate_count=5`. The precise Nsight-Compute counters are admin-only on the host (`RmProfilingAdminOnly=1`), so the TOML cuda row now states that `unavailable_reason` and documents the runtime-telemetry proxy source + proving test, an honest counter source, not fabricated ncu values. The occupancy work above is what made the `occupancy_proxy` sourceable.

### Added: per-kernel occupancy evidence (`vyre-driver-cuda`)

- Every CUDA kernel launch now records its driver-measured achieved occupancy as telemetry evidence (W3-6). The launch path queries `cuOccupancyMaxActiveBlocksPerMultiprocessor` once per kernel shape and caches the result by `(function, threads_per_block)`, occupancy is constant per shape, so after the first launch it is a map lookup, never per-launch FFI (Law 7). The active-blocks count feeds a shared `occupancy_estimate_from_blocks` helper (extracted from the theoretical `estimate_occupancy` so both the register/shared-limit estimate and the driver measurement compute occupancy as the *same* fraction of `max_warps_per_sm`, ONE PLACE) and lands on `CudaTelemetrySnapshot` as `launch_occupancy_bps_sum` / `occupancy_measured_launches` / `occupancy_unmeasured_launches` with a derived `mean_occupancy_bps()` and four Prometheus series. A launch whose geometry or driver query is unusable is counted as *unmeasured* (loud), never silently dropped, so a partial mean is never mistaken for full coverage (Law 10). Occupancy recording never fails a launch (the kernel has already run). The single `cuOccupancyMaxActiveBlocksPerMultiprocessor` FFI is now behind one `query_active_blocks_per_sm_raw` helper shared with the cooperative-residency validator. Proven by telemetry unit tests (mean arithmetic, accumulate + reset) and a real-GPU `steady_state_launches_report_per_kernel_occupancy_evidence` test that runs a 256-thread dispatch loop and asserts every launch is measured, none unmeasured, and the mean is a real fraction in (0, 10000] bps consistent with the raw sum/count.

### Added: device-allocation-pool hit-rate telemetry (`vyre-driver-cuda`)

- Instrumented the transient `DeviceAllocationPool` with hit/miss counters (an acquisition served from the free-list is a hit; one that falls through to a real `cuMemAlloc_v2` is a miss) and surfaced them on `CudaTelemetrySnapshot` as `device_pool_hits`, `device_pool_misses`, and a derived `device_pool_hit_rate_bps()` (basis points, zero-safe, exact through a u128 intermediate), plus three new Prometheus series. The counters live on the pool, its only source of truth, since the caller cannot tell a hit from a miss, and are overlaid at the backend's `telemetry_snapshot()` boundary; `reset_telemetry()` resets them into the same epoch as the rest of the counters. This is the W3-4 "pool-hit-rate evidence" deliverable: a real re-dispatch consumer workload can now see whether the pool is actually serving from cache. Proven by a pure hit-rate-arithmetic unit test and a real-GPU `steady_state_redispatch_loop_reports_high_device_pool_hit_rate` test that runs a 32-dispatch identical-shape loop and asserts the steady-state hit rate is majority-hits (the pool working), with the rate exactly consistent with the raw counters.

### Added: paged corpus benchmark (`vyre-bench`)

- Added the `scan.literal_set.paged_corpus` benchmark case: it scans a multi-megabyte corpus split into thousands of small files with a window budget far smaller than the corpus (many windows) through both `scan_paged_fused` and `scan_paged_fused_async`, reporting throughput and the sync-vs-async pipeline overlap factor. Correctness is hard-gated two ways: the paged matches must equal an independent CPU `reference_scan` of the concatenated corpus, and the async result must be byte-identical to the sync result.

### Added: pattern-database sharded scanning (`vyre-libs`)

- Added `vyre_libs::scan::scan_pattern_sharded(shards: &[PatternShard], backends, haystack) -> Vec<Match>`: the W3-5 `pattern-database-replicated-shards` workload, it stripes the RULE database (not the haystack) across a device set. Each `PatternShard` is a sub-matcher over a disjoint rule subset plus a local→global pattern-id map; it runs on `backends[shard % n]`, its matches are remapped to the global rule numbering, and all shards merge into the canonical `(pattern_id, start, end)` report order. Because literal matching is independent per rule, the striped union equals the full un-sharded matcher's match set, the plan's replicated/striped parity policy. Fails closed on an empty device set and on a malformed shard map (a local id with no global mapping errors rather than dropping or mis-attributing the finding). Proven on the RTX 5090 (a 2-shard stripe over 1- and 2-device sets equals the full-database scan; malformed map errors).

### Added: multi-GPU sharded scanning (`vyre-libs`)

- Added `vyre_libs::scan::scan_sharded_fused(matcher, backends: &[&dyn VyreBackend], files, window_budget_bytes, max_matches)`: the W3-5 `regex-haystack-byte-range-shards` architecture. It distributes the corpus's byte-range window shards round-robin across a device SET (window `k` → `backends[k % n]`), each backend holding its own resident fused session, so on a multi-GPU host the shards run concurrently on distinct peer devices. The partition, halo (`L-1` overlap), and aggregation (host globalize + stable sort by `(region, start, end, pattern_id)`) reuse the exact `scan_paged_fused` helpers (ONE PLACE), so the sharded result is byte-identical to a single-shot scan for any device-set size, the plan's parity policy. Fails closed on an empty backend set; one ordered free pass so no resident session leaks. Proven on the RTX 5090 (1-device and 3-device sets both equal the single-device scan, boundary-spanning match survives sharding, empty set errors). On a single-device host the shards run sequentially; only cross-device parallelism awaits a second physical GPU.
- Added `scan_sharded_fused_timed(...)` (with `ShardTiming` / `ShardedScanTiming`): the per-shard-timed twin, an identical result plus a per-device breakdown of windows, byte-work, wall time, and device (kernel) time. This is the `per-shard-active-ns` signal the plan's `load_balance_policy` rebalances on: a skewed timing across shards under equal round-robin is the evidence to feed proportional `weights` into `scan_sharded_fused_weighted` next batch. Each shard's `device_ns` stays `Some` only while every window on it reported device time (loud `None` otherwise, never a fabricated 0); an idle shard reports `Some(0)`. Proven on the RTX 5090 (per-shard window counts and byte-work sum to the totals, each active shard reports real wall + non-zero device time, timed result == untimed).
- Added `scan_sharded_fused_weighted(matcher, backends, weights: &[u32], files, window_budget_bytes, max_matches)`: the throughput-weighted twin, cumulative byte-work per device tracks `weights[i]` (the plan's `device-throughput-weight` / `load_balance_policy`) via a deterministic greedy least-loaded-by-weight assignment shared with the round-robin path (ONE PLACE `shard_assignment`; zero weight treated as 1, never starved). Fails closed on a weights/backends length mismatch. Because aggregation is order-independent, the weighted result is byte-identical to round-robin and single-shot for any weights, only the work distribution changes. Proven by a pure host unit test (3:1 weight → 3 of 4 windows to shard 0) and the RTX 5090 parity test.

### Added: paged corpus scanning (`vyre-libs`)

- Added `vyre_libs::scan::paged_corpus::scan_paged_fused` (with `PagedScanResult` and `GlobalMatch`): scans a corpus of files that may exceed one resident window as a sequence of resident fused-window dispatches, returning the per-region presence bitmap in a single global region numbering plus every positioned match in u64 global coordinates. Files are planned into byte-budgeted windows at file boundaries with stable global region ids; each window runs as an independent local scan and is globalized on the host with `L-1`-byte overlap, a discardable dummy overlap region, and start-based dedup, so the result is byte-identical to a single-shot scan of the concatenated corpus (no boundary miss, no over-fire, no double count), while host RSS stays bounded by one window instead of the whole corpus. Proven on real GPU against a single-shot scan including a boundary-spanning match.
- Added `scan_paged_fused_timed` (with `PagedScanTiming`): the timed twin of `scan_paged_fused`, extending W3-3 "attribution everywhere" onto the paging path. It returns a result byte-identical to the untimed driver plus an honest aggregate over the per-window dispatches, window count, total own bytes scanned (overlap excluded, a valid throughput denominator), summed wall-clock time, and summed device (kernel) time. The device aggregate is `Some` only when every window reported a device timer; a single timer-less window collapses it to a loud `None`, never a fabricated 0 (Law 10). It differs from the untimed driver in exactly one call (`scan_into_timed` vs `scan_into`) and reuses the same shared staging/globalization helpers, so the paged result cannot drift. Proven on real GPU (timed == untimed, device time present and non-zero) plus an empty-corpus test locking the zero-window `Some(0)` aggregate.
- Added `scan_paged_fused_async`: the asynchronous twin that pipelines the windows (window `k+1`'s staging and upload overlap window `k`'s device execution, two dispatches in flight) via the borrowed async fused dispatch. It shares the exact overlap/dummy-region/dedup globalization with the synchronous driver, so its result is bit-for-bit identical (proven on real GPU (async == sync)).
- Added `scan_paths_paged`: the disk-backed paged scanner, it takes file paths and reads only one window's files into memory at a time, so host RSS stays bounded by the window rather than the corpus. It shares the same globalization as the in-memory driver, so its result is identical (proven on real GPU (disk == in-memory) plus a no-GPU test of the window disk-read + overlap prefix).
- Added `scan_paths_paged_prefetched`: the prefetching disk scanner, a background thread reads window `k+1`'s files while the GPU scans window `k`, so disk I/O overlaps device compute, with a depth-1 bounded channel keeping host RSS to at most two windows. Result is identical to the synchronous disk scan (proven on real GPU (prefetched == sync)).

### Added: fast-path corpus example (`vyre-libs`)

- Added `vyre-libs/examples/scan_corpus_fast_path.rs`: a runnable consumer example that coalesces a set of files (a real directory tree, or a built-in multi-file corpus) into a haystack plus `region_starts`, compiles the matcher once, prepares a resident fused session, and runs one timed dispatch producing both the per-region presence bitmap and the positioned matches, the runnable companion to the fast-path guide. With no GPU it falls back loudly to the portable `scan_all` on the CPU reference backend.
- Added `vyre-libs/examples/scan_paged_corpus.rs`: the disk-ingress companion for a corpus larger than one window. It materializes a multi-file corpus on disk (or pages a real directory-tree argument), plans windows under a deliberately tiny byte budget to force multi-window paging, and runs `scan_paths_paged_prefetched`, printing per-file presence and every positioned match in global (file-index, u64-byte) coordinates. A pattern that straddles a window boundary is reported exactly once. With no GPU it says so loudly and falls back to reading every file into memory plus `scan_paged_fused_async` on the CPU reference backend, surrendering the bounded-RSS property but yielding the same global match set.

### Added: fast-path scanning guide (`docs`)

- Added `docs/scanning-a-corpus-the-right-way.md`: the intended route through the resident/async/fused/count-then-collect APIs, a decision table for which API to use and a five-step fast path (compile once → prepare a resident session → overlap batches with the async twins → leave timed attribution on → let the device count with `scan_all`). Every signature is copied verbatim from the current `GpuLiteralSet` public surface, and the guide is listed in `docs/INDEX.md`.

### Added: head-to-head vs CPU aho-corasick benchmark (`vyre-bench`)

- Added the `scan.literal_set.vs_cpu_aho_corasick` benchmark case: it runs the same pattern set over the same consumer-shaped corpus through vyre's resident GPU literal-set scan (end-to-end, staging included) and the `aho-corasick` crate (built with `MatchKind::Standard` + `find_overlapping_iter`, the all-overlapping semantics vyre's DFA emits), and reports the end-to-end speedup plus the GPU device-vs-staging split. Correctness is a hard gate, the GPU matches must be byte-identical to the aho-corasick matches (a fast wrong answer fails), and the CPU baseline is pre-checked to reproduce the engine's `reference_scan` set exactly. The performance delta is reported, not gated: this is the standing head-to-head that makes the "beats the best CPU path end-to-end" claim (and any gap) visible per release.

### Added: consumer-shaped cold-start & decode-heavy benchmarks (`vyre-bench`)

- Added the `scan.literal_set.cold_start` benchmark case: it times the full cold-start path of a one-shot literal-set scan, building the matcher (`try_compile`), the first table upload, and the first dispatch with cold caches, against the warm steady-state per-dispatch cost, and reports the cold-start overhead factor plus the compile-vs-first-touch split. This is the cost a consumer that scans one corpus and exits actually pays, invisible to a steady-state loop.
- Added the `scan.literal_set.decode_heavy` benchmark case: it measures the decode-bound regime on a dense-match corpus (the shortest pattern tiled every 128 bytes, ~32k matches over 4 MiB) scanned through a resident session, so the immutable tables upload once and every dispatch is dominated by writing the match triples, reading them back, and decoding them on the host, reporting the device-vs-host-decode split. Both cases hard-gate correctness: the GPU matches must be byte-identical to the independent CPU `reference_scan` (Law 10), verified via exact-output comparison and, without a GPU, by `CpuRefBackend` unit tests.

### Added: async two-batch overlap benchmark (`vyre-bench`)

- Added the `scan.literal_set.async_overlap.2batch` benchmark case: it runs the asynchronous literal-set position scan over two distinct consumer-shaped batches both sequentially (submit → await → submit → await) and overlapped (submit A → submit B → await A → await B), and reports the overlap factor plus the sequential kernel-vs-host-staging split. Correctness is a hard gate: the overlapped matches must be byte-identical to the sequential ones for both batches (Law 10, overlap changes no result bit), verified via the case's exact-output comparison and, without a GPU, by a `CpuRefBackend` unit test. This is the quantitative companion to the existing `literal_set_async_two_batch_pipeline` correctness gate.

### Added: distinct regex-unsupported diagnostics (backreference / huge alternation / nested repeats / capture)

- The GPU-NFA regex frontend now DISTINCTLY detects four constructs that previously collapsed into a generic `Parse` or `TooManyStates` error, so a consumer can route each on its canonical `REGEX_UNSUPPORTED_DIAGNOSTICS.toml` code. Backreferences (`\1`, `\k<name>`, `(?P=name)`) are classified by an escaping-aware structured source scan (run only on parse failure, never by matching parser error text) and map to `VYRE_SCAN_UNSUPPORTED_BACKREFERENCE`. Over-budget alternations map to `VYRE_SCAN_UNSUPPORTED_HUGE_ALTERNATION_BUDGET` and nested bounded repeats whose unroll product exceeds the state budget map to `VYRE_SCAN_UNSUPPORTED_NESTED_REPEAT_BUDGET`, both detected before lowering collapses them into `TooManyStates`. The reclassification is sound: both budgets equal the state cap, so no pattern that compiled before now errors.
- Capture groups remain a successful whole-match compile (making them an error would regress acceleration); `CompiledRegexSet::captures_present` and `CompiledRegexSet::capture_extraction_diagnostic_code()` surface the `VYRE_SCAN_CAPTURE_EXTRACTION_REQUIRES_VERIFIER` signal for a consumer that needs submatch spans.
- New public API: `RegexConstruct` enum and `regex_construct_diagnostic_code`: the single owner of every regex-construct diagnostic code string, through which both `RegexCompileError::diagnostic_code` and the capture-signal path route.

### Added: resident fused presence+positions pipeline (`ResidentFusedRegionScan`)

- Added `GpuLiteralSet::prepare_resident_fused_scan` and the `ResidentFusedRegionScan` session it returns (`scan_into`, `scan_into_timed`, `max_regions`, `max_matches`, `haystack_capacity`, `free`): the resident twin of the FUSED per-region presence + positions scan (`scan_presence_and_positions_by_region`). It is the fusion of `ResidentPresencePipeline` (per-region presence bitmap + region controls) and `ResidentLiteralScan` (positioned match output), one all-resident dispatch of the 14-binding fused program produces BOTH outputs, uploading the immutable DFA + suffix-prefilter tables ONCE and re-staging only the haystack, region controls, and two zeroed accumulators (presence prefix + match counter) per scan. All 14 bindings are resident (incl. the two read-write accumulators and the `matches` output), so it runs on the CUDA backend; the fixed-size `matches` buffer fails CLOSED on overflow (Law 10), and an over-capacity haystack or over-cap region count fails closed before dispatch. Real-GPU parity: the resident presence bitmap AND triples are byte-identical to the borrowed fused scan across repeated re-dispatches.

### Added: resident position-scan pipeline (`ResidentLiteralScan`)

- Added `GpuLiteralSet::prepare_resident_scan` and the `ResidentLiteralScan` session it returns (`scan_into`, `scan_into_timed`, `max_matches`, `haystack_capacity`, `free`): the positioned-scan sibling of `prepare_resident_presence`. It uploads the immutable DFA transition/output/pattern-length tables and the three suffix-prefilter masks into backend resources ONCE, then re-dispatches the literal MATCH program across a corpus re-uploading only the per-file haystack and resetting a 4-byte match counter, eliminating the multi-MiB per-scan table re-upload the borrowed `scan_into` repeats on every file. All 11 bindings are resident (including the `matches` output buffer, which the resident dispatch resolves as an output and reads back), so it runs on the CUDA backend with no borrowed mix. The fixed-size resident `matches` buffer FAILS CLOSED when the device match count exceeds `max_matches` (never a silent truncated decode), and an over-capacity haystack fails closed before any upload. Measured 1.84× faster than borrowed across a 400-detector / 192-scan corpus on an RTX 5090.

### Added: attribution (`TimedDispatchResult`) twins for every literal dispatch path

- Added `GpuLiteralSet::scan_presence_by_region_timed` and `scan_into_timed`: timed twins of the hot region-presence and position (`scan_into`) paths, returning `vyre_driver::TimedDispatchResult` (wall / device / enqueue / wait) alongside the same result the untimed entry produces, so a consumer or benchmark can split per-scan cost between the GPU kernel (`device_ns`) and host staging/readback. The untimed hot paths are untouched and pay no timing cost; `device_ns` is a loud `None` on a backend without a device timer, never a fabricated zero.
- Added `GpuLiteralSet::scan_presence_timed` (global-presence path) returning `(bitmap, TimedDispatchResult)`, built on a new owned dispatch-staging path that reuses the shared immutable-table encoder so every presence path encodes byte-identical tables.
- Added `GpuLiteralSet::scan_presence_and_positions_by_region_timed` (fused presence+positions path) returning `(bitmap, TimedDispatchResult)` and decoding the `(pattern_id, start, end)` triples into a caller buffer, with the same fail-closed overflow contract as the untimed fused scan (a match count over `max_matches` errors, never a silent truncated decode).
- Added `GpuLiteralSet::scan_all_timed` (auto-resize complete-match path) returning `ScanAllTimed { timed, resized }`: the timing describes the dispatch that produced the returned matches, and `resized` loudly states whether that was the resize re-dispatch (the two-launch case is reported, never silently summed).

### Added: asynchronous (`PendingDispatch`) twins for every single-dispatch entry point

- Added `GpuLiteralSet::scan_presence_async` (→ `PendingPresence`), `scan_into_async` (→ `PendingMatches`), and `scan_presence_and_positions_by_region_async` (→ `PendingFusedRegion`): submit the GPU dispatch and return a handle immediately so callers can overlap host-side work with the in-flight scan, then decode via `await_words` / `await_into` / `await_matches`. Each retains its owned upload buffers until the decode and, on a non-pipelining backend, yields a trivially-ready handle whose result is byte-for-byte identical to the synchronous entry (no silent change on the degraded path). Together with the pre-existing `scan_presence_by_region_async` this covers every single-dispatch scan entry point.
- New public types: `ScanAllTimed`, `PendingPresence`, `PendingMatches`, `PendingFusedRegion`.

### Added: device-side per-region compaction primitive

- Added `vyre_primitives::matching::region::compact_first_per_region_pattern_flag_program` (op id `COMPACT_FIRST_PER_REGION_PATTERN_OP_ID`) and its CPU-parity oracle: a per-invocation first-occurrence kernel keyed on the `(region, pid)` pair that emits a survivor flag for the first match of each pair, so stream-compaction leaves exactly one positioned representative per pair, the positioned companion to the presence-by-region bitmap, computed on device with no host per-region group-by after readback. Completes the W2-5 device-side post-processing set (sort, dedup, per-pattern cap, per-region compaction).

### Added: grid-aware reference evaluation

- Added `vyre_reference::reference_eval_with_dispatch` / `run_arena_reference_with_dispatch`, which let a caller pass the true byte-scan grid (invocation count) so the interpreter covers what the real GPU dispatch would. `reference_eval` is unchanged (grid floor 0). This closes a silent under-coverage in the reference oracle where a byte-scan over a haystack larger than its max buffer element count skipped high positions on CPU-ref only (the GPU was always correct).

## [0.6.4]  -  2026-06-23

- Added `GpuLiteralSet::prepare_resident_presence` and the `ResidentPresencePipeline` it returns: a resident literal-set region-presence session that uploads the immutable DFA transition/output/pattern-length tables and suffix-prefilter masks into backend resources ONCE, then re-dispatches across a corpus's coalesced batches re-uploading only the per-file haystack and resetting the per-region presence buffer, eliminating the multi-MiB per-scan table re-upload the borrowed `scan_presence_by_region` path repeats on every file. All-resident so it runs on the CUDA backend.

- Added `ResidentPresencePipeline::scan_into_timed` returning `TimedDispatchResult` (wall / device / enqueue / wait nanoseconds) so callers can attribute a region-presence dispatch's GPU-kernel time separately from host staging and decode; `scan_into` now wraps it. Direct CUDA attribution on an RTX 5090 (8 MiB, 900 detectors) measured the region-presence kernel at ~41 µs (the borrowed path's cost is per-scan table re-upload, not the kernel).

- Made `prepare_resident_presence` fail closed at prepare time when the requested resident haystack capacity is smaller than the NFA program's statically-declared input buffer (binding 0), with an error naming the required byte count and the fix, instead of dispatching against an undersized resident buffer.

- Added `GpuLiteralSet::scan_presence_and_positions_by_region[_with_scratch]`, a single suffix3 dispatch that folds per-region literal presence and confirmed match positions into one GPU pass (previously two separate dispatches), with GPU-vs-exhaustive-CPU-reference differential coverage.

- Added row-strided queue-to-queue delta enqueue for skewed CSR fixpoint waves, wired IFDS queue closure to select it for high-degree rows, and refreshed public API snapshots for the exposed graph/frontier planning surfaces.

- Made the CUDA-resident C sparse lexer compact terminal path read back `out_counts` first and then download only the live dense token column ranges, cutting host transfer volume for sparse translation units without breaking the resident GPU chain.

- Sized C sparse-lexer compact outputs from the scanned token count instead of source byte count for staged and block-total compaction paths, reducing readback and downstream token-buffer pressure on whitespace-heavy translation units.

- Made budgeted resident CSR queue batches plan ordered chunks from each chunk's effective frontier popcount, so sparse runs before and after a dense outlier still pack tightly under the resident scratch budget.

- Clamped resident CSR frontier-queue dispatch capacity from in-domain frontier popcount, reducing graph-sized scratch allocation and overlaunch for sparse single-query and batched traversals while keeping caller queue capacity as a hard cap.

- Sized resident adaptive sparse-queue traversal from the active frontier popcount instead of graph node count, reusing larger queue scratch across smaller frontiers and preserving row-strided traversal for high-degree rows.

- Added 30,000 generated row-strided CSR queue primitive checks covering skewed graph traversal, caller-owned output reuse, malformed CSR rejection, and dispatch-grid coverage.

- Routed the IFDS skewed active-queue and queue-materialization benchmarks through the row-strided CSR queue consumer for high-degree rows, increased the benchmark fixture hub degree to 2,048 edges, and added telemetry proving when the strided traversal path is active.

- Added a row-strided CSR queue traversal primitive for skewed active frontiers, wired resident CSR and adaptive sparse-queue paths to select it for high-degree rows, and refreshed the `vyre-primitives` public API snapshot.

- Made the sparse C tokenizer's raw `U8` haystack runtime-sized, removing the host-side bucket padding copy before token classification while keeping bucketed GPU output shapes.

- Moved the full C comment/splice fallback to runtime-sized raw `U8` source buffers, removing the remaining padded splice-input staging from the byte-filter pipeline.

- Added a backend-extension gate proving new backends remain one crate plus `inventory::submit!`, and declared SPIR-V dispatch capability through the same inventory path as CUDA and wgpu.

- Hardened the base monument benchmark check so it proves the executable `vyre-bench` meta-harness, JSON registry, thesis workload IDs, and deep coverage dimensions instead of only checking for the PRD.

- Added a million-node graph frontier benchmark to `vyre-bench`, with exact CPU-oracle verification and release-suite thesis coverage contracts so benchmark evidence cannot regress to element-wise-only workloads.

- Added explicit graph launch sizing for CSR frontier degree-sum and refreshed the public API snapshot for the current graph/dispatch surfaces.

- Added explicit RLE segment-length dispatch sizing and multi-block CPU/CUDA parity coverage for packed decode workloads.

- Added explicit bigint add-carry dispatch sizing and multi-block CPU/CUDA carry-pattern coverage for large limb arrays.

- Added explicit union-find dispatch sizing through the self-substrate path and multi-block CUDA coverage for large edge batches.

- Added explicit d-DNNF evaluation dispatch sizing and multi-block CUDA coverage for literal-heavy knowledge-compile waves.

- Reworked Scallop single and wide lineage fixpoint kernels to preserve high-cell and high-word seed facts without CUDA grid-barrier races, with CUDA parity coverage for the exposed high-word case.

- Restored multi-block Scallop dispatch for large relation matrices through split-visible GridSync phases while keeping small matrices on the block-local persistent path.

- Packed `tensor_flow_forward` source-node dataflow lanes into 256-lane workgroups and added CUDA parity for context/field propagation past the first block.

- Made GPU region dedup cluster-aware for nested/touching scanner spans, added merged-end metadata for on-device compaction, and proved multi-workgroup CUDA parity.

- Added a 256-lane parallel `bracket_match` path when parser depth caps cannot affect output, with CUDA parity for large nested token streams and retained bounded-stack fallback for overflow-capped shards.

- Routed large adaptive sparse-queue traversal frontiers through the deterministic word-prefix queue materializer, with resident CUDA parity for a large sparse graph step and refreshed adaptive traversal program-cache identities.

- Replaced multi-block word-prefix queue scatter's per-word previous-block loop with an in-place block-offset scan and precomputed-offset scatter, with resident CSR/adaptive wiring and live CUDA coverage for generated multi-block frontier queries.

- Added a CSR-only resident adaptive sparse-queue graph upload and step path so sparse-queue workloads avoid dense adjacency allocation/upload, with live CUDA telemetry coverage and generated sparse-queue matrix coverage on the no-dense path.

- Added CSR frontier queue property gates covering 40,000 generated materialization, traversal, adversarial queue, and validation cases, and doubled live CUDA adaptive sparse-queue generated coverage to 1,024 resident steps per materializer.

- Removed the redundant resident atomic sparse-queue `queue_len` init dispatch from CSR and adaptive traversal paths, dropping small resident sparse-queue steps from four kernels to three while keeping queue length initialization inside `frontier_to_queue`.

- Added packed-`U8` line indexing, UTF-8 validation, and C line-splice classification for text scans, fixed CUDA/PTX byte and halfword memory ops, and covered the paths with generated reference parity plus live CUDA boundary matrices.

- Moved the C preprocessing byte filter to raw `U8` source buffers through preflight, line/block comment paths, full comment masking, and compact scatter, fixed literal-close handling before later comments, and added live CUDA generated-corpus coverage for the end-to-end filter.

- Moved the sparse C tokenizer pipeline to a raw `U8` haystack while preserving packed and expanded compatibility entrypoints, with reference-eval ABI checks and live CUDA generated-corpus parity for token and directive columns.

- Moved the C directive-metadata stage used by the preprocessing pipeline to raw `U8` source bytes while preserving the packed standalone ABI, eliminating another source repack between tokenization and directive classification.

- Moved fused `#define`/`#include`/`#undef` payload parsing in the preprocessing pipeline to raw `U8` source bytes while preserving packed standalone parser ABIs.

- Moved `#ifdef`/`#ifndef` and `#if`/`#elif` compatibility evaluators in directive extraction and live conditional re-evaluation to raw `U8` source rows and macro-name tables while preserving packed standalone evaluator ABIs.

- Removed the now-unused C GPU-preprocess U32 byte-padding staging helper so raw-byte directive and live conditional paths cannot route back through padded host macro-name buffers.

### New

- **`vyre-foundation`  -  effects-handler lowering is on the release path.**
  `PassScheduler` now has an effects-handler enforcement gate: rewrites may
  discharge existing effects, but any newly introduced effect row bit is
  reverted unless the pass declares it through `allowed_effect_additions`.
  Backend `pre_lowering::optimize` enables this gate beside cost-monotone
  enforcement, and pass metrics now expose before/after effect-row bits.
- **`vyre-foundation`  -  linear BufferAccess is on the release path.**
  `PassScheduler` now enforces `BufferDecl::linear_type` postconditions for
  backend pre-lowering: rewrites may repair existing violations but cannot
  introduce new linear/affine/relevant usage violations before lowering. Pass
  metrics expose before/after linear-violation counts.
- **`vyre-foundation`  -  liquid BufferDecl shapes are on the release path.**
  `PassScheduler` now enforces `BufferDecl::shape_predicate` postconditions for
  backend pre-lowering: rewrites may repair existing shape violations but cannot
  introduce new predicate/count contradictions before CUDA or WGPU lowering.
  Pass metrics expose before/after shape-violation counts.
- **`vyre-foundation`  -  liquid shapes now erase dynamic loop guards.**
  `loop_var_range_fold` consumes `ProgramShapeFacts` so comparisons between a
  loop induction variable and `buf_len(buffer)` fold when `ShapePredicate`
  min/max facts prove the branch true or false. Runtime-sized buffers with
  `AtLeast`/`Exactly`/bounded affine shape facts can now drop redundant
  per-iteration bounds checks before CUDA lowering.
- **`vyre-foundation`  -  wire parser adversarial properties run in normal CI.**
  Added generated `Program::to_wire`/`Program::from_wire` property coverage for
  10,000 generated programs, 10,000 arbitrary hostile byte blobs, 10,000
  truncations, and 10,000 digest-refreshed body mutations. The new tests found
  and fixed a decoder gap where tampered but checksum-correct bytes could
  produce zero workgroup dimensions; `from_wire` now rejects zero workgroup
  dimensions and invalid output byte ranges at parse time.
- **`vyre-foundation` / `vyre-driver-cuda` / `vyre-reference`  -  explicit
  single-rank collectives execute through one shared transform.** Added
  substrate-neutral lowering for `CommGroup::WORLD` `AllGather` and
  `ReduceScatter` into bounded copy IR while reducing single-rank `AllReduce`
  and root-0 `Broadcast` to identity semantics. CUDA dispatch, CUDA compiled
  pipelines, and the reference oracle now consume the same transform. Non-world
  groups and nonzero single-rank broadcast roots fail closed with actionable
  errors, so multi-rank transport is never silently emulated. New proptests
  generate 16,384 collective-lowering/reference cases and live CUDA tests cover
  host dispatch, native compiled pipelines, and adversarial root rejection.
  Capability scanning now distinguishes lowerable single-rank collectives from
  collectives that genuinely require transport, and the canonical pre-emit
  pipeline applies the same transform before descriptor lowering.
- **`xtask` / release gates  -  recursion thesis is load-bearing.** Repaired
  `recursion-gate` root detection for the standalone Vyre workspace, made it
  scan the current `vyre-self-substrate/src` tree recursively plus the primitive
  catalog surface, taught it to parse grouped Rust imports across newlines,
  excluded private helper modules from the public primitive inventory, and
  wired `scripts/check_recursion_gate.sh` into release signoff so missing
  self-consumers fail release validation. Added the self-substrate
  `data::parsing_dispatch_pipeline` so packed-AST constant folding and
  bytecode dispatch-table packing consume the parsing primitives on the
  production substrate path.
- **`vyre-foundation`  -  derived pass-order artifact.** Added
  `optimizer::derived_order` with a live inventory-derived pass order,
  declared requirement edges, causal invalidation adjacency, and
  adjustment-set back-door safety checks. Release pass-order validation now
  consumes this artifact instead of reconstructing an independent ordering.
- **`vyre-foundation`  -  planar rewrite batching on the optimizer execution
  path.** Added a foundation-owned non-overlap batch planner,
  `ProgramPass::batch_apply`, refusal-aware `try_batch_apply`, and scheduler
  wiring so high-candidate passes can apply disjoint rewrite waves instead of
  relying on one-candidate-at-a-time launches. The primitive reference oracle
  now delegates to the same planner, keeping CPU contracts and GPU primitive
  tests on one source of truth. The batch activation threshold is runtime
  configurable through `VYRE_PLANAR_REWRITE_BATCH_THRESHOLD`.
- **`vyre-driver` / `vyre-driver-wgpu`  -  natural-gradient launch resolver
  on release paths.** Exported the canonical workgroup candidate table and
  shared launch resolver, wired CUDA `LaunchPlan` and WGPU pre-lowering
  config through safe-gated natural-gradient cold-start workgroup selection,
  and cached the selected launch shape per program/element-count/limit tuple
  so the hot path does not rebuild policy vectors. CUDA timed dispatch now
  records real `device_ns` measurements back into the bounded launch cache,
  allowing later automatic launches to move away from the cold-start
  heuristic when hardware timing proves another candidate faster. WGPU timed
  dispatch now returns timestamp-query `device_ns` as structured
  `TimedDispatchResult` data and feeds it into the same launch-feedback path.
  Measured launch decisions now persist across process restarts through the
  existing bounded tuner TOML cache.
- **`vyre-primitives`  -  dominator-tree public primitive surface.** Added
  the registered graph primitive to the self-consumer catalog, moved its
  scale/VRAM benchmark into the central `vyre-bench` release harness, and
  refreshed the public API snapshot for the new graph contract.
- Document `vyrec` / `vyre-frontend-c` as beta active-development consumers
  rather than the core Vyre `0.4.2` release proof.
- [A06] Document workspace member listing convention (S13)
- [A11] Bulk-fill Jules ticket queue (fixture_sweep + cve_replay)
- [A05] Examples consume published crates via patch.crates-io
- [A03] Validator error code documentation (S8)
- [A02] Rename vyre-cc to vyre-frontend-c
- **`vyre-foundation`  -  `BinOp::MulHigh` IR primitive.** Widening unsigned
  32×32→64 multiply returning the upper 32 bits. Wire tag `0x21`.
  Full support: const-fold in `ir_eval.rs`, interpreter in `node_kind.rs`,
  wire encode/decode in `bin_op_tag.rs`/`bin_op_from_tag.rs`, and
  `Expr::mulhi()` builder. Required for Granlund-Montgomery division.

- **`vyre-foundation`  -  Granlund-Montgomery constant division.** Strength-reduce
  pass now rewrites `x / d` (for constant non-power-of-two `d`) into a
  `MulHigh + Shr` sequence using Hacker's Delight Algorithm D. Eliminates the
  ~70-cycle hardware division in favor of ~5-cycle multiply-shift. Exhaustive
  correctness tests cover all divisors 2–1000 plus extreme boundary cases
  (2³¹±1, 2³²−1). Located in `optimizer/passes/strength_reduce/arithmetic.rs`.

- **`vyre-driver`  -  `LoweringStrategy` trait + capability-driven selector.**
  Two-layer optimization architecture: Layer 1 (IR-level math rewrites in
  `vyre-foundation/optimizer/passes/`) is backend-agnostic. Layer 2 (backend
  lowering strategies in `vyre-driver/strategy/`) is target-dependent.
  Strategies declare capabilities via `BackendCapabilities` and are selected
  by priority. `select_strategy()` picks the highest-priority applicable
  strategy. See `docs/ARCHITECTURE.md § Two-layer optimization architecture`.

- **`vyre-libs`  -  `c_lower_ast_to_pg_nodes` Cat-A op.** Added registration for
  `vyre-libs::parsing::c::lower::ast_to_pg_nodes`, a pure-IR lowering from
  structural VAST rows to packed `PgNode` tuples
  `(kind, span_start, span_end, parent_idx, payload_lo, payload_hi)`.
  Added witness fixture, pure CPU reference oracle, WGSL emission smoke test,
  GPU dispatch parity sample, and adversarial coverage (60 fixtures + proptest).

- **`vyre-runtime`  -  persistent megakernel + `io_uring` NVMe streaming.**
  The GPU becomes a VIR0 bytecode interpreter that loops forever reading
  slots the host publishes into a ring. Linux-only NVMe zero-copy via raw
  `io_uring_setup` + mmap of SQ/CQ rings, with a `uring-cmd-nvme` feature
  for `IORING_OP_URING_CMD` passthrough (kernel 6.0+). Three-buffer
  layout (control / ring / debug_log), 256-lane × N-workgroup sharding,
  opcode extension hook for vendor intrinsics, per-tenant authorization
  masks, atomic `done_count` counter, and a PRINTF debug channel.
- **`vyre-libs`  -  Category A composition ecosystem.** Pure-IR
  compositions over `vyre-ops` primitives (`math`, `nn`, `matching`,
  `crypto`). No raw shader source  -  every library function is a
  `Program` consumers can round-trip, validate, and inline.
  `substring_search` lands with a real byte-by-byte equality instead of
  the earlier LAW 1 placeholder.
- **10 io_uring + IR innovations.** `IORING_REGISTER_BUFFERS` +
  `READ_FIXED`, `IORING_REGISTER_FILES` + `IOSQE_FIXED_FILE`, GPUDirect
  Storage `GpuMappedBuffer::from_bar1_peer`, `futex_waitv` completion
  doorbell, per-workgroup slot sharding, ring-credit backpressure,
  opcode extension hook, tenant-mask routing, PRINTF debug channel,
  AF_XDP/RDMA ingress demonstrated via a TCP smoke test.
- **Error-code catalog grew a `P-*` family** for
  `vyre-runtime::PipelineError`.
- **Workspace docs pristine.** `cargo doc --workspace --all-features
  --no-deps` runs clean  -  zero unresolved intra-doc links, zero
  private-link leakage, zero output collisions.

### Fixed

- **Descriptor `identity_elim` fma-zero fold ignored inf/NaN**  -  it folded
  `Fma(a, b, c) → c` whenever a factor was a literal numeric zero, with no
  check on the other factor. vyre Fma is float-only and `0.0 * inf =
  0.0 * NaN = NaN`, so `Fma(0.0, inf, c)` is NaN, not `c`: the fold silently
  replaced a NaN with the addend. Now requires the other factor to be a
  *finite literal*, matching the foundation `simplify_fma` guard (one
  auditable contract via the new `ScalarLiteral::is_finite_numeric`).
  Regression test asserts `Fma(0.0, inf, c)` is not folded.
- **Descriptor LICM hoisted convergent subgroup collectives out of loops**  -
  `SubgroupBallot/Shuffle/Broadcast/Reduce` were classified hoistable. Their
  result depends on the participating-lane set, so lifting one out of a loop
  (execution count N → 1) changes that context and the result. Now fail-closed
  for the four collectives, matching the authoritative foundation
  `expr_is_observably_free` gate; `SubgroupLocalId`/`SubgroupSize` stay
  hoistable as per-lane loop-invariant constants. Regression test asserts a
  `subgroupAdd` of a loop-invariant value stays inside the loop.
- **Loop fusion fused across a compare-exchange `expected` cross-loop read**  -
  `collect_vars_in_expr` walked an atomic's `index` and `value` but dropped the
  CAS `expected` operand, so a fusion that reordered a scalar the `expected`
  reads was not blocked. Now walks `expected` (and is exhaustive over leaf
  variants); proven by a `reference_eval` oracle differential.
- **LAW 1 placeholder in `vyre-libs::matching::substring_search`**  -  the
  inner-byte check was `Expr::u32(1)` (matched every position); now
  `load(haystack, i+k) == load(needle, k)` routed through a select to
  stay integer. Gap L-7 closed with a structural regression test that
  fails if the compare ever collapses back to a constant.
- **LAW 9 evasion audit sweep**  -  removed all `// TODO` / `// FIXME`
  markers from shipped code. Subgroup intrinsics return a structured
  error pointing at RFC 0004 instead of a TODO; the autotune workgroup
  heuristic is documented as intentional default instead of a TODO.
- **Driver binary name collision**  -  `vyre-driver-wgpu`'s CLI bin
  renamed from `vyre` → `vyre-wgpu` so it no longer collides with the
  `vyre` lib target in `cargo doc`.
- **Workspace version drift**  -  `vyre-runtime` workspace dep bumped
  from `0.1.0` → `0.6.0` to match the crate's own manifest.
- **`vyre-libs::security::aliases_dataflow` RAW-hazard barrier gap.**
  The local `merge_programs` helper concatenated the seed / hop /
  merge / intersect / union sub-programs without inserting any
  `Node::Barrier`. Threads in later warps observed pre-seed
  `reach_x_buf` state and the BFS frontier silently dropped nodes
  past the warp boundary on every aliases-using rule. Routed
  through `vyre_foundation::execution_plan::fusion::fuse_programs`
  so RAW/WAR hazards get precise barriers. Local helper deleted.
  Two regression tests pin the structural barrier presence and
  unique non-Workgroup binding numbering in the fused output.
- **`vyre-libs::parsing::python` validator-rejected programs.**
  Lex-level `is_ident_start` / `prev_identish` lets stored bool
  exprs that the validator rejected when later compared with
  `u32(0)`; coerced through `select` so the bool→u32 lift happens
  at the let_bind. Structure / call / decorator extractors hoisted
  every cross-block name (`name_end`, `cursor`, `dot_pos`,
  `after_dot`, `target_tok`, `target_name`, `target_kind`,
  `async_def`, `after_decorator`, `after_type_params`, `after_params`,
  `decorator_end`) into the outer body so they outlive the
  if-then blocks that assign them, with new
  `search_next_token_into` / `find_matching_delimiter_into`
  assign-only helpers used inside if-blocks to skip the redundant
  outer let_bind. Closes 13 cascading V008 / V032 / undeclared-var
  validation errors that hid behind a single bool/u32 mismatch.
- **`vyre-primitives::reduce::workgroup_tree`** E0382 use-of-moved-
  value on `dtype: DataType` consumed three times in a single
  `Program::wrapped` BufferDecl block; first two uses now
  `dtype.clone()` so the third use lands on the still-owned value.
- **`vyre-primitives::effects::handler_apply::tests::from_bits_round_trip`**
  literal `0b101_0011` corrected to `0b0010_1011` (bits 0, 1, 3,
  5 = BufferWrite + Atomic + GpuDispatch + AsyncLoad). The pre-fix
  literal had bits 0, 1, 4, 6 set (Atomic + Barrier + Trap) but
  the assertions read GpuDispatch / AsyncLoad → guaranteed test
  failure regardless of the runtime behavior.
- **`vyre-libs::nn::attention::attention_reference_program`** signature
  drift: the function returns `Program` but the body used `?` /
  `Ok(...)`, which only compile under a `Result<…>` return.
  Reverted to panic-on-overflow (callers wanting the fallible path
  go through `try_attention_reference`, which already returns
  `Result<Program, TensorRefError>`).

### Changed

- **Driver boundary and shared-driver lifts.** Concrete backend crates now own
  concrete runtime/API names, while `vyre-driver` hosts shared AOT emitter
  registration, validation cache, binding/program walks, specialization maps,
  tuner framework, subgroup taxonomy, and cross-dispatch fusion decisions.
  Public API snapshots were refreshed for the resulting shared surfaces.
- **Frozen/public API snapshots refreshed.** Snapshots now reflect the
  intentional 0.6 contract surface for borrowed output reuse, borrowed async
  dispatch, subgroup visitors, required lowering implementations, categorical
  laws, and the current published public items for driver/wgpu/foundation/
  primitives/spec crates.
- **`vyre-foundation` program-shape analysis surface.** Public snapshots now
  include `program_shape_facts`, the reusable buffer-shape analysis used by
  optimizer passes and downstream cache consumers.
- **`Node::forever(body)`** helper in `vyre-foundation::ir::Node`. Linus
  principle  -  `forever` lowers to `Node::Loop { 0..u32::MAX, body }`,
  no new enum variant, no cascade of match arms. Persistent kernels
  use it.

## [0.6.0]  -  2026-04-19
(layered workspace: foundation → driver → ops; single inventory registration path)

### New in 0.6.0

- **Nine-crate layered workspace.** Extracted `vyre-foundation` (IR, wire format, visitor traits, extension resolvers), `vyre-driver` (registry, runtime, pipeline, routing, diagnostics), `vyre-driver-wgpu` (wgpu backend, buffer pool, bind-group cache, pre-recorded dispatch), `vyre-driver-spirv`, `vyre-ops` (stdlib dialects), from what was a single god-crate. `vyre` remains as a back-compat meta shim.
- **Machine-checked layer DAG.** `scripts/check_layering.sh` enforces R1–R3+R5 from `COMPUTE_2_0.md §3`: foundation has no driver/ops/backend deps, driver has no ops/backend deps, ops has no backend deps, reference has no backend deps. Cross-layer imports go DOWN only; violations fail CI.
- **True IR openness.** `Expr::Opaque` and `Node::Opaque` now round-trip through the wire format (tag `0x80`) via inventory-registered `OpaqueExprResolver` / `OpaqueNodeResolver`. Validator, optimizer passes, and visitor adapters all honour Opaque explicitly  -  no wildcard fallthrough remains in foundation transforms.
- **Single op registration path.** `inventory::submit!{OpDefRegistration::new(...)}` is THE way to publish an op. `OpSpec` surface is gone; `DialectRegistry` is the frozen index.
- **Zero-alloc dispatch hot path.** `bound_handles` returns `SmallVec<[_; 8]>`, bind groups cache keyed by bound-buffer identity, buffer pool recycles power-of-two allocations across dispatches.
- **`vyre-reference` Memory** replaced `HashMap<String, Buffer>` with `BufferMap` (`SmallVec<[(Arc<str>, Buffer); 8]>`)  -  branch-predicted inner-loop lookups, no per-access SipHash, no per-name `String` allocs. `LocalSlots` interns via `FxHashMap<Arc<str>, _>`.
- **Invariant catalog truthful.** Every descriptor in `vyre-spec/src/invariants.rs` now references a real file at `conform/vyre-conform-enforce/tests/invariants.rs`, enforced by `scripts/check_invariant_paths_exist.sh`.
- **Ratchet CI gates.** `scripts/check_no_string_wgsl.sh` caps Law-B string-WGSL violations at 54 and `naga::front::wgsl::parse_str` sites at 84. `scripts/check_warning_budget.sh` caps workspace warnings at 921. Each gate decreases only; regression fails CI.

### Breaking

- Op registrations must go through `vyre-driver::registry::OpDefRegistration`. Consumers using legacy `OpSpec` surface must migrate.
- `vyre-core/src/` is reduced to `lib.rs` (meta-shim re-exports). Files that reached into `vyre::ir::transform::...` etc. must import from `vyre_foundation` directly  -  the meta-shim still provides the `vyre::ir::X` paths for surgec/pyrograph/warpscan consumers.

## [0.5.0]  -  2026-04-19
(substrate-neutral IR: open extensions + conform certificates)

### New in 0.5.0 final

- **VIR0 wire-format spec published**  -  `vir0-spec.md` at repo root declares the wire format stable across 0.5.x, reserves the `0x80..=0xFF` tag range for third-party extensions in perpetuity, and documents conformance requirements for non-Rust bindings (Phase 22).
- **Bytes extraction validation**  -  `BufferDecl::with_bytes_extraction(true)` opt-in relaxes V013 on load/store of `DataType::Bytes` buffers for legitimate bytes-producing ops like `decode.base64`, `compression.lz4_decompress`, and the decoder family. `Signature` gained `#[non_exhaustive]` + `bytes_extraction` field + `bytes_extractor` constructor (Phase 3).
- **Canonicalized 7 primitive programs** to match the emit-asserted WGSL shape  -  `abs_diff` routes through `max(a,b) - min(a,b)`, `div` / `mod` wrap in zero-guard `select`, `logical_not` uses boolean-style `select(x==0, 1, 0)`, `negate` uses two's-complement `~a + 1`, and `shl` / `shr` zero-guard shifts `>=32` (Phase 2).
- **photonic backend crate** lives in `backends/photonic/` as a registered non-dispatching substrate with `supports_dispatch = false`  -  proves the three-substrate surface claim today, while photonic compute remains future work.
- **SPIR-V backend skeleton** in `backends/spirv/`  -  `SpirvBackend::emit_spv` consumes `naga::Module` built by the shared builder family and calls `naga::back::spv::write_vec`, giving vyre a second real compute-capable backend alongside wgpu (Phase 14).
- **Conform crates scaffolded**  -  `vyre-conform-spec` (witness sets + composition laws), `vyre-conform-generate` (proptest-style shrinking minimizer), `vyre-conform-enforce` (algebraic-law prover over witness pairs), `vyre-conform-runner` (CLI + Certificate schema) at `conform/vyre-conform-*` (Phase 17).
- **rules/op/ certificate library**  -  5 op certs (`decode.base64`, `compression.lz4_decompress`, `match.dfa_scan`, `string_matching.aho_corasick_scan`, `graph.bfs`) plus `SCHEMA.md` defining op_id / signature_blake3 / allowed_backends / witness_set_blake3 / laws metadata (Phase 4).
- **NFA bytecode micro-interpreter fully retired**  -  the remaining `nfa_scan` kernel was deleted in the 2026-04-19 zombie sweep, README/CHANGELOG/VISION cross-references scrubbed, scan and lexical ops now compose in vyre IR end-to-end (Phase 7).
- **Docs**  -  `docs/THESIS.md`, `docs/ARCHITECTURE.md`, `docs/memory-model.md`, `docs/targets.md`, `docs/wire-format.md` authored as load-bearing spec.

### Breaking

- `Signature` is `#[non_exhaustive]`  -  out-of-crate literal construction must move to `Signature::bytes_extractor(...)` or `Signature { inputs, outputs, attrs, ..Signature::default() }` equivalent.
- `BufferDecl` gained the `bytes_extraction: bool` field; source-compatible through the builder API (`::read`, `::output`, `::read_write`, `::storage`, `::workgroup`), but direct struct literals must set it.

### Fixed

- `all_primitives` arithmetic / bitwise assertions now see the canonical WGSL shapes emitted by `naga_emit`  -  `abs_diff`, `div`, `mod`, `logical_not`, `negate`, `shl`, `shr` all validate against the assertion set.
- V013 no longer blocks valid decode / decompress flows that read and write typed `Bytes` buffers.
- README no longer describes a bounded `nfa_scan` bytecode micro-interpreter; it was deleted.

### Substrate (Claude)
- core: structured `Diagnostic` API with stable `E-*` / `W-*` codes,
  rustc-style human render, JSON round-trip for LSP / CI integration
  (A-C1b).
- wire: rev 3 framing  -  schema version bumped to 3 with structured
  `Error::VersionMismatch { expected, found }` replacing string-based
  version mismatch (A-C2).
- dialect: op versioning + migration table (`Migration`,
  `Deprecation`, `AttrMap`, `Semver`) via `inventory::submit!`; chain
  resolution + deprecation diagnostics (A-C2b).
- perf: `BENCHMARKS.md` performance contract  -  10 targets, numerical
  stability per-op ULP bounds, regression gate spec (A-C14b).
- optimizer: `AdapterCaps` + `PassCtx` + `AnalysisCache`; typed-error
  conversion from `PassSchedulingError` to `Diagnostic` (A-C7b part 1).
- core: runtime introspection API  -  `dialects()`, `ops()`, `backends()`,
  `lowerings()`, `coverage_matrix()` (A-C11b).
- docs: op-id stability catalog + regen-on-demand gate
  (`docs/catalogs/op-id-catalog.md`); coverage matrix + regression gate
  (`docs/catalogs/coverage-matrix.md`) (A-B4d, A-C11b).
- scripts: layout / file-size / mod.rs-size / prelude / readmes CI
  law scripts under `scripts/laws/` (A-C11c part 1).

### Dialects (Gemini A)
- core: dialect foundation types  -  `OpDef`, `LoweringTable`,
  `DialectRegistry`, `InternedOpId`, `BackendRegistration` (A-B0).
- core: every Cat C intrinsic migrated to `naga::Module` builders  - 
  91 ops, zero shader assets remain in op trees (A-B1).
- core: primitive Cat A ops migrated; KAT coverage for 7 previously-
  missing programs (A-B2).
- core: `io` dialect  -  4 Cat C zero-copy intrinsics
  (`io.dma_from_nvme`, `io.write_back_to_nvme`, `mem.zerocopy_map`,
  `mem.unmap`) registered with no backend opt-in (B-B3 scope).

### Backends (Gemini B)
- wgpu: dispatch via `DialectRegistry.get_lowering`  -  `OpSpec::intrinsic`
  read path removed (B-B1).
- wgpu: `impl Executable` + `impl Compilable` for `WgpuBackend` with
  `WgpuIR` progressive-lowering artifact (B-B5).
- reference: `dialect_dispatch` module routes op ids through
  `DialectRegistry.get_lowering(CpuRef)` (B-B4).

### Performance (Gemini C)
- wgpu: lock-free `BufferPool` via crossbeam; `PrerecordedDispatch`
  pre-recording (C-B1).

### Pre-existing (landed earlier in the cycle)
- core: blake3 fingerprinting for IR stability and cache invalidation (MOD-008)
- core: arena-backed reference interpreter (P-2)
- runtime: zero-copy output-slice readback (P-5)
- runtime: streaming chunked dispatch (P-7)
- validator: tightened atomic indexes, fma/select typing, mixed arithmetic typing, and u64 bitwise-unary acceptance (VAL-001..004)
- conform: widened overflow-contract surface for primitive arithmetic regression coverage (CONF-001)
- conform: added build-scan regression coverage for generated operation metadata (CONF-002)
- wire: added depth-cap regression coverage for hostile nested IR blobs (EDGE-001)

### Changed
- `vyre-conform::specs::primitive` now walks `vyre::ops::registry` for every `primitive.*` op and builds specs from core metadata plus normalized `rules/kat/primitive/<family>/<op>.toml` vectors. Legacy per-op modules that were not present in the core registry, including `logical_and`, `logical_or`, `logical_xor`, `logical_nand`, `logical_nor`, `avg_floor`, `wrapping_neg`, and `popcount_sw`, were removed rather than kept as conform-only specs.

## [0.4.0-alpha.2]  -  2026-04-17

### Added
- Architecture and process contracts were formalized with `ARCHITECTURE.md`, `rules/SCHEMA.md#kat`, and `docs/PRIMITIVES.md`, giving a stable contributor contract for frozen traits, op classification, and community rulesets.
- New publishable package structure was established: `vyre-spec` (`0.1.0`) and `vyre-build-scan` (`0.1.0`) plus release-ready crate metadata for the workspace surface.
- Conformance foundations landed for this release with canonical `CpuOp` CPU reference plumbing in `core::ops::cpu_op`, `conform` pipeline cleanup, and the move of `reference` into `vyre` so evaluator semantics and wire-era tooling are co-located.
- Benchmark and evidence publishing pipeline landed: `primitives_showcase` entrypoint, `benches/RESULTS.md`, and synchronized benchmark presentation in README + book.

### Changed
- DeepPerf wave cleanup converted temporary tree-gen and generated-cruft artifacts into a stable one-file-per-op structure, including conform command/layout simplification and generated module deduplication.
- Core/conform import surfaces and type contracts were adjusted for category and registry stability, including `Category`/`IntrinsicTable` migration into `vyre-spec` and elimination of brittle cross exports.
- Documentation and validation semantics were tightened: `Fix:`-prefixed actionable diagnostics, contract-first doc language, and release-oriented invariant text for affected public surfaces.
- Package and build metadata was harmonized for publishability and release continuity.

### Fixed
- Fixed immediate compile/dependency coupling regressions from the prior refactor wave by removing dead or misleading generated surfaces and restoring stable compile boundaries.
- Fixed benchmark evidence drift by rebaselining published values from `benches/RESULTS.md` and aligning user-facing benchmark tables.
- Fixed stale release-state items by auditing all open coordination entries and refreshing statuses with explicit reopen criteria.

### Perf
- DeepPerf benchmark capture completed for primitive ops across 1K/10K/100K/1M element sizes with CPU and GPU end-to-end timings, crossover annotations, and the full 48-op table in `benches/RESULTS.md`.
- Preserved the end-to-end performance gate by excluding structural hacks and ensuring benchmark coverage remains tied to committed results data.
- Captured remaining hotspot context for future release polish (`gcd`, `lcm`, and uncovered KAT boundary classes) in coordination notes for targeted follow-up.

## [0.4.0-alpha.1]  -  previous

### Added
- Workspace merge of `vyre` core and `vyre-conform` into a single workspace.
- `SANTH_STANDARD.md` and `template_op.rs`  -  standardized contributor template for adding new ops (8fa6ab6, 436264b).
- `automod` wired across all op categories (bitwise, math, reductions, data_movement, string, scan, sort, encode, stats, buffer, compiler_primitives, rule, decode, match_ops, string_similarity, graph, workgroup, security_detection, hash) (c6af953, c4ab1f7, a39a9c5).
- CI workflow for check + clippy + doc (3c57a49).

### Changed
- Core consolidated from ~2000 files down to 1117 files with 0 compile errors (0956373, 5b6e1e5, 436264b).
- Conform merged and consolidated from 3645 files down to 883 files with 0 compile errors (09a6496).
- GPU feature gates stripped from conform; conform now assumes GPU is always available (ac760a8, b1b7991).

### Fixed
- Original 80-entry op registry restored after agent overwrites (b1b7991).
- Tree-gen damage consolidated and reverted where it broke the module graph (ade08d5, c91ad8c, 35f7342, dd71607).
