# Mukund Optimization Handoff

Scope: work for Mukund to execute while Codex continues parser, CUDA megakernel, release gates, and repository consolidation.

Objective: make Vyre 0.4.1 and Weir 0.0.1 prove GPU-first C parsing plus megakernel condition evaluation can beat Tree-sitter-class CPU parsing on the frozen Linux lib subsystem corpus, with CUDA as the release path and WGPU as fallback.

## 1. Token tile persistent CUDA lexer

Files:
- vyre-driver-cuda/src/codegen.rs
- vyre-driver-cuda/src/codegen/descriptor_gate.rs
- vyre-frontend-c/src/pipeline/dispatch.rs
- vyre-frontend-c/src/pipeline/buffers.rs

Task:
Replace one-shot token dispatch with persistent resident token tiles. Keep the full corpus in one contiguous device arena, split only by GPU tile metadata, and avoid host-side per-file loops. The kernel should consume {arena_base, file_span_table, tile_table} and emit token windows in one CUDA graph-captured pipeline.

Success evidence:
- Resident parse path has no per-file CUDA launch.
- Token throughput improves on release/corpora/linux-subsystem.toml.
- Metrics expose token_tiles, resident_arena_bytes, token_kernel_replays, and launch_count.

## 2. AST window compaction and span-local writeback

Files:
- vyre-frontend-c/src/pipeline/vast_pg.rs
- vyre-frontend-c/src/pipeline/span_validate.rs
- vyre-frontend-c/src/pipeline.rs
- vyre-libs/src/parsing/c/preprocess/gpu_pipeline.rs

Task:
Compact AST node emission into span-local windows before global writeback. Remove scattered global atomics where possible. Use per-block prefix counts, compact into a node-window buffer, then perform one coalesced global append per block or warp group.

Success evidence:
- AST coverage stays identical or improves.
- Global atomics per parsed byte drops.
- Median resident parse time improves on the Linux lib corpus.

## 3. Macro-state cache for translation-unit preparation

Files:
- vyre-frontend-c/src/tu_host.rs
- vyre-frontend-c/src/tu_host/preprocess.rs
- vyre-frontend-c/src/tu_host/system_includes.rs
- xtask/src/c_parser_corpus.rs

Task:
Cache macro expansion state by include graph fingerprint and define-set fingerprint. Repeated Linux headers must not rebuild macro state for every source file. The cache must be bounded, keyed by canonical include path metadata plus define hash, and must fail loudly on stale/invalid inputs.

Success evidence:
- include_cache_hits and macro_state_cache_hits are positive on the release corpus.
- No silent fallback to CPU parsing.
- Missing includes remain hard errors with actionable paths.

## 4. Include arena deduplication

Files:
- vyre-frontend-c/src/tu_host.rs
- vyre-frontend-c/src/pipeline/buffers.rs
- vyre-frontend-c/src/api/mod.rs
- xtask/src/c_parser_corpus.rs

Task:
Deduplicate repeated header bytes in the resident source arena. Multiple TUs including the same header should reference one arena slice. Preserve deterministic source spans, provenance, and diagnostics.

Success evidence:
- resident_include_dedup_bytes_saved is emitted.
- resident_source_bytes drops without reducing parsed file coverage.
- Diagnostics still map back to original file/include paths.

## 5. CUDA graph shape bucketing

Files:
- vyre-driver-cuda/src/backend/cuda_graph.rs
- vyre-driver-cuda/src/backend/dispatch.rs
- vyre-driver-cuda/src/pipeline.rs
- vyre-bench/src/cases/release_workloads.rs

Task:
Bucket CUDA graph captures by stable dispatch shape: program fingerprint, descriptor fingerprint, buffer layout, workgroup/grid shape, and static parameter slab size. Reuse graph replay across repeated C parser and megakernel workloads.

Success evidence:
- cuda_graph_records is small relative to cuda_graph_replays.
- Workload 10 emits positive graph replay counts.
- C parser corpus emits CUDA backend id and graph replay evidence.

## 6. Static launch-parameter slab

Files:
- vyre-driver-cuda/src/backend/host_dispatch.rs
- vyre-driver-cuda/src/backend/module_cache.rs
- vyre-driver-cuda/src/backend/dispatch.rs
- vyre-driver-cuda/src/pipeline.rs

Task:
Move stable launch parameters into a persistent GPU/host-pinned slab. Dispatch should patch only changed offsets, not rebuild/upload the whole parameter block each run.

Success evidence:
- cuda_static_param_uploads is lower than dispatch count after warmup.
- cuda_static_param_dispatches is positive.
- No per-dispatch heap allocation in the hot path.

## 7. Megakernel predicate specialization

Files:
- vyre-driver-cuda/src/codegen.rs
- vyre-driver-cuda/src/codegen/descriptor_gate.rs
- vyre-lower/src/rewrites
- vyre-bench/src/cases/megakernel_latency.rs
- vyre-bench/src/cases/release_workloads.rs

Task:
Specialize condition-eval megakernel predicate paths using descriptor facts: opcode family density, branch probability, constant masks, alias facts, and rule-shape fingerprints. Generate predicated CUDA code instead of divergent branch ladders where descriptor facts are stable.

Success evidence:
- Branch divergence proxy metric decreases.
- Megakernel latency benchmark improves.
- Speculation side compile cost is emitted separately from timed replay.

## 8. Alias-aware optimizer aggression

Files:
- vyre-foundation/src/optimizer/passes
- vyre-lower/src/rewrites
- /media/mukund-thiru/SanthData/Santh/libs/dataflow/weir/src

Task:
Use Weir reaching-def/points-to/alias facts to make loop fusion, loop fission, DSE, store-to-load forwarding, LICM, and CSE more aggressive. Do not duplicate the same transform at multiple IR levels; factor shared legality and proof logic.

Success evidence:
- New alias-positive fixtures show optimizations firing where structural checks previously blocked.
- Correctness/conformance remains equivalent.
- Benchmark evidence shows dispatch-time or instruction-count wins.

## 9. E-graph rewrite expansion

Files:
- vyre-foundation/src/optimizer
- vyre-lower/src/rewrites
- vyre-lower/tests
- vyre-bench/src/cases/release_workloads.rs

Task:
Move high-value algebraic rewrites into the egglog/e-graph substrate instead of maintaining isolated match-and-replace passes. Start with boolean normalization, arithmetic identities, comparison folding, common predicate extraction, and strength reduction.

Success evidence:
- Saturation produces fewer descriptor/program instructions than old passes on at least three release workloads.
- Compile-time budget is bounded and reported.
- Fallback to hand-written pass is not silent; it is a metric and release-gated.

## 10. Tree-sitter defeat harness hardening

Files:
- xtask/src/c_parser_corpus.rs
- xtask/src/release_benchmarks.rs
- xtask/src/vyre_weir_release_gate.rs
- xtask/src/release_completion_audit.rs
- release/corpora/linux-subsystem.toml

Task:
Make the comparison impossible to game. Count full resident end-to-end GPU parse time against Tree-sitter-class CPU parse time for the same frozen Linux lib corpus. Preserve file counts, byte counts, AST/token coverage, backend provenance, cache counters, and speedup thresholds.

Success evidence:
- At least 10 release workloads have benchmark evidence.
- Linux lib C parser corpus uses CUDA backend only for release pass.
- Median and best speedup gates are explicit.
- Prepared syntax, pipeline cache, include cache, CUDA graph replay, and no-native-fallback metrics are all release-gated.

Execution rule:
Do not add docs-only progress. Every task must land with a measurable performance or correctness signal in the release evidence path.
