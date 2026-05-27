# Mukund-only optimization handoff

This file is intentionally only the work I want Mukund to take while I continue the C parser, megakernel, runtime, and release path work.

## Rules for this handoff

- Do not add CPU fallback paths.
- Do not weaken parser correctness to win benchmark numbers.
- Do not optimize only one benchmark shape; every change should generalize to at least the frozen Linux subsystem plus one synthetic adversarial case.
- Prefer CUDA-first implementation. WGPU can follow the same contract after the CUDA path is proven.
- Every task below needs a before/after metric or a hard structural proof that a copy, allocation, dispatch, or synchronization was removed.

## 1. Kill host token-stream reupload after GPU lexing

Files:
- vyre-frontend-c/src/pipeline.rs
- vyre-libs/src/parsing/c/lex/lexer/core.rs
- vyre-libs/src/parsing/c/parse/structure.rs
- vyre-libs/src/parsing/c/parse/statement.rs
- vyre-bench/src/cases/c_parser.rs

Task:
Keep the lexer token buffers resident on GPU and pass resident resources directly into structure, statement, AST, and call extraction stages. The current prepared syntax path keeps the source haystack resident, but still exposes host_token_stream_upload_bytes. That number must go to zero for the prepared frozen subsystem path.

Success criteria:
- c_parser_resident_haystack_used == 1
- c_parser_measured_host_haystack_upload_bytes == 0
- c_parser_host_token_stream_upload_bytes == 0
- No CPU fallback branch added.

## 2. Fuse parser stages where the intermediate is single-consumer

Files:
- vyre-libs/src/parsing/c/lex/lexer/core.rs
- vyre-libs/src/parsing/c/parse/structure.rs
- vyre-libs/src/parsing/c/parse/statement.rs
- vyre-libs/src/parsing/c/parse/ast.rs
- vyre-frontend-c/src/pipeline.rs

Task:
Find parser stages that write an intermediate buffer consumed exactly once by the next parser stage. Replace those pairs with a fused CUDA descriptor/kernel where the intermediate stays in registers or shared memory when physically possible.

Initial targets:
- token classification plus declaration/function boundary detection
- function span extraction plus call scan seed generation
- statement boundary scan plus AST node emission

Success criteria:
- Fewer parser dispatches for the frozen subsystem path.
- Lower parser GPU global-memory bytes per source byte.
- No loss in diagnostic detail.

## 3. Add CUDA persistent parser graph for the full prepared syntax pipeline

Files:
- vyre-driver-cuda/src/backend/graph.rs
- vyre-driver-cuda/src/backend/dispatch.rs
- vyre-driver-cuda/src/backend/compiled.rs
- vyre-frontend-c/src/pipeline.rs
- vyre-bench/src/cases/c_parser.rs

Task:
Make the prepared C parser run as one persistent CUDA graph replay after warmup, including resident source, lexer, parser, AST, call graph, and summary extraction. The benchmark should prove graph replay coverage across the complete parser path, not just individual kernels.

Success criteria:
- CUDA graph replay count matches prepared parser submissions.
- Host submit count drops for repeated prepared parses.
- No graph rebuild on identical frozen corpus input.

## 4. Replace parser global atomics with prefix-sum allocation where counts are dense

Files:
- vyre-libs/src/parsing/c/parse/structure.rs
- vyre-libs/src/parsing/c/parse/inline_asm.rs
- vyre-libs/src/parsing/c/parse/ast.rs
- vyre-lower/src/rewrites

Task:
Current parser extraction paths use initialized read-write atomic counters. For dense outputs, move to count plus prefix-sum plus scatter. Keep atomics only for genuinely sparse or adversarial outputs.

Success criteria:
- Lower atomic operation count in parser metrics.
- Equal output counts on frozen subsystem and adversarial macro-heavy fixture.
- Better throughput on high function-count files.

## 5. Specialize parser kernels by source layout and corpus shape

Files:
- vyre-frontend-c/src/api/mod.rs
- vyre-frontend-c/src/pipeline.rs
- vyre-driver/src/compile_cache.rs
- vyre-driver-cuda/src/backend/compiled.rs

Task:
Add specialization keys for prepared C syntax workloads: source byte length class, file count class, maximum file length class, and enabled parser outputs. This should produce stable compiled handles for the frozen subsystem without per-run dynamic branching.

Success criteria:
- Prepared parser reuses the same compiled handles across repeated runs.
- Dynamic branches in CUDA parser kernels are reduced or moved to specialization constants.
- Compile cache key includes the shape fields that affect generated code.

## 6. Build Tree-sitter comparison harness without making Tree-sitter part of the hot path

Files:
- vyre-bench/src/cases/c_parser.rs
- vyre-bench/src/runner/execute/metric_keys.rs
- vyre-bench/src/baselines

Task:
Add a CPU baseline harness that runs Tree-sitter only as an external comparison/oracle path. It must never be used by vyre parsing. It should report throughput, latency, AST node count, and corpus hash against the same frozen Linux subsystem.

Success criteria:
- Baseline metrics are clearly separated from vyre metrics.
- The benchmark can state speedup over Tree-sitter using identical corpus bytes.
- No runtime dependency from vyre parser code to Tree-sitter.

## 7. Add adversarial C parser corpora beyond the easy subsystem

Files:
- vyre-bench/src/cases/c_parser.rs
- vyre-libs/src/parsing/c/tests or existing parser fixture area
- release/corpora or existing corpus manifest area

Task:
Create at least two adversarial parser corpora that stress different failure modes: macro-heavy headers and declaration/function-pointer dense C. These are not the release headline workload, but they prevent hardcoding the easy subsystem.

Success criteria:
- Metrics report corpus hash, file count, source bytes, token count, AST node count.
- Parser handles all corpora without CPU fallback.
- Performance remains meaningfully above the CPU baseline on at least the easy subsystem and does not collapse on adversarial corpora.

## 8. Move lower/foundation duplicate optimization logic toward one canonical pass contract

Files:
- vyre-foundation/src/optimizer
- vyre-lower/src/rewrites
- vyre-lower/src/analyses

Task:
For passes that still exist twice as independent implementations, extract the algebraic legality check and profitability model into one shared contract. Foundation Program and lower KernelDescriptor can keep separate adapters, but the transform rule should not be duplicated.

Initial targets:
- CSE
- DCE
- LICM
- loop fusion
- loop fission
- loop unroll

Success criteria:
- Shared legality/profitability functions are used by both IR adapters.
- No re-export-only migration.
- Tests cover both Program and KernelDescriptor adapters with the same rule contract.

## 9. Expand alias-aware optimizer aggression

Files:
- vyre-lower/src/analyses
- vyre-lower/src/rewrites
- vyre-foundation/src/optimizer/passes

Task:
Use reaching-def, points-to, and alias information to make DSE, store-to-load forwarding, LICM, loop fusion, and loop fission less structurally conservative. The goal is fewer redundant memory operations before CUDA emission.

Success criteria:
- Alias analysis participates in each target pass decision.
- Benchmarks expose memory op count before/after.
- Conservative fallback only happens when alias state is genuinely unknown, not merely because the old structural check failed.

## 10. Add CUDA instruction-level backend optimizations for parser and condition workloads

Files:
- vyre-driver-cuda/src/backend/emit.rs
- vyre-driver-cuda/src/backend/ptx.rs
- vyre-driver-cuda/src/backend/scheduler.rs
- vyre-lower/src/analyses

Task:
Prioritize concrete CUDA backend wins that affect parser plus conditional workloads: predicated execution instead of divergent branches, vectorized loads for packed token/source streams, ldmatrix or cp.async where memory staging is regular, and instruction scheduling that separates dependent memory ops.

Success criteria:
- PTX pattern tests prove the intended instructions or branch shapes exist.
- Parser and condition benchmarks show dispatch-time or throughput improvement.
- WGPU fallback remains semantically compatible but CUDA is the release path.

## 11. Add megakernel work-queue compaction for sparse conditional eval

Files:
- vyre-runtime/src
- vyre-driver-cuda/src/backend/dispatch.rs
- vyre-bench/src/cases/megakernel_condition.rs
- vyre-lower/src/rewrites

Task:
For conditional eval where only a subset of rules remains live after early predicates, compact live work into a GPU-resident queue and continue in-kernel or in graph-replayed stages without host intervention.

Success criteria:
- Fewer wasted rule-lane executions on sparse workloads.
- No CPU scheduling loop.
- Benchmarks include dense, medium, and sparse rule activity.

## 12. Make benchmark evidence impossible to fake accidentally

Files:
- vyre-bench/src/runner/execute
- vyre-bench/src/cases/c_parser.rs
- vyre-bench/src/cases/megakernel_condition.rs
- release

Task:
Add metrics that prove the optimization actually ran: resident resource use, graph replay use, upload bytes, readback bytes, dispatch count, kernel count, atomic count, global memory bytes, and corpus hash. Any release benchmark missing these should fail the release gate.

Success criteria:
- Release parser and megakernel benchmarks include structural proof metrics.
- A CPU fallback, host reupload, graph miss, or corpus mismatch is visible as a hard blocker.
- Metrics are registered centrally and not silently dropped.

## Priority order

1. Kill host token-stream reupload.
2. Persistent CUDA graph for full prepared parser.
3. Parser stage fusion.
4. Tree-sitter comparison harness.
5. Adversarial corpora.
6. Alias-aware optimizer aggression.
7. CUDA backend instruction-level optimizations.
8. Megakernel work-queue compaction.
9. Optimization pipeline consolidation.
10. Release benchmark proof metrics.
