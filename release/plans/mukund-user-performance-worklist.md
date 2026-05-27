# Mukund-owned C parser + megakernel performance worklist

Context: this is only the work Mukund asked to personally take while Codex continues implementation. It is intentionally not the shared master plan.

## Goal

Help make Vyre 0.4.1 and Weir 0.0.1 release-ready by pushing the C parser + megakernel path toward significant Tree-sitter-class CPU baseline wins on a frozen Linux subsystem corpus, without adding CPU fallback paths.

## Mukund tasks

1. Freeze the release subsystem target.
- Pick the exact Linux subsystem directory for 0.4.1.
- Record the commit/hash or source snapshot used.
- Keep it easy enough for this release but large enough to be credible.

2. Define the benchmark contract.
- Decide the headline baseline: Tree-sitter C parsing per file over the frozen subsystem.
- Decide required reported metrics: total files, total bytes, total parse ns, median file parse ns, functions, calls, parse errors, speedup x1000.
- Keep CPU usage restricted to oracle/baseline only.

3. Build the Tree-sitter baseline runner target.
- Ensure Tree-sitter parses each file separately, not one concatenated mega-source.
- Count AST nodes, function_definition nodes, call_expression nodes, and parse-error presence.
- Emit stable JSON metrics that can be compared in release gates.

4. Define the correctness oracle for GPU C parsing.
- File count must match corpus manifest.
- Function records must match Tree-sitter function definitions for the supported subset.
- Call records must not exceed Tree-sitter call expressions.
- Output record byte sizes must be nonzero and aligned.

5. Identify and remove host traffic in parser measurement.
- Host uploads allowed: packed corpus buffers, offsets, file spans, static parser tables.
- Host downloads allowed: compact parser record buffers and small counters.
- No full-token stream download in the measured path.
- No host-side parse substitution.

6. Push CUDA graph/replay path for parser.
- Capture resident parser dispatch once.
- Replay for measured iterations.
- Record graph records, replays, cache hits, misses, evictions.
- Gate on zero graph evictions for release evidence.

7. Fuse parser stages where safe.
- Combine lex/classify + shallow syntax state where it reduces global memory round trips.
- Keep tables resident.
- Prefer SoA outputs and compact append buffers over per-byte verbose structures.

8. Wire parser into megakernel pressure path.
- Make C parser workload exercise the same resident CUDA dispatch infrastructure as condition/Weir paths.
- Avoid a bespoke parser-only fast path that cannot improve the rest of Vyre.

9. Stress optimizer passes against parser-shaped kernels.
- Include divergence-heavy token classification.
- Include offset-heavy file-span walking.
- Include append-buffer output patterns.
- Include table-driven state transitions.

10. Remove CPU parser escape hatches.
- Search for parser-side CPU fallback, software fallback, max TU size gates, and no-op resident parser markers.
- Replace with GPU-first hard failure if GPU configuration is broken.
- Do not add silent skips.

11. Add perf gates that fail honestly.
- Best and median C parser resident speedup over Tree-sitter must meet the selected release threshold.
- Host traffic metrics must remain bounded.
- CUDA resident batch/graph metrics must prove fast path usage.

12. Consolidate optimizer architecture.
- Prefer one canonical optimization contract shared between Program and KernelDescriptor levels.
- Do not keep duplicate passes that independently implement the same transform algebra.
- Make alias/reaching-def facts reusable by both high-level and lowered passes.

13. Add workload diversity beyond one parser case.
- At least 10 workloads for release evidence.
- Include C parser subsystem, conditional eval, conditional batch, Weir dataflow, megakernel queued batches, scatter/readback, coalesce/shared-memory/vector-pack analysis, alias-heavy optimizer cases, and rule-eval dispatch.

14. Compare against more than Tree-sitter where useful.
- Tree-sitter is the main parser baseline.
- Add Clang/libclang or another CPU parser only if it helps credibility and does not distract from the release path.

15. Keep benchmark honesty.
- No hardcoded workload shortcuts.
- No measuring an empty parser.
- No counting transfer-free GPU time as end-to-end unless separately labeled.
- Report resident GPU-only and end-to-end numbers separately.

## File targets likely relevant

- vyre-bench/src/cases/c_parser.rs
- xtask/src/c_parser_corpus.rs
- xtask/src/vyre_weir_release_gate.rs
- xtask/src/release_completion_audit.rs
- vyre-driver-cuda/src/
- vyre-frontend-c/src/
- vyre-lower/src/
- vyre-foundation/src/optimizer/
