# Mukund direct worklist for Vyre 0.4.1 and Weir 0.0.1

Owner: Mukund
Scope: work for Mukund only. This is not the shared agent plan.

## Objective

Make the release path prove a GPU-first C parser plus CUDA-first Vyre execution can beat Tree-sitter-class CPU parsing on a real Linux subsystem, while removing CPU parser dependence from release semantics.

## Rules for this work

1. Do not add CPU parser fallback.
2. CPU and Tree-sitter are baseline or oracle only.
3. Do not hardcode to one fixture file.
4. Every optimization needs a measurable counter or benchmark artifact.
5. Prefer resident GPU state, packed batches, fused dispatch, and sparse readback.
6. Treat extra host copies, repeated submits, and unnecessary readbacks as correctness bugs.

## Work item 1: Select and freeze the Linux subsystem corpus

Files to touch:

release/corpora/c/linux-subsystem.toml
xtask/src/c_parser_corpus.rs

What to do:

Choose the easy first subsystem for 0.4.1, preferably a small Linux kernel subsystem with many normal C files and limited macro hostility. The goal is not full Linux yet. The goal is a real subsystem that Tree-sitter parses and Vyre can parse end-to-end faster.

Acceptance:

The corpus manifest records exact root path, include policy, file count, total bytes, ignored-file reasons, and SHA256 of the file list.

## Work item 2: Define the Tree-sitter baseline exactly

Files to touch:

xtask/src/c_parser_corpus.rs
vyre-bench/src/cases/c_parser_linux.rs

What to do:

Make the baseline measure parse-only Tree-sitter time over the exact same files and bytes as the Vyre GPU parser. Do not include file discovery in either side. Do not include unrelated AST walking unless Vyre is doing the same semantic extraction.

Acceptance:

Artifacts include tree_sitter_parse_ns, tree_sitter_files, tree_sitter_bytes, tree_sitter_nodes, and tree_sitter_errors.

## Work item 3: Make input packing one contiguous upload

Files to touch:

vyre-frontend-c/src/pipeline.rs
vyre-frontend-c/src/tu_host.rs
xtask/src/c_parser_corpus.rs

What to do:

Pack all selected C files into one resident byte arena plus one file-offset table. The parser should not upload each file independently in the measured path.

Acceptance:

Measured artifacts show one logical corpus upload, zero host token stream upload, and no per-file parser dispatch loop.

## Work item 4: Add packed file metadata for GPU parsing

Files to touch:

vyre-frontend-c/src/pipeline.rs
vyre-frontend-c/src/gpu_file_table.rs

What to do:

Add or finish a compact GPU file table: file_start, file_len, line_start_base, include_flags, and output ranges. This should let kernels map byte offsets back to files without host intervention.

Acceptance:

Parser outputs can identify function/call/statement records by file id and byte range without a CPU-side remap pass.

## Work item 5: Fuse lexer plus statement boundary passes where safe

Files to touch:

vyre-frontend-c/src/pipeline/vast_pg.rs
vyre-frontend-c/src/pipeline.rs
vyre-lower/src/descriptor.rs

What to do:

Reduce submit count by fusing lexer classification and statement-boundary production where dependencies permit. Do not fall back to an unfused path silently. If fusion is impossible, emit a blocker artifact.

Acceptance:

Artifacts include resident_vyre_parse_resident_fused_statement_dispatches greater than zero and resident_vyre_parse_resident_fused_statement_host_submits less than or equal to fused dispatches.

## Work item 6: Suppress intermediate readbacks

Files to touch:

vyre-frontend-c/src/pipeline.rs
xtask/src/c_parser_corpus.rs

What to do:

Keep lexer, statement, and AST-window intermediates resident on GPU. Only final semantic records should read back in the measured path.

Acceptance:

Artifacts report positive resident_vyre_parse_resident_lexer_readback_suppressed_bytes, resident_vyre_parse_resident_statement_readback_suppressed_bytes, and resident_vyre_parse_resident_ast_readback_suppressed_bytes.

## Work item 7: Batch AST windows instead of submitting one by one

Files to touch:

vyre-frontend-c/src/pipeline.rs
vyre-driver-cuda/src/pipeline.rs
vyre-driver/src/backend/compiled_pipeline.rs

What to do:

Convert AST-window parse work into a resident batch. The host should submit batches, not individual windows.

Acceptance:

When resident_vyre_parse_ast_windows is greater than one, resident_vyre_parse_host_submit_count is lower than resident_vyre_parse_gpu_dispatch_count.

## Work item 8: Add C parser throughput counters

Files to touch:

xtask/src/c_parser_corpus.rs
vyre-bench/src/runner/execute/metric_keys.rs
xtask/src/vyre_weir_release_gate.rs
xtask/src/release_completion_audit.rs

What to do:

Add stable metrics for bytes/sec, files/sec, functions/sec, calls/sec, host submits, GPU dispatches, upload bytes, readback bytes, and speedup versus Tree-sitter.

Acceptance:

Release gates fail if a required counter is missing or non-finite.

## Work item 9: Set the release performance threshold

Files to touch:

xtask/src/vyre_weir_release_gate.rs
xtask/src/release_completion_audit.rs
release/plans/vyre-0.4.1-release-plan.md

What to do:

Pick the first release threshold honestly but aggressively. I recommend requiring at least 10x Tree-sitter speedup on the selected subsystem for 0.4.1, then tracking 100x and 1000x as stretch gates until the architecture earns them.

Acceptance:

The release gate has explicit fields for required_speedup, observed_speedup, and whether the threshold is release-blocking.

## Work item 10: Remove CPU parser language from release semantics

Files to touch:

vyre-frontend-c/src/pipeline.rs
vyre-frontend-c/src/tu_host.rs
README.md
release/plans/vyre-0.4.1-release-plan.md

What to do:

Search for wording that implies CPU parser fallback, CPU parser release path, or graceful CPU degradation. Replace it with explicit baseline/oracle wording. Runtime failures should be loud if GPU execution is unavailable.

Acceptance:

Docs and artifacts distinguish GPU parser execution from CPU baseline measurement.

## Work item 11: CUDA megakernel resident batch pressure

Files to touch:

vyre-driver-cuda/src/pipeline.rs
vyre-driver-cuda/src/backend/cuda_graph.rs
vyre-driver-cuda/src/backend/cuda_graph_replay.rs

What to do:

Make the CUDA path prefer resident graph replay and grouped readback for parser and conditional workloads. Avoid per-item synchronize. Avoid per-item output allocation in the measured path.

Acceptance:

Batch metrics show grouped launch/replay, one grouped readback phase, and no measured per-item output clear except required counters.

## Work item 12: Descriptor-level optimization evidence

Files to touch:

vyre-lower/src/rewrites/
vyre-lower/src/analysis/
vyre-bench/src/cases/

What to do:

For every parser-relevant lower rewrite, add a before/after metric that proves dispatch count, bytes moved, shared-memory pressure, register pressure, or instruction count improved.

Acceptance:

No parser-critical rewrite is only covered by correctness tests. It must have benchmark evidence or a release blocker explaining why it is disabled.

## Work item 13: Alias-aware pass upgrades for parser IR

Files to touch:

vyre-foundation/src/optimizer/passes/
vyre-lower/src/analysis/
vyre-lower/src/rewrites/

What to do:

Use reaching-def, points-to, and alias information to make DSE, store-to-load forwarding, loop fusion, and loop fission less conservative on parser kernels.

Acceptance:

At least one parser workload shows reduced memory traffic or fewer dispatches from alias-aware optimization.

## Work item 14: Release artifact schema cleanup

Files to touch:

xtask/src/c_parser_corpus.rs
release/schema/

What to do:

Keep one schema for parser benchmark artifacts. Do not let report, manifest, and throughput JSON drift into three independent contracts.

Acceptance:

Every required parser metric exists in exactly one documented schema and the gate reads the same names.

## Work item 15: Failure-mode corpus

Files to touch:

release/corpora/c/linux-subsystem-negative.toml
xtask/src/c_parser_corpus.rs

What to do:

Add malformed but realistic files: unterminated comments, macro-heavy headers, weird attributes, inline asm, huge initializer lists, and conditional compilation gaps. These are not release success corpus files. They are parser robustness probes.

Acceptance:

The negative corpus produces structured parser errors without panics, silent CPU fallback, or corrupt output records.

## Work item 16: Do not touch while I am editing

Files I am actively likely to touch:

vyre-driver-cuda/src/pipeline.rs
vyre-driver-cuda/src/backend/cuda_graph.rs
vyre-driver-cuda/src/backend/cuda_graph_replay.rs
xtask/src/c_parser_corpus.rs
xtask/src/vyre_weir_release_gate.rs
xtask/src/release_completion_audit.rs
vyre-frontend-c/src/pipeline.rs
vyre-frontend-c/src/pipeline/vast_pg.rs

Coordinate before editing these, because conflicts here will slow us down.

## Best first task for Mukund

Start with Work item 1 plus Work item 2. That gives us the exact battlefield: the corpus and the Tree-sitter baseline. Once those are fixed, every parser optimization has a concrete target and we stop arguing in abstracts.
