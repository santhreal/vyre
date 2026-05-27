# Vyre + The dataflow consumer release release plan

Date: 2026-05-05

Scope: `vyre` and the dataflow consumer. The C parser is explicitly excluded because that
work is owned separately for this release.

## Release thesis

This release is not "GPU is faster at GPU-shaped work." The release thesis is:

1. Vyre is the GPU execution stratum for branchy, conventionally CPU-only logic.
2. CUDA is the primary release path for speed.
3. WGPU is the correctness fallback and portability path.
4. The dataflow consumer proves Vyre is useful above matching: dataflow, IFDS, reachability, and
   proof extraction compose over Vyre without raw shader code.

## Release bar

Release-ready means:

1. A user can understand what Vyre is, what The dataflow consumer is, and how they compose.
2. CUDA has a concrete CPU-beating conditional-eval proof.
3. WGPU runs the same release suite correctly without pretending to own the speed
   claim.
4. The dataflow consumer has no missing public landing page, no fake implementation, and a real
   bridge from IFDS reachability to witness extraction.
5. The runtime path does not hide avoidable copies or per-finding rebuilds.
6. Claims in docs are narrower than measured reality, never broader.

## Current release facts

Measured on the local GPU machine during this pass:

1. CUDA release suite: `22 passed, 0 failed`.
2. WGPU release suite: `22 passed, 0 failed`.
3. CUDA condition eval proof: about `92x` over the scalar CPU baseline in the
   latest full release run.
4. CUDA bytecode dispatch: over `100x` in full-suite runs.
5. CUDA hashtable probe: about `45x` in the latest full release run.
6. CUDA batched multi-file condition eval: about `22x` in the latest full
   release run.
7. CUDA megakernel condition opcode: correct through the control-report hot path
   and roughly CPU parity on the 64k-slot release case.
8. WGPU condition eval is correct and can beat CPU on condition workloads, but
   WGPU remains the fallback because broader release-suite speed is inconsistent.

## Architecture decisions for this release

1. CUDA is the release speed backend.
2. WGPU is not allowed to block CUDA speed claims when it is correct but slower.
3. Megakernel remains strategically primary, but direct CUDA dispatch can remain
   the measured release path until the persistent queue beats direct dispatch on
   the release workloads.
4. Foundation microbenchmarks remain diagnostics. They are not allowed to make
   broad CPU-SOTA claims unless the measured workload actually supports that
   claim.
5. The dataflow consumer is a first-class wrapper crate, not a hidden Vyre module.

## End-to-end work plan

### Phase 1: Public contract cleanup

Status: in progress.

Tasks:

1. Create The dataflow consumer `README.md` because `Cargo.toml` points to it and it was missing.
2. Create The dataflow consumer `VISION.md` so architectural decisions have a source of truth.
3. Update Vyre release docs so CUDA-primary/WGPU-fallback is explicit.
4. Document the condition-eval proof honestly.
5. Remove or rewrite stale release text that references obsolete crate names or
   old publish order.

Done in this pass:

1. This cross-project plan was created.
2. The dataflow consumer public docs are being created next.

### Phase 2: Conditional-eval proof hardening

Status: in progress.

Tasks:

1. Keep `conditions.yara_like.eval.1m` in the release suite.
2. Keep the CPU baseline scalar, branchy, and CPU-favorable.
3. Avoid per-run host upload cost on CUDA using resident resources.
4. Avoid per-rule metadata loads when metadata is constant for the scanned file.
5. Add a second condition workload that evaluates multiple files per rule batch,
   because rule engines are rarely one-file-only at scale.
6. Add a sparse-fired-rules output mode so readback is proportional to findings,
   not rule count.

Done:

1. Resident buffers are used when the backend supports them.
2. CPU oracle output is precomputed during prepare.
3. File metadata constants are encoded as IR literals.
4. Single-file condition eval now emits sparse fired-rule IDs.
5. Batched condition eval exists as `conditions.yara_like.batch.16x64k`.
6. Batched condition eval uses packed rule descriptors instead of one buffer per
   descriptor field.

### Phase 3: Megakernel release path

Status: in progress.

Tasks:

1. Compare direct CUDA vs megakernel CUDA on the full external-buffer condition
   workload.
2. Add sparse result append for rule firing in the persistent megakernel queue.
3. Keep direct dispatch as fallback only when the megakernel path is measurably
   worse for a specific workload.
4. Remove shape-cache ambiguity permanently across CUDA and WGPU metadata paths.

Done:

1. `runtime.megakernel.condition.64k` executes a real branchy condition opcode
   through the finite megakernel builder.
2. CUDA control-report readback avoids ring/debug/IO copies by honoring
   zero-length output ranges.
3. The release suite includes the megakernel condition case for CUDA and WGPU.

### Phase 4: The dataflow consumer completeness

Status: in progress.

Tasks:

1. Keep `solve_cpu` as the exact oracle for IFDS reachability.
2. Keep `ifds_gpu_step` as a real Vyre `Program`, not a shader escape hatch.
3. Provide a bridge from exploded IFDS triples to statement-id reachability masks.
4. Make witness extraction reusable across many findings by preparing reverse CSR
   once per graph.
5. Add a public example: build small IFDS graph, solve, decode to statement mask,
   extract witness path.
6. Add adversarial tests for malformed CSR, out-of-bounds block maps, sanitizer
   crossing, cycles, and depth cap.

Done:

1. `exploded_reachability_to_statement_mask` exists.
2. `prepare_witness_graph` and `extract_path_prepared` exist.
3. Witness tests cover sanitizer rejection, per-source masks, depth cap, OOB
   seeds, exploded decode, and prepared graph reuse.
4. `examples/ifds_to_witness.rs` runs the public IFDS-to-witness path.
5. `examples/witness_many.rs` measures repeated witness extraction over one
   prepared graph.
6. Weir's normal Vyre dependencies are versioned for crates.io packaging instead
   of path-only workspace dependencies.

### Phase 5: Performance work remaining

Status: open.

Tasks:

1. Add sparse fired-rule output to condition eval.
2. Add packed rule descriptors to reduce condition benchmark bindings and memory
   streams.
3. Replace branch-heavy condition IR with select/bitmask form where it improves
   CUDA occupancy.
4. Add batched multi-file condition eval and measure crossover points.
5. Move repeated The dataflow consumer proof extraction callers to `PreparedWitnessGraph`.
6. Add The dataflow consumer microbenchmarks for witness extraction over many findings on one CSR.

Done:

1. Packed descriptors are implemented for the batched condition workload.
2. `examples/witness_many.rs` measures 2,048 witnesses over one prepared graph.
3. `runtime.megakernel.condition.64k` executes a real custom condition opcode
   through the megakernel and reports only control on the hot path.
4. CUDA direct dispatch and cudaGraph now honor `output_byte_range`, including
   zero-length readback and nonzero byte offsets.

### Phase 6: Conformance and docs

Status: open.

Tasks:

1. Add a doc test or example for the The dataflow consumer witness pipeline.
2. Add a release note that clearly says CUDA owns speed, WGPU owns fallback.
3. Audit every public crate README for claims that exceed measured behavior.
4. Ensure every publishable crate has a landing page and accurate crate metadata.
5. Update publish order to match current workspace crates.

### Phase 7: Final release readiness

Status: in progress.

Tasks:

1. Run publish dry-run for publishable crates.
2. Read every release note and README claim against measured output.
3. Only then tag the release.

Done:

1. Full CUDA release suite passes with 22 cases.
2. Full WGPU release suite passes with 22 cases.
3. Focused The dataflow consumer witness tests pass.
4. CUDA output-range driver contracts pass for direct dispatch and cudaGraph.
5. Publish metadata audit passes for all publishable Vyre crates.
6. Normal publish dependencies no longer contain path-only internal deps.
7. `./cargo_full publish --dry-run` passes for the registry-independent prefix:
   `vyre-lints`, `vyre-macros`, and `vyre-spec`.
8. The publish process documents the crates.io dependency-index boundary for
   `vyre-foundation` and later crates.
9. The dataflow consumer `./cargo_full check -p dataflow consumer` passes after publish-dependency cleanup.
10. The dataflow consumer focused witness tests pass: 13 passed, 0 failed.
11. The dataflow consumer publish dry-run reaches only the expected registry boundary:
    `vyre = "^0.6.0"` is not on crates.io until the Vyre alpha is published.

## Non-negotiable release claims

These claims are allowed:

1. Vyre evaluates branchy rule-like conditions on CUDA faster than a strong CPU
   baseline.
2. Vyre has a WGPU fallback that runs the same release workload correctly.
3. The dataflow consumer composes dataflow primitives over Vyre IR and exposes CPU oracles for
   correctness.
4. The dataflow consumer can decode exploded IFDS reachability into statement witness masks.
5. Vyre executes megakernel condition opcodes on CUDA through the control-report
   hot path and avoids redundant non-control readback.
6. CUDA and WGPU both pass the 22-case release suite.

These claims are not allowed yet:

1. WGPU is faster than CPU for conditional eval.
2. Megakernel is the measured default speed path for every release workload.
3. The dataflow consumer is a complete source-language analyzer.
4. The C parser is complete.

## Next execution order

1. Run the Vyre alpha publish loop in topological order from `docs/RELEASE.md`.
2. After each Vyre crate is indexed, run the next crate's dry-run and publish.
3. Run The dataflow consumer dry-run after `vyre 0.6.0-alpha.1` and its dependencies are
   indexed.
4. Re-run CUDA, WGPU, and The dataflow consumer witness suites on the published alpha consumer
   path.
