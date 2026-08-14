# Changelog

All notable changes to vyre are documented here. Follows Keep a Changelog.

## [Unreleased]

### Added

- `xtask dup-scan --report [CRATE]` attributes duplicated lines to individual
  files and names the files each one shares shingles with. The gate could report
  that a crate exceeded its pin but not which copy to collapse, so a failure
  named a crate and left the search manual.
- `MegakernelBarrierPlan::from_groups` derives the global barrier count from the
  groups it is given. The count was open-coded at both construction sites, so a
  caller that re-grouped waves, such as the frontier memory-budget splitter,
  could report a barrier count that no longer matched its own groups.
- Grouped affine INT4 linear now provides a typed batched program builder that
  dequantizes each immutable weight tile once and reuses it across independent
  resident batch rows. Release evidence measures normalized per-inference
  latency.
- The vyre-safetensors adapter reads bounded safetensors headers and sharded
  indexes without reading tensor payloads, confines shard paths to the
  checkpoint root, rejects duplicate or unmapped tensors, validates
  caller-supplied dtype and shape requirements, and streams complete shards
  against an exact trusted BLAKE3 set before returning an immutable checkpoint
  identity.
- The runtime now owns immutable resources, reusable artifact instances, and
  mutable leased state through one budgeted residency boundary. Cold and warm
  admission, rollback, cancellation, generation-checked reset, completion,
  eviction, and manager destruction release resources without exposing stale
  state.
- ProgramGraph now composes reusable Programs through canonical typed value
  identities, explicit consumer and output ports, symbolic or concrete shapes,
  access and lifetime contracts, and validated state transitions. Its bounded
  VGR0 wire format embeds existing VIR0 Programs and rejects implicit casts,
  rank drift, alias conflicts, dangling state, malformed framing, and hostile
  counts before mutation.
- ProgramGraph now validates complete compositions and derives canonical
  topological schedules, inclusive value liveness, and deterministic
  interval-colored allocation plans. Invocation-local values reuse only
  nonoverlapping slots, while immutable weights, sequence-state generations,
  and caller-visible outputs retain dedicated storage.
- ProgramGraph now derives one versioned, domain-separated BLAKE3 identity from
  canonical topology and Programs, typed port contracts, artifact schema,
  validated model configuration, exact symbolic bindings, and verified
  immutable-weight identities. Mutable sequence-state contents are excluded, so
  cache growth reuses compiled artifacts while any executable or provenance
  change invalidates the key.
- The neural library now provides gated RMSNorm with float32 accumulation,
  source-dtype rounding, learned scaling, float32 SiLU gating, and exact
  last-dimension row isolation. The reference interpreter now executes
  canonical F16 and BF16 loads, stores, and F32 conversions with
  round-to-nearest-even semantics while preserving raw element-byte APIs.
- The neural library now provides last-dimension L2 normalization for grouped
  query and key heads. It accumulates sum-of-squares in F32, applies epsilon
  inside the canonical inverse-square-root contract, isolates rows exactly, and
  converts output once to F32, F16, or BF16.
- The neural library now executes floating channel-major depthwise causal
  convolution with exact left padding, masks, bias, SiLU, F16/BF16 conversion,
  and output truncation. A short-chunk route emits an explicit next-state
  generation whose outputs and tail match full prefill across arbitrary token
  partitions and reset.
- The neural library now executes recurrent gated delta attention with F32 Q/K
  normalization, grouped heads, scaled queries, exponential decay, sigmoid
  beta, F32 matrix-state continuation, source-dtype output, and explicit
  validated prior and next state generations.
- The neural library now composes F32 query and key normalization,
  cache-position partial rotary embedding, explicit query-to-KV head grouping,
  and dynamically bounded causal attention in one typed ProgramGraph. Prompt
  and cached-decode routes exclude future cache rows and support configurable
  head ratios, head widths, and rotary dimensions.
- The neural library now executes a reusable dense gated-MLP ProgramGraph with
  learned RMSNorm, checkpoint-native output-major gate and up projections, F32
  SwiGLU math, output-major down projection, and residual addition. F16, BF16,
  and F32 storage use F32 normalization, projection accumulation, activation,
  and residual arithmetic with source-dtype boundaries.
- The neural library now executes chunk-size-64 gated delta prefill with F32
  cumulative log-decay, a strict lower-triangular solve, initial-state
  correction, chunk output reconstruction, and explicit final matrix state.
  F16, BF16, and F32 inputs retain F32 internal math. Guarded rows in the final
  structural tile cannot read padding, change state, or appear in truncated
  output.

### Changed

- "Substrate" now names one thing: the GPU pass engine. Four modules that
  borrowed the word for unrelated concepts are renamed to what they are.
  `vyre_driver::speculation_substrate` is `vyre_driver::speculation_verdict`.
  `vyre_foundation::optimizer::fact_substrate::FactSubstrate` is
  `fact_cache::FactCache`, and the scheduler metrics it feeds are
  `fact_cache_reused` / `_recomputed` / `_invalidated`.
  `vyre-libs`' `substrate_catalog` is `builder_catalog`, and the two shared
  child-region ops it registers moved from the `vyre-libs::substrate::`
  namespace to `vyre-libs::builder::`. `vyre-libs`' `linear_algebra_substrate`
  was a three-line re-export of `math::linalg` and is gone; its two callers
  import from the owner directly.
- `structure-gate` resolves the workspace root from the environment at run time
  instead of from `env!("CARGO_MANIFEST_DIR")`. A shared cargo target directory
  hands one compiled gate binary to every worktree, so the baked path made a
  gate run inside a worktree report the main checkout's tree and hide the
  worktree's own findings.
- The dispatch seam moved below the composition library.
  `vyre_self_substrate::optimizer::dispatcher::OptimizerDispatcher` is
  `vyre_foundation::program_dispatch::ProgramDispatcher`, and `DispatchError`,
  `ResidentDispatchStep`, `ResidentReadRange`, `ResidentStaticBufferSet`, and
  `declared_dispatch_outputs` moved with it. The trait only speaks `Program`,
  buffers, and resident handles, so nothing about it was engine-specific, and
  every crate that dispatches a `Program` needed it above the pass engine.
  `vyre_driver_cuda::CudaOptimizerDispatcher` follows the trait and is
  `CudaProgramDispatcher`. The host-side byte marshalling that goes with the
  seam is `vyre_libs::dispatch_buffers` rather than a second module in
  `vyre-foundation`, because it delegates to `vyre_primitives::wire`, which
  foundation sits below. Its former `#[cfg(test)]` helpers
  (`f32_slice_to_le_bytes`, `decode_u32_input_aligned`,
  `decode_f32_input_aligned`, `read_u32s`, `read_f32s`) are unconditional now
  that their callers are in another crate. The scalar oracle that shared the old
  `dispatcher.rs` file is `vyre_self_substrate::optimizer::cpu_oracle`, still
  gated on `cpu-parity`. Public API snapshots for `vyre-foundation`,
  `vyre-libs`, `vyre-self-substrate`, `vyre-driver-cuda`, and
  `vyre-driver-wgpu` are refreshed for the move.

- `vyre_foundation::allocation::reserve_exact_cleared` is public and is now the
  single owner of the clear-then-refill reservation idiom. It was `pub(crate)`
  in `vyre-primitives`, so seven other crates hand-rolled it as
  `try_reserve(target - capacity())`. That form derives the additional count
  from capacity instead of length, so a warm buffer whose capacity is between
  `target / 2` and `target` stayed short and the following fill reallocated.
  Thirteen reserve sites across the drivers, the runtime, the wire decoder, the
  C preprocessor scratch path, and the tensor-train scratch retention now route
  through `reserve_exact_cleared` or, where the buffer keeps its contents,
  through the existing `allocation::try_reserve_vec_to_capacity`.
  `scripts/check_no_under_reserve.sh` runs in the `tree-rules` gate job.
- The standalone `vyre-intrinsics` package is gone. Its nine Category C
  hardware intrinsics now live in `vyre-primitives/src/hardware/` behind the
  crate's `hardware` feature, and every op id moved from
  `vyre-intrinsics::hardware::<op>` to `vyre-primitives::hardware::<op>`.
  `vyre-primitives` is the single Category C home; the region helper is
  `vyre_primitives::hardware::region`. The archived
  `docs/migration-vyre-ops-to-intrinsics.md` page is removed; the Category A
  and C classification rule it carried is owned by `docs/lego-block-rule.md`.
- The three emitter pattern-audit reports no longer carry inherent copies of
  the `PatternAudit` methods they already inherit. `NagaAuditReport` and
  `PtxAuditReport` drop `total_candidates`, `has_any`, `format_short`, and
  `is_clean`; `SpirvAuditReport` drops `total_findings`, `requires_action`,
  `format_short`, and `is_clean`. Each forwarded to the trait, and the three
  that shared a name with the trait method shadowed it. Callers import
  `vyre_lower::pattern_audit::PatternAudit` and use `finding_count` and
  `has_any`.
- `vyre-emit-ptx` publishes one vector memory fusion module instead of two.
  `patterns::vec_load_fusion` and `patterns::vec_store_fusion` were facades
  over the same detector whose only difference was spelling one field
  `first_load_idx` and `first_store_idx`; both are replaced by
  `patterns::vec_memory_fusion` with `analyze(desc, MemoryFusionKind)`,
  `MemoryFusionCandidate::first_op_idx`, and `MemoryFusionPlan`.
  `PtxAuditReport::vec_load` and `::vec_store` keep their names and now share
  that one plan type.
- The four emitter crates no longer open with their own `#![allow(...)]`
  block. Every lint named in those blocks is already allowed by
  `[workspace.lints]`, which all four crates inherit.
- Emitter test descriptors are built through
  `vyre_lower::descriptor_builder`, which gains `binop`, `load_global`,
  `store_global`, and neutral emission-target capability fixtures
  (`workgroup_limits`, `permissive_workgroup_limits`,
  `all_subgroup_capabilities`, `emission_target`, `target_without_subgroups`)
  behind its existing `test-fixtures` feature.
- Public API snapshots for `vyre-debug`, `vyre-driver`, `vyre-driver-cuda`,
  `vyre-emit-naga`, `vyre-emit-ptx`, `vyre-emit-spirv`, and `vyre-libs` now
  match their live surfaces. `vyre-debug` drops the `scan_explain` report,
  error, exactness, and factor-role types that left with the scan product.
  `vyre-libs` folds `graph::ast_walk_preorder` and `graph::ast_walk_postorder`
  into one `graph::ast_walk` module and publishes `scan::pack_haystack_u32`.
  The emitters and drivers publish the megakernel frontier plan error, the
  neutral execution planner, and device capability constants that had shipped
  without a snapshot refresh.
- The standalone `vyre-harness` package is gone. Semantic operation identity,
  tier classification, and registration now live in `vyre-foundation`; library
  fixture views live in `vyre-libs`; conformance execution and parity policy
  live in `vyre-conform`; self-substrate behavior tests live with their owner.
- Scan products now return the foundation-owned `ByteRange { tag, start, end
  }`. The deprecated `Match` and `LiteralMatch` surfaces and the duplicate
  primitive range type are gone.
- The reference interpreter now consumes foundation-owned IR, diagnostics, and
  operation metadata directly instead of depending on the public `vyre` facade.
- PTX f32 canonicalization now uses native flush-to-zero multiplication plus
  NaN selection, preserving signed zero and canonical NaN semantics with fewer
  instructions and registers.
- Release benchmark commands now run 300 warmup samples before measurement so
  accelerator clock preconditioning is explicit and reproducible.
- Foundation now exposes IR-specific `IrError` and `IrResult` contracts instead
  of a cross-domain error sink. Reference interpretation, backend execution,
  WGPU device selection, and runtime framing return owner-local typed failures.
- Documentation pages now declare audience, owner, authority, kind, and
  generated/manual ownership. Crate dependency records declare purpose,
  features, target conditions, visibility, and destination seam, and optimizer
  pass reference pages are generated from the live pass registry.
- Floating-point parity now uses one foundation-owned comparison contract for
  semantic operation witnesses, including each operation's declared tolerance.
  The library operation catalog distinguishes the complete semantic inventory
  from its deterministic executable-fixture projection.
- C typedef row phases remain canonical callable operations. The operation
  matrix marks them as inlined callees whose execution coverage belongs to
  fixture-backed parent operations.

### Removed

- `OperationTier::Primitive` and `OperationTier::Runtime` are gone.
  `vyre-primitives::` classifies as `OperationTier::Intrinsic`, the only Category C
  tier, and the operation-matrix spelling is `intrinsic`.
- `vyre-driver` no longer registers `core.indirect_dispatch`, `io.dma_from_nvme`,
  `io.write_back_to_nvme`, `mem.zerocopy_map` or `mem.unmap`, and
  `vyre_driver::registry::INDIRECT_DISPATCH_OP_ID` is gone with them. A host-side
  runtime capability has no program to lower and no fixture to compare, so it
  carries no operation identity; indirect dispatch is reached through
  `RequiredCapabilities` and `VyreBackend::supports_indirect_dispatch`, and NVMe
  ingest and zero-copy mapping through the `vyre-runtime` io_uring driver. The
  registry now refuses an id whose namespace names no owning crate, so the
  fixture-coverage exemption for those five ids is gone too.
- Self-substrate no longer publishes source-text validators for deleted
  C-frontend test files or parser release artifacts. Diagnostic and
  preprocessing conformance now belongs to the live frontend and conformance
  paths.
- Neural operations and opaque-payload helpers now use their category-owned
  module paths. Flat compatibility re-exports and the `matching::ops` shim are
  gone; unclassified backend failures use `BackendError::Other`.
- Backend registration is now consumed from `vyre-driver`, the `ReferenceKind`
  alias is gone in favor of `vyre-spec::CpuFn`, and `gpu_int_literal_scan()` no
  longer accepts an ignored source-length parameter.
- The WGPU host-ingress and raw persistent-kernel compiler routes are gone.
  Persistent product execution uses authenticated artifact sessions; concrete
  pipeline compilation remains available only as a hidden oracle helper for
  driver cache tests.
- The macro crate now exports only the production-used AST registry and
  semantic pass registration generators. Test-only operation registration,
  algebraic-law derive, no-op builder marker, and generated decoder stubs are
  gone.
- Twelve `vyre-libs` operations that re-registered a `vyre-primitives` kernel
  under a second identity are gone: `hash::{adler32, crc32, fnv1a32, fnv1a64,
  multi_hash}`, `logical::{and, or, xor}`, `parsing::{bracket_match,
  core_delimiter_match}`, `security::path_reconstruct` and
  `math::succinct::select1_query`. Each wrapper only re-tagged the primitive
  program, so the kernel keeps one id and callers use the `vyre-primitives`
  builder directly. `vyre-libs::hash` now holds `blake3_compress` alone and
  `vyre-libs::logical` holds the synthesized `nand` and `nor`.

### Fixed

- `xtask-registry --help` and `xtask-evidence --help` exit 0 and list the
  subcommands each binary dispatches, derived from its `IMPLEMENTED` table. Both
  treated `--help` as an unknown subcommand and exited 1.
- The `structure-gate` binary is registered in `docs/CLI.toml`. It is the
  program `.github/workflows/gates.yml` runs for the workspace structure check,
  and it was the only workspace binary absent from the CLI contract.
- `xtask dup-scan` no longer advertises an `--output PATH` option. No such
  option was ever implemented.
- Emitted SPIR-V is validated again. When `spirv-val` was absent the shared
  assertion fell back to checking that a blob held at least five words and
  carried a plausible version word, then returned, so every emission passed on a
  machine without the validator and the gate built on it proved nothing there.
  The validator is now required, the suite is registered behind the new
  `spirv-val` feature of `vyre-driver-spirv` so a default `--workspace` run skips
  the target instead of running the header-only path, and the `spirv-validation`
  job in `gates.yml` installs the validator and runs it. No device is involved.
- Every `vyre-driver-cuda` test target is now gated on a real device. Fifty-five
  `*gpu_parity*` targets were named by no workflow, and the script meant to cover
  them still named a test file that no longer exists, so it exited at its first
  target and measured nothing. Its roster is derived from tracked test targets,
  so a target added later is covered by existing, and the `CUDA parity suite` job
  in `gpu-parity.yml` runs it with the GPU release gate requiring the result.
- Documentation builds again. Sixteen intra-doc links resolved to nothing, and
  because the workspace denies `broken_intra_doc_links` that made `cargo doc`
  fail outright for `vyre-driver`, `vyre-foundation`, `vyre-libs`,
  `vyre-primitives`, `vyre-self-substrate`, and `vyre-conform`. A module's inner
  documentation merges with the outer comment at its declaration, so a link
  written against a sibling by bare name resolves in the parent module instead;
  those links now carry a full path. Links that pointed at items compiled only
  under `cpu-parity` name the item as code instead of linking it.
- `vyre-lints` exposes `Allowlist::default` and `Allowlist::measured_roots`,
  which the recorded public API surface had not captured.
- Driver decorators now preserve the concrete backend device profile, including
  device-timestamp capability and timing quality.
- Enforced benchmark contract failures now retain correctness, timing metrics,
  device identity, and measured speedup in the failed case report instead of
  collapsing into an unprobed error shell.
- Resident throughput batches preserve complete device-timestamp totals and
  normalize them per logical item. String bitmap scatter uses subgroup ballots
  to materialize 16 independent output rows in one resident dispatch, with
  exact CPU-oracle parity.
- `xtask heuristic-audit` now resolves both standalone Vyre checkouts and the
  enclosing Santh workspace without duplicating the Vyre path.
- Public API checks now discover every committed crate snapshot, parse exact
  package names, use dependency-noise-free output, and reject ordinary snapshot
  updates that remove or change an existing item.
- Empty QK-gain tensor shapes now declare a zero-byte output range instead of
  an unknown-size backend allocation, while overflowing positive shapes fail
  closed with an actionable trap program instead of wrapping their element
  count.
- Grouped-query attention now composes the canonical max, normalization-sum,
  and weighted-write primitives with explicit KV-head bases. Overflowing row or
  element counts fail with a sharding error before buffer declarations are
  built.
- WGPU resident dispatch now splits `GridSync` programs at launch boundaries
  before compilation, preventing oversized resident fixed-point grids from
  deadlocking inside a software global barrier.
- The WGPU stream-sharding error is now nameable as
  `engine::multi_gpu::StreamShardError` without changing existing signatures.
- Workspace documentation now resolves NFA conversion and megakernel table
  links.
- The reduction benchmark now measures atomic-scalar and workgroup-tree sums on
  the same GPU at 32 and 1,048,576 elements. It verifies both routes exactly,
  selects the measured winner per size, and records contention and barrier
  counters. NVIDIA idle clocks no longer invalidate a cold, low-utilization
  microbenchmark as thermal instability.
- V055 now accepts a post-barrier loop exit only when its full return path is
  workgroup-uniform. It derives same-address loads from an acquiring barrier
  and rejects intervening writes, divergent indices, atomics, and
  lane-dependent guards. The DCE fixpoint loop therefore removes one redundant
  barrier per iteration without weakening the unsafe-exit rejection.
- Public API snapshots now cover every workspace package whose Cargo manifest
  permits publication, including CUDA and every emitter/runtime library. The
  manifest-derived gate rejects both missing snapshots and stale snapshots for
  packages that no longer publish.
- Removed the unreachable `vyre-bench` dataflow baseline module whose
  undeclared feature could never be enabled and whose engine dependency does
  not exist in this workspace. Benchmark feature guards now have a manifest
  agreement gate, so a hidden undeclared case cannot recur.
- Regex DFA replay now gives open-ended repetitions an explicit finite policy
  instead of treating their minimum as a maximum. Whole-buffer variable-length
  matches derive exact starts from candidate origins, and region evidence
  returns one longest extent per pattern and origin.
- Weighted paged-corpus scans now expose per-device timing and byte balance.
  The physical two-adapter benchmark verifies exact single-device parity and
  records paired end-to-end speedup, topology, staging overhead, and raw
  samples.
- Package readiness now validates unpublished, version-matched release
  dependencies through local registry patches after Cargo normalizes path
  dependencies for packaging. Cross-repository `weirflow` archive evidence now
  records its real files, examples, Rust sources, and file-list digest.
- Workspace crate ownership now comes from one manifest-checked registry. The
  tier gate rejects missing crates, undeclared production edges, and stale
  generated graph or ownership guides, while planned compiler boundaries stay
  visibly separate from current workspace members.
- Testing guides are now generated for all 36 workspace members from Cargo
  features and targets plus maintained hardware, evidence, skip, and failure
  metadata. The documentation gate rejects missing, orphaned, or stale guides.
- Every workspace crate README now carries a manifest-backed contract for
  purpose, boundaries, a runnable example, features, errors, testing, release
  status, and ownership. Retired 0.4.x package claims and README drift fail the
  documentation gate.
- Operation documentation now has one generated JSON authority covering every
  linked library, primitive, intrinsic, and runtime dialect operation.
  Schema-derived inventories and subsystem catalogs expose exact tiers,
  categories, program or dialect signatures, Cargo feature routes, oracles,
  backend support evidence, algebraic laws, composition chains, and counts.
- The root README now derives every workspace crate's publication and support
  status from manifests and maintained metadata. Operation tier counts come
  from the canonical operation schema, backend claims come from executable
  backend evidence, and the architecture identifies Metal as Apple-active
  instead of planned.
- Architecture guides now use the generated 36-crate dependency graph, joined
  operation registries, CUDA-first backend evidence, typed cross-program
  composition, and explicit runtime/compiler/driver megakernel boundaries. The
  earlier device-bytecode-interpreter RFC is retained as superseded rationale.
- Documentation coverage now reports measured gates instead of universal
  completeness. Public guides distinguish generic consumers from named
  integrations, and the documentation gate rejects missing or gitignored
  repository inputs hidden in code spans and shell examples.
- The documentation matrix now covers every indexed public document and
  workspace crate README. Each row records audience, owner, authority, source
  artifacts, verification date, executable examples, version coherence, support
  status, and claim-evidence blockers.
- Every current public guide is now revalidated for Vyre 0.7.2. Historical
  architecture, migration, release, operation, and testing documents are
  explicitly archived or superseded, generated views identify their source, and
  crate-local paths remain reproducible in a clean checkout.
- Release operations now use one runbook and one generated checklist derived
  from release-train versions, repositories, package groups, tags, approval
  actions, and validated changelog fragments. The guarded launcher pushes
  candidate tags before publication, final tags afterward, and records
  completion only after external actions succeed.
- Command-line documentation now inventories and executes all 12 workspace
  binaries and 84 subcommands, publishes exact help, exit-code, environment,
  configuration, hardware, and failure contracts in crate READMEs, and gates
  drift in documentation CI. The vyre-wgpu demo is documented and exercised on
  the real GPU lane, while helper --help routes are side-effect free.

## [0.7.1] - 2026-08-01

### Fixed

- Release benchmark source fingerprints now exclude operator-internal files such
  as `AGENTS.md`, `CLAUDE.md`, and `SKILL.md`. A public checkout therefore
  reproduces the same runtime source identity as the private release workspace.

## [0.7.0]  -  2026-07-30

One release. The work that had been staged as 0.6.6 is folded in here: it could not
ship as a patch, because making its release gate pass required canonicalizing
eigenvector sign in the shared Jacobi body, and that changes the observable output of
a published op.

The only source edit an upgrade requires is the dataflow-import rename. See the
migration table under "Removed".

### Fixed: fusing a narrow synchronizing arm produced an intermittently wrong kernel (`vyre-foundation`)

`fuse_programs` set the fused workgroup size to the axis-wise maximum over the
arms and fused anyway. For an arm whose invocations are independent that is only
a launch-size change. For an arm that synchronizes its workgroup or keeps state
in workgroup memory it changes the meaning of the arm: such an arm guards its
body for its own width, so under a wider workgroup the invocations with no work
skip the guarded body and never reach the barrier the working invocations wait
on. A workgroup barrier that is not reached by every invocation in the workgroup
is undefined.

Piping `sinkhorn_scale::consumer_b` (workgroup 256) into `scan_prefix_sum` at
n=4 (workgroup 4, two workgroup buffers, five barriers) produced a kernel that
returned the wrong final lane on 49 dispatches out of 500 of the same input:
the prefix sums came back as `[4, 7, 17, 8]` instead of `[4, 7, 17, 19]`. Being
intermittent, it read as flakiness rather than as unsound fusion.

`fuse_programs` now refuses such a batch with the new
`FusionError::WorkgroupGeometry`, naming the arm, both geometries, and what in
the arm makes the widening unsafe. Arms whose invocations are independent are
still widened, and arms that already agree on their workgroup still fuse even
when both synchronize. If you hit the refusal, dispatch that arm separately.

A fused program also keeps `non_composable_with_self` as the OR over its arms.
It used to be reset to `false`, so a second round of fusion could place two
copies of a scratch-carrying body in one kernel. The same loss is fixed in the
decode-scan fusion pass, the streaming decode adapter, and the two scan
programs that tag themselves with a region: all of them rebuilt through
`Program::wrapped`, which constructs a NEW program and so starts the metadata
fresh. Use `with_rewritten_buffers`, `with_rewritten_entry`,
`with_rewritten_wrapped_entry` or `map_entry` to change part of a program.

`vyre-foundation/tests/fusion_workgroup_geometry.rs` and
`vyre-foundation/tests/fusion_composability_metadata.rs` pin both.

### Fixed: the raw-byte C syntax parser under-counted tokens (`vyre-frontend-c`)

Any source above two 1024-token blocks reported a token count that was too low,
with no error. 4096, 8192 and 66560 semicolons all reported 2048 tokens; 2049
reported 1025.

Sparse token compaction runs a block-total stage, one 1024-lane workgroup per
block, that writes each block's token count to `block_totals[block]`. That
stage's only sized buffer is `block_totals`, one word per block, and its input
arrives as a resident device blob whose length the dispatch grid inference cannot
read. Inference therefore chose `ceil(num_blocks / 1024)`, which is one workgroup
for every source under a million tokens. Block 0 computed its total and every
later block kept the zero it was allocated with, so the scanned prefix that ranks
tokens in the compact stage collapsed to `block_totals[0]`.

Both block-total dispatches now state their grid instead of leaving it to be
inferred. `vyre-frontend-c/tests/raw_syntax_multi_block_token_counts.rs` pins the
exact count at each block boundary.

### Fixed: three rewrite passes reused result ids across bodies (`vyre-lower`)

Result ids are unique across a whole `KernelDescriptor`, not per body: the PTX
emitter keeps one flat result-id to register map for the entire kernel, so an id
two bodies both define resolves to whichever producer the emitter walked last.
Three passes broke that.

- `branch_collapse` inlined a collapsed `StructuredIfThen` body into its parent
  but left the child body populated. Child indices are positional, so the slot
  cannot be removed without reindexing its siblings; it is now emptied instead.
- `egraph_saturation` and `shared_mem_promote` recursed over the body tree and
  rebuilt their result allocator at every level, so a nested body seeded its
  high-water mark from its own subtree. Both now thread one allocator from the
  descriptor root, the same shape the 0.7.0 `loop_unroll` fix uses.

The debug-only post-pass verify surfaced this on the int4 CUDA parity suites. It
stayed hidden because three descriptor builders, including the soundness fuzzer,
assigned ids per body and so failed `verify` before any rewrite ran.

### Fixed: GPU dead-code elimination supplied one input buffer too few (`vyre-self-substrate`)

The persistent-BFS analysis program `gpu_dce` dispatches gained a `converged`
output in this release. A ReadWrite buffer binds as InputOutput, so it consumes
an input slot as well as an output slot, and the direct path kept filling eight
slots for a program declaring nine. Every dispatch failed with "expected 9 input
buffer(s) from Program declarations but received 8". The resident path already
passed all nine. A new suite runs the pass against a recording dispatcher, so a
future slot-count drift fails without a GPU.

### Removed: the `strict-fp` feature (`vyre-harness`, `vyre-test-harness`)

`strict-fp` claimed to forbid multiply-add contraction and demand bit-identical
f32 results. It forbade nothing: no emitter read it, and its only effect was to
force `f32_ulp_tolerance` to 0 for backend-vs-reference comparisons. Since
contraction is a documented backend right, and both cuda and wgpu fold `a*b+c`
into one FMA, that made `cargo test --workspace --all-features` unable to pass:
`newton_schulz_poly5_f32` drifted 4 ULP, `newton_schulz_5step` 2 and `ema_apply`
1, with the two backends agreeing bit-for-bit with each other and differing only
from the CPU reference.

If you enabled `strict-fp`, drop it from your feature list. The elementary and
transcendental ULP budgets are unchanged and still apply. Bounding contraction
is an emitter job; a tolerance constant cannot do it.

### Security: two advisories cleared in the dependency graph

- `crossbeam-epoch` moves 0.9.18 to 0.9.20, clearing RUSTSEC-2026-0204: the `fmt::Pointer`
  impl for `Atomic` and `Shared` dereferenced the underlying pointer, so formatting a null
  pointer was an invalid dereference.
- `anyhow` moves 1.0.102 to 1.0.104, clearing RUSTSEC-2026-0190: adding context with
  `Error::context` and then calling `Error::downcast_mut` on the result violated borrow
  rules and was undefined behaviour.

`cargo deny check` is now green on advisories, bans, licenses, and sources.

### Added: a composite op can take a whole buffer (`vyre-foundation`)

An op could only receive scalars, so a phase that indexes a table could not be
split into its own composition. The only way to name a buffer at a call site was
`Expr::Var`, which the validator reads as a scope-bound variable, so every such
program was rejected with "reference to undeclared variable". The
composition-discipline gate therefore told over-budget ops to split into
compositions while the pipeline refused to compile the result.

- `Expr::BufferRef { buffer }` names a buffer. It is not a value: it has no type,
  and the validator rejects it (V051) anywhere except a call argument. Build one
  with `Expr::buffer_ref("table")`.
- An op signature declares such a parameter as `buffer<u32>`. The validator checks
  that the argument is a buffer reference (V053), that the buffer is declared
  (V052), and that its element type matches (V054).
- Inlining a call with a buffer argument RETARGETS the callee's loads, atomics, and
  `BufLen` at the caller's buffer, keeping the callee's index expressions. A scalar
  argument still substitutes its value, so `BufLen` of a scalar parameter stays 1
  and `BufLen` of a buffer parameter is now the caller buffer's real length.
- Wire format rev 5 adds expression tag 22 for it. The decoder still reads rev 4,
  since rev 5 only appends a tag. See `docs/wire-format.md`.
  `framing::wire_format_version_is_supported` is now the single owner of the accepted
  range: three decode paths had each spelled the comparison for themselves, and one was
  missed when the range widened.
- `V047` and `V051` through `V054` are cataloged in `docs/error-codes.md`. Call-signature
  validation moved to its own `validate::call_rules` module.

### Fixed: the reference interpreter computed nothing for a composite op (`vyre-reference`)

`Expr::Call` was always dispatched to the op's registered CPU function. A composite
op is defined by its IR body and registers no CPU function, so it landed on the
non-executable sentinel in `LoweringTable::empty()`, which clears the output buffer
and returns. The interpreter reported success and produced zeros.

- The interpreter now inlines every composite body before execution, through the
  single `program_for_interpreter` funnel, so only intrinsics reach the CPU dispatch.
  `vyre_foundation::ir::inline_composite_calls` is the new entry point;
  `UnresolvedCalls` selects whether an unresolvable call is an error or is left in
  place.
- Reaching the sentinel is now a hard error naming the op, instead of a silent
  empty result.
- Inlining returns a call-free program untouched instead of rebuilding its node tree,
  so running this on every reference execution costs nothing when there is no call.

### Changed: typedef annotation is three ops instead of one monolith (`vyre-libs`)

`vyre-libs::parsing::c11_annotate_typedef_names` carried every phase inline: 613
statement nodes against a 200 budget, control-flow depth 20 against 6, and 37 loops
against 8. The composition-discipline gate has no exemption list, so the op was red,
and it could not be split because a callee could not take a buffer.

- The three per-row phases are now registered ops of their own, each answering one
  question about one row: `c11_typedef_scope_open_for_row`,
  `c11_typedef_visible_name_for_row{,_packed_haystack}`, and
  `c11_typedef_decl_kind_for_row{,_packed_haystack}`. They take the node table and
  haystack as buffer references and the row index as a scalar.
- The calls inline before lowering, so the emitted kernel is unchanged and the C
  parser's oracle parity is unaffected.
- `emit_typedef_visibility_scan` and `emit_current_declaration_annotation`, the two
  wrappers the annotator no longer uses, are removed.
- `vyre_libs::dialect_init::ensure_ops_resolvable` installs the driver registry as the
  process op lookup. A builder that emits a call now calls it, so the program it returns
  still inlines and validates for a caller who never touches `vyre-driver` directly.

### Added: device convergence flags for persistent BFS (`vyre-primitives`, `vyre-self-substrate`)

A persistent-BFS closure that exhausts its `max_iters` budget while still growing
produces an under-approximated frontier. Until now the device path returned that
partial frontier with no way to tell it apart from a real fixpoint, so a caller
silently reasoned over a truncated reachability set.

- Every persistent-BFS program now writes a `converged` output: one u32 word for
  the single-query programs, a per-query u32 array for the batch programs. It is
  `1` when a step added nothing before the budget was exhausted, and `0` when the
  loop ran all `max_iters` steps while still growing, or when `max_iters == 0`.
  `BINDING_CONVERGED` names the binding.
- `validate_persistent_bfs_converged_flag` rejects any other value, and
  `cpu_ref::PersistentBfsConvergence` plus `try_cpu_ref_converged` give the CPU
  reference the same signal so device and reference results are comparable
  flag-for-flag.
- `vyre-libs` borrow checking now uses it: `enforce_borrow_closures_converged`
  fails the dispatch with a `Fix:` message when any forward loan-issue or
  backward loan-use closure did not converge, because borrow-checking a truncated
  loan reachability set silently drops conflicts.
- The optimizer's dispatched DCE uses it too. `build_dce_bfs_program` declares the
  `converged` word its module doc already promised was part of the layout, sets it
  on the early-exit fixpoint branch, and leaves it zero when the loop burns its
  whole budget while still growing. `gpu_dce` now reads it and fails closed:
  liveness is a reachability closure, so DCE over a truncated one deletes live
  code. The failure is a miscompile, not a missed optimization, which is why this
  path refuses rather than degrades.

### Added: per-iteration frontier density telemetry (`vyre-primitives`)

- `persistent_bfs_with_density`, `persistent_bfs_batch_with_density`, and
  `try_persistent_bfs_batch_with_density` build programs that declare one extra
  u32 output, `density_active` (`BINDING_DENSITY_ACTIVE`,
  `DENSITY_ACTIVE_BUFFER`), holding the frontier popcount after each traversal
  step. The batch layout is `q * max_iters + i`. A host reconstructs every
  per-iteration density aggregate from this array plus the seed popcount instead
  of a per-step device round-trip.
- The density write is a recompute-and-store, not an accumulating atomic, so it
  lands the same value when the grid-sync split re-executes a segment to a
  fixpoint. An atomic would double-count there.
- The base `persistent_bfs` and `try_persistent_bfs_batch` programs are
  byte-for-byte unchanged, so callers that do not want telemetry pay nothing.
- `try_cpu_ref_density` is the CPU reference counterpart. New device-parity
  suites cover both the converged flag and the density array.

### Added: closure-driven grid-sync splitting (`vyre-driver`)

- `dispatch_with_grid_sync_split_via_into` and its allocating wrapper
  `dispatch_with_grid_sync_split_via` take an opaque single-launch dispatch
  closure instead of a `&dyn VyreBackend`. A host-loop fixpoint solver can move
  its convergence loop onto the device without holding a backend handle, plugging
  in the CPU reference, CUDA, or wgpu as a closure.
- The split, input-refresh, and adaptive-convergence logic moved into a shared
  `dispatch_grid_sync_split_generic`, so the backend entry and the closure entry
  run the same code and converge to identical output. Neither path has its own
  copy of the loop.

### Added: `DispatchConfig::dispatch_grid` (`vyre-driver`, `vyre-reference`)

- The CPU reference interpreter inferred its coverage from buffer shapes,
  distributing the dispatch only across workgroup axes larger than one. A program
  fanning a `[256, 1, 1]` workgroup across `grid.y`, which is how batched
  persistent BFS runs one query per block, collapsed to `grid.y == 1` and
  computed only the first query with no diagnostic.
- `dispatch_grid: Option<[u32; 3]>` states the real per-axis workgroup grid. When
  set it overrides shape inference entirely, so the interpreter covers every
  workgroup the GPU would. `None` keeps the previous inference. It takes
  precedence over `dispatch_elements`, which is a 1-D floor.

### Changed: selective fused positioned evidence (`vyre-libs`)

- Add `GpuLiteralSet::prepare_resident_fused_scan_positioned_from`, which keeps
  per-region presence complete for every literal while emitting match triples
  only for an appended positioned-evidence segment. Dense admission-only rows
  no longer consume atomic counter capacity or readback bandwidth.
- The bounded-range suffix3 prefilter gains the same shape:
  `classic_ac_bounded_ranges_suffix3_presence_and_positions_by_region_program_filtered_ext`
  and `try_build_ac_bounded_ranges_suffix3_presence_and_positions_by_region_program_filtered`
  keep presence complete for every pattern while filtering only the atomic triple
  append to IDs at or above a supplied boundary.

### Fixed: paged corpus windows read exactly their planned byte length (`vyre-libs`)

- `fill_window_from_paths` appended each file in the window whole. A file that
  grew between planning and reading overran its window, so the haystack no longer
  matched the offsets the plan had computed and reported matches at wrong
  positions. It now reads at most the remaining window budget per file and errors
  with the path and both byte counts if a file overruns or the window comes up
  short.

### Removed: product-specific names in the dataflow-import API (`vyre-lower`, `vyre-libs`)

vyre's dataflow-import surface named a specific downstream consumer. The API is
generic: it imports alias and reaching-definition facts from any external
dataflow engine. The names now say that, which also stops the published API from
coupling vyre to one sibling product.

| Before | After |
| --- | --- |
| `analyses::weir_alias` | `analyses::alias_import` |
| `analyses::weir_reaching_def` | `analyses::reaching_def_import` |
| `dead_store_with_weir_alias_facts` | `dead_store_with_external_alias_facts` |
| `licm_with_weir_alias_facts` | `licm_with_external_alias_facts` |
| `load_forwarding_with_weir_alias_facts` | `load_forwarding_with_external_alias_facts` |
| `loop_fission_with_weir_alias_facts` | `loop_fission_with_external_alias_facts` |
| `loop_fusion_with_weir_alias_facts` | `loop_fusion_with_external_alias_facts` |
| `security::weir_ifds` | `security::external_ifds` |
| `route_security_taint_through_weir_ifds` | `route_security_taint_through_external_ifds` |
| `security_witness_path_from_weir` | `security_witness_path_from_external_path` |
| `WeirIfdsSecurity{Buffers,Dispatch,RouteError}` | `ExternalIfdsSecurity{Buffers,Dispatch,RouteError}` |
| `WEIR_IFDS_SECURITY_BACKEND_ID` | `EXTERNAL_IFDS_SECURITY_BACKEND_ID` |
| `cfg(feature = "weir_ifds_external_engine")` guard | `cfg(feature = "external_ifds_engine")` guard |

- The shared fact schema's producer id changes from `weir` to
  `external-dataflow`. A serialized fact header carries the producer id, so a
  consumer matching on the old string must update. The schema version itself is
  unchanged.
- `rules/security_predicates.toml` renames the `weir_mapping` key to
  `external_mapping` across all ten predicates. This is a Tier-B data file, so a
  user who copied it to extend the catalog updates the key name.
- The rename is mechanical. There are no behavior or signature changes beyond the
  names in this table.


### Fixed: unrolled loops reused result ids and the CUDA backend miscompiled the address (`vyre-lower`)

A store indexed by the invocation id came out of the CUDA path as a store at a
constant offset, so every lane wrote the same element and element 0 was never
written. `sinkhorn_iterate` and both of its catalog wrappers returned zeros on
CUDA while the reference and wgpu agreed.

- `loop_unroll` reseeded its free-id counter from the subtree it was currently
  visiting, at every level of the recursion. Ids inside a child body are small, so
  unrolling a short loop nested under a long one minted ids that were fresh within
  that subtree and already owned by a sibling or an ancestor. It now threads one
  descriptor-wide counter through the whole recursion.
- The PTX emitter keys its literal map on the raw result id across the whole
  kernel, so a store index produced by `GlobalInvocationId` in one body resolved to
  a sibling body's `Literal(1)` and the address folded to a constant. The
  descriptor was well formed by every check that existed; only the ids were
  ambiguous.
- `verify` gained invariant 6, `ResultIdReusedAcrossBodies`. It previously
  collected produced ids fresh per body, so it caught only duplicates sitting side
  by side in one op list and cross-body reuse verified clean.

### Fixed: a split op could not be lowered (`vyre-foundation`, `vyre-lower`)

The canonical pre-emit pipeline inlines through `inline_calls`, whose default
resolver returned `None` for every op id, so any program containing an `Expr::Call`
failed with `InlineUnknownOp` before a backend saw it. Only `vyre-aot` passed a real
resolver. The composition-discipline gate meanwhile instructs an over-budget op to
split into sub-ops connected via `Expr::Call`, so the prescribed remedy produced code
the pipeline refused to compile. The default resolver now asks the installed dialect
lookup, the same dependency-inversion boundary the reference interpreter uses. An op
resolves when it is registered and carries a composition body; intrinsics and
unregistered ids still do not.

This unblocks half of the composition-discipline split. A callee still takes only
scalar arguments, so a phase that indexes a table cannot yet be factored out.

### Changed: eigenvectors come back with a canonical sign (`vyre-primitives`)

An eigenvector is only defined up to sign, so the Jacobi rotation accumulation was free
to return `v` or `-v` and both were correct. That made every consumer of
`jacobi_eigen_body` unpinnable: a backend that rounded one rotation differently landed
on the opposite sign, and anything dividing by the vector flipped with it.

- `jacobi_eigen_body` now canonicalizes each eigenvector column so its first component
  larger than `EIGENVECTOR_SIGN_EPSILON` in magnitude is positive. `symmetric_eigen_jacobi`
  and `tensor_train_decompose` inherit it.
- If you consumed raw eigenvector columns and applied your own sign convention, remove it.
  If you compared columns against stored values, half of them may now differ by a sign.
  Eigenvalues, the eigen-decomposition itself, and anything invariant to sign (a
  reconstruction, a projection) are unchanged.

### Fixed: registered ops that no backend actually executed (`vyre-primitives`)

Two ops were registered with no fixtures, so they counted as covered while nothing ever
checked a value.

- `tensor_train_decompose` shipped without an oracle on the grounds that a truncated SVD
  is basis-dependent. Two of the three ambiguities are removed rather than tolerated: sign
  by the canonicalization above, and the degenerate eigen-subspace by moving the fixture
  from a wide 2x4 unfolding to a tall 4x2 one. A wide unfolding makes the Gram matrix
  rank-deficient, and a degenerate null space has no single correct eigenbasis. The oracle
  is derived analytically from the closed-form decomposition.
- `multi_block_prefix_scan_inclusive_sum` had `test_inputs: None`. It now runs a
  64-element inclusive scan against the closed-form triangular-number expectation.

### Fixed: the cross-backend parity matrix never resolved calls (`vyre-conform-runner`)

The parity harness did not install the process-wide dialect lookup, so validation rejected
any op carrying an `Expr::Call` with V016 before it reached a backend. This was not
specific to the coverage bundle that surfaced it; it applied to every op with a call. The
harness now installs the registry first, and the expr-variant bundle calls a registered
callee instead of a placeholder id.

### Fixed: `docs/INDEX.md` listed documents that are not published

The index gate enumerated `docs/` from the filesystem, so every gitignored operator
document failed it, and the index in turn pointed at 22 documents that `.gitignore`
excludes and a published crate therefore does not contain. The gate now enumerates tracked
files, and the private rows are gone.

### Fixed: two guards disagreed about release runbooks

`vyre-lints` and `scripts/check_platform_consumer_docs.sh` both enforce the
downstream-consumer naming boundary and each carried its own exemption list, so
`docs/RELEASE.md` was exempt in one and scanned by the other. The list now lives in
`vyre-lints/rules/release_coordination_docs.txt` and both read it.

### Fixed: the workspace now builds from a clean clone (`vyre-conform-enforce`)

- 31 governance test suites embedded `docs/optimization/ALL_AXES_ACCELERATION_PLAN.md`
  with `include_str!`. That file is private operator state that `.gitignore` excludes
  from the public repository, so `cargo test` failed to compile on every fresh clone
  with `couldn't read ...: No such file or directory`. Nothing caught it because the
  file is always present in a maintainer's checkout.
- The removed assertions only checked that the private document contained literal row-ID
  strings such as `VX-1081`. Every requirement they were meant to prove is still asserted
  directly against the committed `docs/optimization/*.toml` artifacts, which carry the
  same row ranges, so coverage is unchanged.
- `tests/clean_checkout_build_governance.rs` now fails if any Rust source embeds a path
  that git does not track, if the private acceleration plan is embedded again, or if a
  tracked file matches a `.gitignore` rule. The scanner skips `include_str!` occurrences
  inside string literals and comments, so the lint that greps for the macro name and the
  raw-string test fixtures that contain sample source are not reported as violations.

### Fixed: release gate resolved three identifiers incorrectly (`xtask`)

- The publish train named the dataflow product's package `weir`. The publishable package
  is `weirflow`; `weir` is only its library target name, and the bare `weir` name on
  crates.io belongs to an unrelated crate. `package-readiness` reported a blocker that no
  version bump could clear. `release_train::weir_package_name` is now the single owner and
  the three sites that hardcoded the name read from it.
- Every gate resolved the security compiler consumer at `libs/surge/surgec`, but it lives
  at `surge/surgec`. Eight sites carried the wrong prefix, so gates reported the tree as
  absent: `distributed-parser-coherence` alone raised 51 blockers claiming `src/lib.rs`
  does not exist for a crate with 229 test files, 5 benches, and 2 fuzz targets on disk.
  `release_train::compiler_consumer_relative_path` is now the single owner.
- `vyre-grammar-gen` had fallen out of the publish train after 0.6.2 and went stale on
  crates.io while every sibling advanced to 0.6.5, which is why in-workspace consumers had
  to pin it path-only. It is back in the train and publishes first, having no internal
  dependencies.
- `package-readiness` now reports zero blockers.

### Fixed: release runbooks contained unrunnable instructions (`docs`)

- A rename sweep had replaced the product name with a two-word phrase inside identifiers,
  producing `git tag vyre-0.4.1-dataflow consumer-0.0.1`, the xtask subcommand
  `vyre-dataflow consumer-release-gate`, the path `release/vyre-dataflow consumer-evidence.toml`,
  and the sentence `The The dataflow consumer repository`. Tags, subcommand names, and
  paths are literal strings an operator types, so they are restored across `RELEASE.md`,
  `RELEASE_CHECKLIST.md`, `RELEASE_ENGINEERING.md`, `PUBLISH_GATE.md`, and the v0.4.1 and
  v0.4.2 release notes.
- The consumer-coupling lint gained a narrow exemption for release runbooks, which name the
  products in the combined release train on purpose. Architecture docs, guides, and all
  Rust source stay under the guard. Three tests pin the exemption, prove it does not leak
  to neighbouring documents, and prove it never covers Rust source.
- The same sweep had neutralized the negative fixture in the coupling lint's own test, so
  the fixture no longer contained the string the lint must flag and the test failed while
  the lint was correct. Restored, with a comment recording why the fixture keeps the name.

### Fixed: stale dependency pins (`vyre-spec`, `vyre-primitives`, `vyre-intrinsics`, `vyre-driver-wgpu`)

- Eight internal dev-dependencies pinned `version = "0.6.1"` alongside their path, three
  releases behind the workspace. They are now path-only, matching the documented pattern:
  cargo strips path-only dev-dependencies at publish, so they cannot demand a stale
  crates.io version or block the publish train again.
- `examples/libs-template` pinned `vyre`, `vyre-foundation`, `vyre-spec`, and
  `vyre-reference` at `0.4.2` while pinning `vyre-libs` at `0.6.5`. The template is what a
  consumer copies, so it resolved a two-minor-old API. All pins now track the release.

### Changed: third-party dependency pins refreshed (`Cargo.toml`)

- Every third-party dependency is exact-pinned with `=`, so `cargo update` cannot move
  them and freshness is a deliberate edit. Seventeen pins advance to the current patch or
  minor release: `serde` 1.0.229, `thiserror` 2.0.19, `rand` 0.10.2, `tokio` 1.53.1,
  `bytemuck` 1.25.2, `proc-macro2` 1.0.107, `toml` 1.1.3, `faer` 0.24.4, `memchr` 2.8.3,
  `regex-syntax` 0.8.11, `rustc-hash` 2.1.3, `clap` 4.6.4, `regex` 1.13.1,
  `regex-automata` 0.4.16, `quote` 1.0.47, `crossbeam-channel` 0.5.16, `openssl` 0.10.81.
- `wgpu`/`naga`, `syn`, and `wide` stay on their current majors. Each of those bumps
  changes APIs vyre calls directly, so they are code changes rather than pin edits and do
  not ride a release-engineering release.

### Changed: repository identity moved to `santhreal` (`docs`, `.github`, crate metadata)

- `repository` and `homepage` metadata, `CODEOWNERS`, issue-template links, `CITATION.cff`,
  and the governance evidence now name `santhreal/vyre`. The workspace `homepage` points at
  `https://santh.dev`.
- README carries crates.io, docs.rs, and license badges.

### Added: adversarial coverage for loop peeling and induction rebinding (`vyre-lower`)

- Second-pass edge cases for loop peeling, induction-variable rebinding helpers, and
  shared-memory uniformity, including the control-flow shapes where a rebind must not fire.

### Fixed: private operator state is no longer stageable (`.gitignore`)

- Planning, status, audit, and agent-handoff documents were being staged out of
  subdirectories that the root-only ignore patterns did not cover, including a 625KB
  backlog and a 909KB operator plan. The patterns now apply at every depth.

### Changed: composition provenance and deduplication audits use canonical ownership

Cat-A wrappers now use the foundation-owned `tag_program` operation. The helper
preserves the primitive program metadata, keeps primitive generator ids as
children, and records the Cat-A operation as their parent. INT4 quantization
wrappers and predicate builders use this single path.

The generated `vyre-libs::catalog::*::consumer_a` and `consumer_b` registrations
have been removed. Primitive coverage now counts real composition callers only.
The operation matrix contains 371 tracked rows: 206 library rows, 149 primitive
rows, 9 intrinsic rows, 5 runtime rows, and 2 foundation IR rows.

The similarity audits now classify operations by canonical implementation
family. Source similarity parses Rust functions and methods with `syn`,
normalizes local bindings, and retains semantic identifiers such as called
operations, types, and constants.

### Fixed: the PTX cache key recomputed the program digest on every dispatch (`vyre-driver-cuda`)

`ptx_for_program_cached_with_key` derived its cache key from
`lower_subgroup_reductions(program.clone(), caps)`. The normalized program
digest that feeds that key is memoized on the program VALUE, and that value was
created and dropped inside a single dispatch, so the memo's only writer was a
temporary and the memo could never be read. A memo whose only writer is a
temporary is a memo that cannot ever be read.

Neither piece was wrong on its own, which is why reading either one could not
find this. `Program::clone` forwards all six memos correctly, and the digest
itself is a sound pure function of the program. The defect lived in the LIFETIME
of the value the key was derived from: a caller's program stayed permanently
cold because nothing ever computed its digest, so every dispatch cloned a cold
program, computed the digest on the clone, and dropped it.

The key is now derived from the caller's own program when the subgroup lowering
pass is a no-op, which is the ordinary case. The pass is already fully
copy-on-write internally: of its three returns, two hand the input straight back
and only the third rebuilds the entry. Pointer equality on the shared `Arc`
fields is therefore an O(1) witness that nothing was rewritten, and in that case
the two programs are the same value differing only in which memos are warm, so
the key receives byte-identical input. A program the pass does rewrite is still
keyed on its lowered form, because keying it on the unlowered digest would file
lowered PTX under the unlowered program's identity and serve a later dispatch a
kernel containing subgroup reductions it never requested.

Measured on an RTX 5090 with the `exatok` encode profile, 45 warm encodes per
corpus shape over 19 distinct program shapes. The digest walk cost 79.0 ns per
IR node (R-squared 0.907) and was 91.8 percent of the host PTX phase, making it
the largest single host term on the encode path. Digest computations per encode
fell from 6 to 0 on the `cjk` and `code` shapes, 4 to 0 on `prose`, and 3 to 0
on `short_pretokens`. The residual per-node rate in that phase fell from 83.1 to
4.0 ns per node, a factor of 21. Programs reach 12,410 IR nodes and 3.9 MB of
PTX for one dispatch, so the cost of getting the memo lifetime wrong grew with
the workload.

Host allocations per dispatch fell about 16-fold on the `short_pretokens`
fixture, from roughly 1,600 calls to roughly 100, deterministic across five
consecutive runs. That is a counted, load-independent instrument separate from
the phase probe and it corroborates the same change. Token ids are unaffected:
the cache key receives identical bytes, and `exatok` parity, determinism, device
parity and specials exactness gates all pass unchanged.

### Fixed: the parallel DCE fixpoint exited a synchronizing loop unordered (`vyre-self-substrate`)

The device DCE program's iteration body ended with an unconditional barrier and
then an early exit: once a step added no bit, lane 0 recorded convergence and the
invocation returned. That exit sat AFTER the body's last synchronizing node, so
one invocation could take the back edge and write while a sibling had not yet
reached the exit, and the sibling then left the kernel while the rest kept
iterating, freezing the data it owned partway through. Nothing hangs, because a
barrier does not count invocations that already returned, so the cost is
ANSWERS rather than liveness and a single workgroup is enough to hit it.

The shape was always there. It was not a hazard until now because a `Return`
nested inside a loop used to be emitted as nothing by `vyre-emit-ptx`, and the
program carried an explicit correctness argument resting on that: on device the
loop ran its full iteration budget and a `converged` gate, not the `Return`, was
what made the early exit real. Lowering a nested `Return` to a real branch turned
that documented no-op into a live exit and made the argument false at the moment
it landed, which is what surfaced the program to the V055 back-edge validator.

The body now ends with an unconditional barrier AFTER the exit branch. That is
safe here for a specific reason worth keeping: the exit condition reads a value
the preceding barrier settles, so it is workgroup-uniform, every lane sees the
same value, and the trailing barrier is reached by all lanes or by none, never by
a subset. It stays at body level, since a barrier inside the convergence gate
would desynchronize a workgroup whose lanes are allowed to read that flag stale.
Cost is one extra barrier per non-converged iteration against a body that already
costs two (INFERRED from the emitted node sequence, not timed).

The stale emitter claim in that program's comment is corrected in place, because
the file records that this reasoning had already misled three separate attempts,
and a wrong mechanical note in the one comment written to prevent a fourth is how
the fourth happens. V055 was not weakened. It still refuses any exit after a
loop's last barrier, including provably uniform ones like this; teaching it to
prove uniformity is real analysis and is deferred, with this program as the
motivating example.

Two suites now hold this shut, both host-only and GPU-free. In
`vyre-self-substrate`, `dce_program_back_edge_contract` asserts the built program
VALIDATES, which is the property that survives any future change to how a
`Return` lowers, and four of its tests mutate the real program to require the
refusal back: trailing barrier removed, barrier moved inside the convergence gate
(the plausible wrong fix), exit moved past the barrier with the barrier count
unchanged, and an unconditional exit after the barrier. That last one recorded a
correction: the rule refuses a provably UNIFORM exit too, so its reach is any
exit textually after the last barrier, not only a lane-dependent one (OBSERVED,
from the test failing against the opposite expectation).

In `vyre-primitives`, `loop_back_edge_audit` asks the question directly instead
of waiting for a downstream symptom, since all four instances of this shape found
so far were found because something else went red. It builds every shipped
program whose file contains both a loop and a barrier and validates it on the
host: thirteen programs at five iteration budgets each, all clean, so there is no
fifth instance among them (a measured absence over that set, not a proof about
the crate). Exactly four of the thirteen put a barrier inside a loop body and are
governed by the rule at all. Two of those four end in an unconditional barrier
and are exit-proof, meaning an exit added later stays ordered; the two density
variants are merely exit-free, legal because they hold no exit. That gap is
recorded rather than closed: an exit added there is refused loudly at validation,
and closing it would cost a real barrier per iteration in a program with no
defect to justify it.

### Fixed: cold-start launch width stranded a third of every SM (`vyre-driver`)

Blocks per compute unit is an integral division: a unit hosts whole workgroups
only. On an RTX 5090 the per-SM budget is 1536 threads, so a 1024-wide group
hosts exactly ONE block and 512 of every SM's 1536 thread slots are unreachable
for the life of the launch. The cold-start estimator had no occupancy term. It
scored candidates on tail waste and per-group overhead alone, which made 1024
the outright winner for any element count that is a multiple of 1024, precisely
where its idle-lane penalty vanishes. Every unmeasured tunable 1-D dispatch on
this class of device therefore launched at two thirds occupancy by arithmetic,
before any kernel ran.

The cooperative consequence is larger than the occupancy one. A grid-sync
program fits a single cooperative launch only while its grid stays inside the
device's resident-thread ceiling, and that ceiling is a function of the width
the tuner RESOLVES, not the width the program declares: 170 SMs x 1024 resident
threads is 174,080 lanes at width 1024, against 170 x 1536 = 261,120 at any
width dividing 1536 evenly. The bad width cut the cooperative ceiling by a third
and pushed programs that should have run as one launch onto the host split
route, turning one dispatch into many on a workload whose measured cost is
already dominated by host-side launch preparation.

Cold start now prefers the candidate maximizing resident threads per compute
unit, breaking ties toward the wider group. On this device that selects 512: 3
blocks per SM, 1536 resident threads, zero stranded slots, and the full 261,120
lane ceiling. `VyreGridSyncAot` confirmed the selection against the hardware
with `cuOccupancyMaxActiveBlocksPerMultiprocessor` on a real emitted kernel
rather than by arithmetic.

This is a residency rule and not a rule against 1024. Where the per-SM budget
divides evenly by 1024, such as a 2048-thread SM, 1024 strands nothing, ties on
residency, and the latency estimate still selects it.

Three protections are pinned by test. Callers that pin geometry are unaffected:
`workgroup_override` and `grid_override` keep their existing precedence, which
is why `exatok`, which sets both, never saw this. Measured feedback still
outranks the preference, so a real timing can select a width cold start would
never propose. And a backend that reports no per-SM budget stays byte-identical
to previous behavior: `LaunchGeometryLimits::max_threads_per_sm` of `0` means
unreported, the residency methods answer `None` rather than a guessed `0`, and
the candidate filter is inert, so wgpu selects exactly what it selected before.

The residency division now has one definition in the workspace,
`vyre_driver::validation::blocks_per_compute_unit`, which CUDA's cooperative
preflight in `vyre-driver-cuda/src/occupancy.rs` also routes through. Two copies
of this arithmetic had already drifted apart once. The shared function models
the thread ceiling only and documents that a caller answering "does this
DECLARED width fit" must additionally clamp by the device-reported block cap
(`CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR`, 24 on this device): at
width 32 the thread arithmetic predicts 48 blocks and 1536 resident threads
where the hardware delivers 24 and 768. Selecting the widest survivor never
reaches that regime, and the 512 this now picks sits at 3 blocks per SM, well
clear of the cap.

BREAKING: `vyre_driver::validation::LaunchGeometryLimits` gains a public
`max_threads_per_sm: u32` field. The struct is not `#[non_exhaustive]`, so every
struct-literal construction site must add it. Use `0` for a backend that does
not report a per-SM thread budget, which preserves prior behavior exactly.

### Fixed: cooperative preflight admitted grids the driver refuses (`vyre-driver-cuda`)

Two independent per-SM ceilings govern cooperative residency and
`cooperative_thread_residency_block_limit` respected only one. It derived
admissible blocks from the per-SM THREAD budget, `max_threads_per_sm /
workgroup`, while hardware separately caps BLOCKS per SM. At narrow widths the
block cap binds first: on an RTX 5090 reporting 24 blocks per SM, width 32 was
admitted at 1536/32 = 48 blocks per SM and 8160 blocks device-wide against a real
24 and 4080. The preflight answered "fits" and `cuLaunchCooperativeKernel` then
refused the launch, which is exactly the predicate-versus-driver disagreement
that giving the residency division one definition was meant to eliminate. It is
reachable rather than theoretical, because grid-sync programs are exempt from
launch-width tuning, so a declared 32 survives to launch.

The limit now clamps by a probed `CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR`.
Widths of 64 and up are unchanged on this device: 1536/64 = 24 is exactly the
cap, which makes 64 the narrowest width still reaching full occupancy and leaves
it no margin. A driver that does not report the attribute stores `0`, which reads
as unreported and applies no clamp, so behavior there is byte-identical to
before. A negative value is treated as unreported too, rather than cast into a
cap near four billion that would clamp nothing.

Measured, not calculated: per-width occupancy came from
`cuOccupancyMaxActiveBlocksPerMultiprocessor` on a real emitted vyre kernel at
each candidate width (10 registers per thread, zero static shared memory, element
count a multiple of every width so tail waste could not skew it). The table and
its method are recorded in the `vyre-driver-cuda::occupancy` module documentation
so the next reader finds measurements instead of re-deriving arithmetic: width 32
gives 24 blocks per SM and 768 resident threads, widths 64 through 512 all reach
1536, and width 1024 gives 1 block and 1024.

### Changed: the cooperative grid-barrier release order is now unrepresentable to get wrong (`vyre-driver-cuda`)

Four launch sites hand-wrote the same sequence: run the launches in a closure,
release the lease, then propagate the launch error. The order is load-bearing and
the reason is not local. `GridBarrierGuard` frees the gate on drop, including on
unwind, so the gate can never be permanently stranded; the hazard is the
opposite. Releasing through `Drop` instead of through the release path frees the
gate while SKIPPING both the stream synchronize and the arrival audit. The next
sequence then acquires the gate and zeroes `_vyre_grid_barrier` underneath a grid
that may still be running, whose remaining barriers wait for a release target
that can no longer be reached. That is a hang rather than an error, it reproduces
only under cooperative launch, and the edits that cause it (hoisting `launched?`
above the release, or deleting the closure so `?` returns directly) both compile
and keep every non-cooperative test green.

`GridBarrierLease::launch_then_release` now owns that order. It consumes the
lease, so a call site cannot skip the release, and it captures the launch
closure's error rather than letting it escape, so no error path can bypass the
synchronize. `release_after_launch` is private to the module, so an open-coded
release will not compile from another module. The release delegates to a small
ordering function whose synchronize and audit steps are closures, which lets unit
tests assert the gate is still HELD while the synchronize runs and freed only
after the release returns, pinning the ordering rather than the end state.

### Changed: the synthetic device profile no longer claims to be real hardware (`vyre-driver-cuda`)

`blackwell_sm120_caps()` named "NVIDIA GeForce RTX 5090" and its module
documentation called it a source of truth, while five of its fields disagreed
with that machine and the three substantial ones all OVERSTATED it: 2048 threads
per SM against a measured 1536, 256 KiB of shared memory per SM against 100 KiB,
and 128 KiB per block against 48 KiB, the last being unreachable even with the
99 KiB opt-in maximum. Every occupancy figure derived from it was therefore
optimistic, and a number that reads as measured and is not is the defect class
that also produced a test pinning a stale cooperative ceiling.

The values are unchanged, deliberately. The roughly twenty tests that use it are
correct arithmetic against a fixed envelope, and an estimator test needs a fixed
envelope rather than a true one: a test pinning `2048 / 256 = 8` checks division,
not any GPU, and rewriting those to chase real hardware would churn them again on
the next device. So the fix is the name and the claims. It is now
`synthetic_sm120_envelope`, its documentation states plainly that it is a test
fixture whose values are not this machine's, names the specific divergences, and
forbids deriving a hardware decision from it. Verified that no shipping path
consumes it: every caller in the workspace is inside `#[cfg(test)]` or under
`tests/`. It also gains a blocks-per-SM value, chosen so the cooperative clamp
above is exercised without a CUDA context.

BREAKING: `blackwell_sm120_caps`, `blackwell_sm120_caps_default` and
`BLACKWELL_SM120_DEFAULT_MEMORY_BYTES` are renamed to `synthetic_sm120_envelope`,
`synthetic_sm120_envelope_default` and `SYNTHETIC_SM120_DEFAULT_MEMORY_BYTES`.
`CudaDeviceCaps` gains a public `max_blocks_per_sm: i32` field; the struct is not
`#[non_exhaustive]`, so struct-literal construction sites must add it, and `0`
means unreported.

### Fixed: `Node::Return` was silently discarded instead of emitted or refused (`vyre-emit-ptx`)

The PTX emitter handled `Node::Return` with a comment and no instruction.
`finish_with_return` writes the single trailing `$L_exit:` / `ret;` at the end of
the kernel, so a `Return` nested in an `If` or a loop emitted NOTHING and fell
through, and the program kept running past its own exit. The branch target
already existed and `Trap` already branched to it, so this was a missing match
arm rather than a missing mechanism.

A dropped control-flow node is a correctness defect, not a performance one. It
happened to be survivable wherever the work after the exit was idempotent, which
is why every correctness test in the tree passed while it was broken: the answers
were right and only the work was wrong. A consumer whose loop body is NOT
idempotent after its exit condition got a wrong answer from an exit the emitter
quietly deleted, with nothing reporting it.

A `Return` now lowers to `bra $L_exit`, and the emitter REFUSES the cases it
cannot honor instead of dropping them. This half matters more than the branch. A
`Return` taken by only SOME invocations lets those invocations leave while the
rest continue, and the ones that left can never arrive at a later `bar.sync` or
cooperative grid barrier, so the ones that stayed block forever. Trading an
invisible slowdown for an invisible hang would not have been a fix, so the
emitter proves the exit is uniform across the grid or refuses at compile time,
naming the reason and the fix.

Uniformity is proven, never assumed: values built from literals, buffer lengths,
the subgroup size, and loads from global or constant memory at a uniform index
qualify. Anything derived from an invocation id, a workgroup id (uniform within a
CTA but not across the grid, and a whole CTA leaving early strands the others),
a subgroup op, shared memory, or an atomic's returned value does not. A loop whose
bounds are not uniform also counts as divergent, because invocations then leave it
on different iterations even with no conditional present. Unproven is treated as
varying, so the failure direction is a build error rather than a hang.

`vyre-emit-ptx/tests/nested_return_branch.rs` pins both halves, including a
control proving the asserted branch comes from the `Return` and not from the
entry's predicated element-count guard, which also branches to `$L_exit`.

### Added: `persistent_fixpoint_grid`, a grid-correct convergence loop (`vyre-primitives`)

`fixpoint::persistent_fixpoint` drives convergence from an in-kernel
`Node::Loop`. Lane 0 clears the single shared `changed` word once per
iteration, ordered only by a `MemoryOrdering::SeqCst` barrier, which is
workgroup scope, while every lane in every workgroup sets that same word with
`atomic_or`, and each workgroup gates its own `Node::Return` on it.

Above one workgroup nothing orders one group's clear against another group's
set. The severe face is a lost set: the clear erases a flag another group had
already raised, that group reads 0, returns early, and leaves its slice of the
state unconverged with no error. For a caller whose convergence means "no work
remains" that is a wrong answer, not merely wasted work. The mild face is a
false verdict: a downstream GPU tokenizer measured correct state and a correct
two-pass convergence with `changed` still reporting non-zero against a
fifteen-pass budget, which is indistinguishable from a real cap-out.

At ONE workgroup the same code is ordered and does not lose a set: the
sequence is clear, barrier, sets, barrier, barrier, read, so the conflicting
accesses to `changed[0]` are never concurrent and a CTA-scope fence is
sufficient. An intermediate revision of this entry claimed the clear made the
builder unsound at one workgroup as well, and that was wrong; it is corrected
here because a consumer selects this builder for its single-group path and the
claim would have implied a live exactness defect there.

`fixpoint::persistent_fixpoint_grid` takes the same positional parameters,
buffer names, bindings, and workgroup size, so selecting between the two is a
`match` on group count. It replaces the in-kernel loop with `max_iterations`
top-level waves separated by `MemoryOrdering::GridSync` barriers, the shape
`persistent_bfs_grid_sync_parallel` already uses for the same reason. Each wave
is five nodes: the caller's transfer body, a grid fence, the per-word compare
and ping-pong, a grid fence, and `if changed[i] == 0 { Return }`.

The early exit survives the grid barrier protocol because it is collective.
`changed` carries one word per iteration and is NEVER cleared, so a set cannot
be lost, and the word is read only after a grid fence, so every group computes
the same verdict and the whole grid leaves together or none of it does. Do not
collapse the per-iteration word back to one cleared word; that reintroduces the
race and turns the return into a stranding hazard in a single edit.

One ABI difference: `changed` is `max_iterations` words wide instead of 1 and
the caller must supply it zero-filled. In exchange the array decodes the pass
count exactly. `changed[i] == 1` iff wave `i` changed the state and the kernel
leaves at the first zero, so iterations entered is the index of the first zero
plus one, or `max_iterations` when no word is zero.

`persistent_fixpoint_grid` also carries a cooperative-residency ceiling that
`persistent_fixpoint` does not, because a `GridSync` lowers to a native
cooperative launch that needs every block co-resident. A dispatch path that
cannot provide it must refuse, naming the block count and the device limit;
`VyreBackend::cooperative_grid_sync_fits` is the preflight and
`VyreBackend::allows_host_grid_sync_split` says whether the kernel-split
fallback is permitted at all. A silent reroute there is a correctness failure.

`persistent_fixpoint` is unchanged: its emitted IR, signature, and first-zero-read
pass semantics all stay as they were, because downstream pass-count bounds are
denominated in them. Its module doc claimed convergence required `changed` to
read zero on two consecutive iterations, which the code never did; that text was
corrected to describe the first-zero-read exit it actually implements. The shared
`[256, 1, 1]` geometry both builders emit is now the exported
`PERSISTENT_FIXPOINT_WORKGROUP_SIZE`, so a caller derives its routing threshold
from the declared geometry instead of a literal.

`vyre-primitives/tests/persistent_fixpoint_grid_contracts.rs` pins all of it,
including a differential test that runs both builders through the reference
interpreter across four transfer bodies and every budget, and a probe that steps
the workgroups back to front: the grid builder returns the same state and the
same verdict in either order, while `persistent_fixpoint` at two workgroups
reports `changed == 1` forward and `changed == 0` reversed for the same input,
which is the race made deterministic.

The same-location property is pinned structurally by
`the_grid_builder_never_writes_changed_with_a_plain_store`, which asserts at
four budgets that no `Node::Store` targets `changed` and that each wave `i`
atomic-ors exactly word `i`, so the zero count means "all atomic" rather than
"nothing written". It has to be a structural assertion on the emitted IR
because the reference interpreter does not model L1 against L2 and cannot
reproduce the hardware race, so a reintroduced clear would keep every
value-level test green. The same test asserts that `persistent_fixpoint` still
shows exactly its one plain clear, which proves the probe detects a plain
store when one is present instead of matching nothing.

The collective exit is honored on PTX, which was NOT true when this primitive
was written. The emitter used to discard a nested `Node::Return`, so every
emitted wave ran regardless of how early the grid converged; that is fixed in
this same release (see the `vyre-emit-ptx` entry above) and a three-wave build
now emits three unpredicated `bra $L_exit` instructions, one per wave. The
`changed` decoding is unaffected either way, because a skipped wave and a wave
that changed nothing both leave their word at 0.

The exit saves LAUNCHES only under a native cooperative launch, which bounds
that guarantee. `GridSync` lowers either to a cooperative grid barrier or to a
kernel split, and under the split each wave is its own launch, so a
`Node::Return` in segment N returns from that launch alone and cannot stop the
host issuing segment N+1 (`vyre-driver/src/grid_sync.rs` dispatches every
segment in order). A run converging at wave 2 of a 16-wave budget still issues
all `2 * max_iterations + 1` segments. The ANSWER is unaffected on that path,
since a converged wave recomputes the same `next`, sets no flag word, and
copies idempotently, so only the saved work disappears and a device-side pass
counter reads the full budget instead of the convergence depth, which looks
like a cap-out and is not one. A downstream caller measured exactly that with
byte-correct state and a correct `[1, 0, 0, ...]` flag buffer. Read convergence
depth from `changed`, which is authoritative on both paths, never from a pass
or launch count. Pinned by
`the_split_path_launches_every_wave_because_return_is_per_segment`, which
asserts the segment count and that the exits are spread across segments rather
than concentrated in one that could short-circuit the host loop.

A budget sweep from 2 to 256 confirmed that the IR, the pre-lowering optimizer,
and the emitted PTX each preserve exactly one exit per wave at every budget, so
no stage drops the exit at any particular wave count.

This primitive also satisfies the emitter's new uniformity requirement by
construction: `changed[i]` is read from global memory at a literal index, which
is grid-uniform, and it is read after a grid fence, so every invocation computes
the same verdict. An exit gated on anything per-invocation is refused at compile
time rather than silently dropped.

Both builders now document a caller requirement that was previously implicit:
the transfer body must write EVERY word `w < words` of `next` on every
iteration. Violating it is a wrong-answer defect that reports success, so
nothing in the run looks wrong. The compare-and-copy step writes
`current[w] = next[w]` for every `w`, not only the words the body touched, so
a word the body never wrote overwrites `current[w]` with a stale `next`; the
buffers then agree everywhere and the loop exits converged on corrupted state.
Pinned by `a_transfer_body_that_skips_words_silently_corrupts_them`, which
asserts the exact bytes both ways: state `[9, 0, 0, 0]` from seed
`[9, 9, 9, 9]`, with `changed` reading `[1, 0, 0, 0]`, a converged verdict.
The docs first claimed this shape would fail to converge instead; the test
falsified that and both doc blocks were corrected to the measured behavior.

### Changed: `persistent_fixpoint` clears its flag atomically (`vyre-primitives`)

That clear was a plain non-atomic `Node::store` to a word every other write
reaches through `atomic_or`. It was correct, because the barriers around it
ordered the clear against the sets, but only for that reason, and the
dependency is invisible at the call site: weaken or move one of those barriers
and the program breaks without anything correctness-shaped being edited. This
primitive already has a failure mode that reports converged while being wrong,
so a write whose safety rests on an unstated ordering assumption is a poor
thing to leave in it.

The clear is now `Expr::atomic_exchange` writing 0 to the same location, so
every write to `changed` in both builders is an atomic. In the emitted PTX the
clear is an `atom.global.exch` instead of a plain `st.global.u32` against an
`atom.global.or.b32` at the same address. Cost is one lane one operation per
iteration. Values and pass counts are unchanged, which the existing
convergence-equivalence and both-builders differential tests confirm, so
callers denominated in this builder's pass counts are unaffected.

This does NOT make the builder multi-workgroup safe. The race is about barrier
SCOPE, not atomicity; above one workgroup use `persistent_fixpoint_grid`. The
property is pinned by `neither_builder_writes_changed_with_a_plain_store`,
which asserts no `Node::Store` targets `changed` in either builder and then
points the same predicate at `next`, which IS written by plain stores, so a
matcher that silently stopped matching could not make the test pass.

### Added: `FRONTIER_TO_QUEUE_WORKGROUP_LANES` (`vyre-primitives`)

`graph::csr_frontier_queue::frontier_to_queue` builds a deliberately
single-workgroup scan, so its declared workgroup size, the stride its lanes walk
`node_count` with, and the lane gate confining the scan to the first workgroup
must agree. They were separate literals, which is the shape that lets a fixed
workgroup declaration drift away from its lane gate and produce silent duplicate
coverage above one workgroup. They are now one exported constant.

### Fixed: a writable buffer declared without a count was mis-sized, and the CPU oracle accepted programs every real target rejected (`vyre-foundation`, `vyre-driver`, `vyre-driver-wgpu`)

A writable `BufferDecl` declared without `.with_count(n)` produced either a
zero-length buffer or a correctly sized buffer with a zeroed tail on the WGPU
backend, while CUDA and the CPU reference both sized it from the declared byte
range. `dynamic_element_count_from_bytes` and `output_binding_layout_parts` are
now exported from `vyre-driver` so the WGPU backend derives the element count by
that same shared rule instead of its own.

The worse half was a certification hole. A buffer the backend allocates itself
(`BufferDecl::output`, any `WriteOnly`, or a `pipeline_live_out` ReadWrite)
declared without `.with_count(n)` has no host bytes to take its size from, so
nothing can size it and the only correct answer is refusal. The CPU reference
instead answered a countless `BufferDecl::output` with `Some([])` while CUDA and
WGPU both refused it, and answered a countless `WriteOnly` with `Some([])` while
CUDA refused, so a program could pass the oracle and then be rejected by every
real target. `BufferDecl::require_static_readback_size` is now the single
refusal, called from both the execution planner and `vyre-reference`, so the
oracle refuses exactly what the backends refuse. It is driven by
`is_backend_allocated_output()` rather than the narrower `is_output()`, which is
what brings `WriteOnly` and `pipeline_live_out` under the same rule.

Reference, CUDA and WGPU now return byte-identical output for a countless
ReadWrite at every length from 1 to 4096, and all three refuse the
un-inferable forms naming `.with_count(n)`.

### Fixed: the CPU reference sentinel could fail open and return an empty result as success (`vyre-foundation`)

`is_cpu_reference_sentinel` identifies an op whose CPU lowering is only the
structured-reference sentinel, and that comparison sits in front of a refusal.
It compared function addresses, and a function's address is not a unique
identity: with more than one codegen unit the compiler may materialize a second
copy or a local thunk, so `fn_addr_eq` compares two different addresses for the
same function and answers `false`. The dispatcher then stopped refusing and
INVOKED the sentinel, which clears the output and returns `Ok(())`, handing the
caller an empty byte vector that looks like a successful CPU reference result.

The identity is now the exported `SENTINEL_CPU_REF` static, which holds a single
pointer resolved once, so a producer that stores it and a consumer that compares
against it compare the same bits by construction.

Two of this release's fixes share one shape, and it is worth naming as a class:
a refusal degrading into an empty output returned as success. The
countless-buffer defect above did it three ways (an empty readback, a zeroed
tail, and a reference oracle answering `Some([])`), and the sentinel did it by
invoking the very lowering it was meant to refuse. This class survives a test
suite because the call reports `Ok` and the output has a plausible shape, so
only an assertion on exact bytes catches it while a shape-only or `is_empty`
check passes happily. A refusal that stops refusing does not throw, it succeeds,
so a refusal path is covered only when the test asserts the refusal AND its text,
never merely that the call returned, with a counted control beside it so a
blanket rejection cannot pass as a fix either.

Asserting a zero length is NOT sufficient, which is the trap in the obvious
reading: a legitimately empty result is indistinguishable from this bug in
isolation. Only a contrast discriminates, the same declaration returning 0 bytes
for an empty seed and exactly 256 for a 256-byte seed, asserted together.

### Added: `FusionWorkgroupGeometryError` (`vyre-foundation`)

A fused launch takes the axis-wise maximum of its arms' workgroup sizes. That is
harmless for an arm whose invocations are independent, and unsafe for an arm that
synchronizes its workgroup or keeps state in workgroup memory: an arm written for
4 invocations guards its body so the other 252 skip it, which makes the workgroup
barrier non-uniform, and its workgroup buffers are sized for the narrow geometry.
The observed symptom was an inclusive prefix scan built for 4 elements, fused
behind a 256-wide elementwise arm, returning the wrong last lane on roughly one
dispatch in ten.

Fusion now refuses that pairing with a typed error naming the arm index, the
geometry it was built for, the geometry the fused program would run it under,
what makes the widening unsafe, and the fix.

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
  Persistent megakernel runtime loops on host-fed ring slots for typed
  Programs (not a general VIR bytecode interpreter). Linux-only NVMe
  zero-copy via raw `io_uring_setup` + mmap of SQ/CQ rings, with a
  `uring-cmd-nvme` feature for `IORING_OP_URING_CMD` passthrough
  (kernel 6.0+). Three-buffer layout (control / ring / debug_log),
  256-lane × N-workgroup sharding, opcode extension hook for vendor
  intrinsics, per-tenant authorization masks, atomic `done_count`
  counter, and a PRINTF debug channel.
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
