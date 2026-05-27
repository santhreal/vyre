# Vyre Release Completion Plan

This is the concrete execution plan for finishing Vyre after the benchmark
meta-harness work starts in parallel.

`vyre-bench/` and removal of the old scattered benchmark surface are owned by
the benchmark agent. All other Vyre core, runtime, backend, reference,
conformance, gate, and performance work is covered here.

## Definition Of Done

Vyre is done for this sweep when:

1. Workspace metadata resolves from the Vyre root and parent Santh workspace.
2. Core crates compile with tests where applicable:
   - `vyre-foundation`
   - `vyre-driver`
   - `vyre-runtime`
   - `vyre-driver-wgpu`
   - `vyre-reference`
   - `vyre-primitives`
   - `vyre-libs`
   - `vyre-intrinsics`
   - `conform/vyre-conform-runner`
   - `conform/vyre-test-harness`
3. No old `vyre-runtime` crate or import remains.
4. Runtime owns megakernel execution under `vyre-runtime/src/megakernel`.
5. CUDA and SPIR-V are explicitly experimental while remaining loud gates.
6. WGPU is the production backend path and has no silent CPU/no-GPU fallback.
7. Foundation structural gates pass.
8. Known correctness mismatches are fixed or rejected by validation before
   backend lowering.
9. Known hot-path performance findings in this plan are fixed.
10. No known TODO/deferred/future/0.7 marker remains without implementation.
11. No `todo!`, `unimplemented!`, or user-input panic remains in production
    paths.
12. No skipped/mocked test pretends to cover production behavior.
13. Core gates pass:
    - `cargo metadata --no-deps --format-version 1`
    - `scripts/check_architectural_invariants.sh`
    - `scripts/check_ownership_boundaries.sh`
    - `scripts/check_crate_metadata_normalized.sh`
    - `scripts/check_layering.sh`
    - `scripts/check_repo_split_readiness.sh`
    - `scripts/check_safe_crates_forbid_unsafe.sh`
    - `scripts/check_no_deferred_work.sh`
    - `scripts/check_gpu_test_loudness.sh`
    - `scripts/check_max_file_size.sh`

## Coordination Rules

- Benchmark agent owns `vyre-bench/` and old benchmark removal.
- This plan owns everything else.
- No agent changes `vyre-bench/` unless explicitly assigned by the benchmark
  owner.
- No agent deletes tests/docs/contracts to hide failures.
- No agent weakens assertions to match broken behavior.
- No compatibility shims for removed crates unless explicitly approved.
- Every changed subsystem must run at least one targeted compile/test command.

## Parallel Agent Plan

Use three broad agents, not many narrow agents. Each agent owns a subsystem and
must close all known items in that subsystem.

### Agent A: WGPU Backend And Runtime Hot Path

Owned paths:

- `vyre-driver-wgpu/src/**`
- `vyre-driver-wgpu/tests/**`

Do not touch:

- `vyre-bench/**`
- `benches/**`
- workspace `Cargo.toml` unless compile requires a WGPU-only manifest fix
- docs/contracts except WGPU-specific config docs

Concrete fixes:

1. Fix `LruPipelineCache` missing `.iter()` compile issue in
   `vyre-driver-wgpu/src/runtime/cache/pipeline.rs`.
2. Keep the single-map pipeline cache design; do not restore a second DashMap.
3. Ensure eviction still chooses by gain/cost without cloning artifacts.
4. Fix WGPU tests that still import old megakernel surfaces.
5. Remove or relocate WGPU tests that require `vyre-runtime` with `wgpu` feature
   and create package cycles.
6. Ensure `vyre-driver-wgpu` does not depend publicly on `vyre-runtime`.
7. Eliminate silent GPU acquisition fallback in tests and production helpers.
8. Replace fixed sleeps in WGPU tests/runtime with bounded spin/yield/park or
   device polling.
9. Audit `record_and_readback` for per-dispatch scratch allocations:
   - owned return paths using `Vec::new`
   - binding lookup scratch maps
   - output staging vectors
10. Extend existing dispatch scratch reuse so owned paths do not allocate
    avoidably.
11. Audit readback ring locking:
    - `ReadbackRingSet`
    - per-ring mutexes
    - small dispatch contention
12. Replace coarse locks where practical with sharded or atomic structures.
13. Audit validation VSA cache:
    - no global `RwLock<Vec<Vec<u32>>>`
    - use sharded `DashMap`/`DashSet` keyed by hash.
14. Ensure pipeline disk cache does not force physical fsync on every insert.
15. Remove static string allocation churn in Naga lowering where still present.
16. Ensure WGPU stats avoid unnecessary adapter name clones.
17. Ensure supported-op static strings are cached or interned.
18. Add or fix tests for device-loss recovery:
    - device lost flag set
    - recovery path reconstructs usable backend
    - pending readbacks fail deterministically
19. Add or fix tests for no skipped GPU path:
    - GPU probe failure is a loud error
    - no `skipped: no GPU` on configured GPU machines.
20. Run:
    - `nvidia-smi`
    - `cargo check -p vyre-driver-wgpu --tests`
    - WGPU targeted tests that compile after foundation is green
    - `scripts/check_gpu_test_loudness.sh`

Definition of done:

- `cargo check -p vyre-driver-wgpu --tests` passes or reports only an
  out-of-scope compile blocker with exact file/line.
- No old megakernel crate references remain in WGPU.
- WGPU hot path does not regain known per-dispatch allocation/lock bottlenecks.

### Agent B: Reference And Conformance

Owned paths:

- `vyre-reference/src/**`
- `vyre-reference/tests/**`
- `conform/vyre-conform-runner/src/**`
- `conform/vyre-conform-runner/tests/**`
- `conform/vyre-test-harness/src/**`
- `conform/vyre-test-harness/tests/**`

Concrete fixes:

1. Fix `vyre-reference/src/hashmap_interp/step.rs` snapshot mismatch:
   - `HashmapLocalSnapshot` expected
   - `HashmapLocals` found.
2. Ensure subgroup snapshotting does not clone full locals per invocation.
3. Use `im::HashMap` or a proper persistent snapshot/diff representation for
   `HashmapInvocation` locals.
4. Replace remaining `std::collections::HashMap` in hot or snapshot-adjacent
   reference paths with `im::HashMap` or `FxHashMap` as appropriate:
   - `hashmap_interp/memory.rs`
   - `hashmap_interp/sync.rs`
5. Remove eval-time `Vec<OpCode>` expression linearization or replace it with a
   no-allocation reusable/small stack path.
6. Keep recursive evaluation semantically equivalent.
7. Remove repeated error-string allocation in validation failure hot paths.
8. Fix integer divide/modulo semantics:
   - unsigned divide by zero deterministic contract
   - signed divide by zero rejected or trapped before backend
   - `i32::MIN / -1` rejected or trapped before backend
9. Ensure validation rejects unsafe divide/mod denominators when statically
   known.
10. Ensure reference and backend contract agree for subnormals:
    - either declared FTZ contract
    - or backend preservation strategy
    - tests must match the chosen contract.
11. Keep deterministic float reference:
    - no platform-varying `std::f32::sin/cos/exp/log` as oracle
    - explicit ULP budget for transcendentals.
12. Ensure CPU-vs-backend lenses use `max(entry.tolerance(),
    f32_ulp_tolerance(program))`.
13. Ensure integer outputs remain byte-identity compared.
14. Remove duplicate untyped comparators that can accidentally apply F32
    tolerance to integer buffers.
15. Ensure skipped tests are not hiding unsupported production paths.
16. Ensure mocks are not counted as backend parity.
17. Add negative tests for malformed calls, divide by zero, and unsupported
    U64/I64 backend semantics.
18. Fix certificate generation tests to use real parity artifacts.
19. Run:
    - `cargo check -p vyre-reference --tests`
    - `cargo test -p vyre-reference --lib`
    - `cargo test -p vyre-conform-runner --tests`
    - `cargo test -p vyre-test-harness --tests`

Definition of done:

- Reference builds.
- Conform builds.
- Known tolerance inconsistency is gone.
- Subgroup snapshot cost is structurally fixed.
- No fake skips/mocks stand in for production backend parity.

### Agent C: Runtime, Libraries, And Primitives

Owned paths:

- `vyre-runtime/src/**`
- `vyre-runtime/tests/**`
- `vyre-libs/src/**`
- `vyre-libs/tests/**`
- `vyre-primitives/src/**`
- `vyre-primitives/tests/**`
- `vyre-intrinsics/src/**`
- `vyre-intrinsics/tests/**`

Concrete fixes:

1. Finish runtime megakernel ownership:
   - `vyre-runtime/src/megakernel` is canonical
   - no driver-megakernel dependency
   - no duplicate megakernel module in WGPU.
2. Keep runtime megakernel organization clear:
   - protocol
   - dispatcher
   - planner
   - scheduler
   - telemetry
   - io
   - policy.
3. Ensure `vyre-runtime/src/io` owns runtime IO surfaces, not megakernel-only
   IO by accident.
4. Ensure `vyre-runtime/src/formats/gpujson.rs` is real, deterministic,
   aligned, and tested.
5. Add/verify GPU-readable format tests:
   - deterministic sorted key table
   - duplicate key rejection
   - truncated table rejection
   - alignment guarantees
   - round-trip decode.
6. Fix pipeline cache persistence across process restarts where runtime owns
   cache stores.
7. Ensure disk cache batching avoids serial fsync on every insert.
8. Finish runtime perf paths:
   - megakernel planning
   - protocol encode/decode
   - dispatch wall time
   - cache lookup.
9. In `vyre-libs`, close matching hot paths:
   - `dispatch_io::pack_haystack_u32`
   - DFA compile inherited outputs
   - literal-set JIT balanced byte tree
   - regex compile dummy allocations
   - mega scan CPU allocations.
10. In parsing, close clone storms:
    - `parsing/core/ast/shunting/operator.rs`
    - `precedence`
    - `ast_opcode`
    - token clone minimization.
11. In NN, replace correctness-only production paths where optimized versions
    are expected:
    - `linear_tiled` delegates to `matmul_tiled`
    - tiled softmax
    - tiled attention
    - optimized RMS norm.
12. In primitives, ensure matching/NFA/DFA structures avoid avoidable
    `Vec<Vec<_>>`, `HashSet`, and clone-heavy construction in hot builders.
13. Ensure subgroup intrinsic builders are enabled by default where workspace
    Naga version supports them.
14. Run:
    - `cargo check -p vyre-runtime --features wgpu,megakernel-batch --tests`
    - `cargo check -p vyre-libs --lib`
    - `cargo check -p vyre-primitives --lib`
    - `cargo check -p vyre-intrinsics --tests`

Definition of done:

- Runtime builds.
- Libs/primitives build.
- Runtime owns megakernel intelligence cleanly.
- Known domain hot-path findings are fixed or moved to failing tests with real
  implementation in progress, not comments.

## Main-Agent Plan

The main agent owns integration and cross-cutting fixes. It must not wait for
all agents before fixing obvious blockers in local scope.

### Phase 1: Stabilize Workspace Shape

Concrete tasks:

1. Stop touching `vyre-bench/`; benchmark agent owns it.
2. Stop touching old `benches/**` unless benchmark agent explicitly hands it
   back.
3. Close stale native agents.
4. Run `git status --short` and identify files changed by other agents.
5. Do not revert user/agent changes.
6. Fix Cargo package cycles:
   - `vyre-driver-wgpu -> vyre-runtime -> vyre-driver-wgpu`
   - WGPU dev-dep on runtime must not enable runtime `wgpu` feature.
7. Ensure parent Santh workspace metadata still resolves.
8. Run:
   - `cargo metadata --no-deps --format-version 1`
   - parent workspace `cargo metadata --no-deps --format-version 1`

### Phase 2: Foundation Compile And Structure

Concrete tasks:

1. Run `cargo check -p vyre-foundation --lib`.
2. Fix generated `Node::Barrier` construction if still broken.
3. Fix `strength_reduce` duplicate unreachable `BinOp::BitAnd`.
4. Finish `fact_substrate` split:
   - `fact_substrate.rs`
   - `fact_substrate/type_facts.rs`
   - `fact_substrate_tests.rs`.
5. Fix foundation oversized module gate:
   - `src/execution_plan/mod.rs`
   - `src/optimizer.rs`
   - `src/transform/autodiff/grad.rs`
   - `src/validate/nodes.rs`
   - `src/validate/expr_rules.rs`
   - `src/ir_inner/model/program/buffer_decl.rs`
   - `src/serial/wire/encode/to_wire.rs`
   - `src/execution_plan/policy.rs`.
6. Move inline tests into sibling test modules where that cleanly reduces file
   size.
7. Split helpers by responsibility where tests are not the reason for file
   size.
8. Run:
   - `cargo fmt -p vyre-foundation`
   - `cargo test -p vyre-foundation --lib fact_substrate -- --nocapture`
   - `cargo test -p vyre-foundation --test workspace_structure_contracts -- --nocapture`
   - `scripts/check_max_file_size.sh`.

### Phase 3: Foundation Semantics And Optimizer Performance

Concrete tasks:

1. Ensure `fingerprint_program()` uses canonical pipeline fingerprint bytes,
   not raw declaration-order-sensitive wire bytes.
2. Ensure canonical IR normalizer covers:
   - commutative operand sorting
   - nested block flattening
   - deterministic temp naming where applicable.
3. Ensure validation resolves `Expr::Call` op IDs when a dialect lookup is
   available.
4. Provide registry-aware validation entry point instead of relying on drivers
   to remember late validation.
5. Add memory ordering fields where missing:
   - `Expr::Atomic`
   - `Node::Barrier`.
6. Ensure wire tags encode/decode memory ordering.
7. Ensure default ordering is explicit and backward-compatible only through
   versioned decode.
8. Fix barrier semantics:
   - no silent downgrade from promised full barrier to workgroup-only barrier
   - subgroup barrier requirement reflected in capabilities.
9. Fix U64/I64 policy:
   - either full lowering support
   - or validation rejects unsupported 64-bit arithmetic before WGPU.
10. Fix divide/modulo undefined behavior by validation or deterministic lowering.
11. Remove panic paths:
    - `extension.rs` duplicate registration
    - `dialect_lookup.rs` duplicate registration
    - `validate/fusion_safety.rs` `unreachable!`
    - `primitives/common.rs` unsupported `CombineOp`.
12. Fix fusion O(n²) pending replacements where documented.
13. Ensure fusion uses shared fact-substrate var-use counts instead of
    rebuilding from scratch.
14. Reduce `dead_buffer_elim` Ident clone traffic in tight loops.
15. Reduce CSE key allocation by using cached Ident hashes/symbols instead of
    repeated `Arc<str>` allocation where possible.
16. Pre-size wire buffers and remove avoidable outer-envelope reallocations.
17. Run:
    - `cargo check -p vyre-foundation --tests`
    - targeted optimizer tests
    - `scripts/check_no_deferred_work.sh`.

### Phase 4: Repo-Wide Performance Instrumentation

Concrete tasks:

1. Add `vyre-foundation/src/perf.rs`.
2. Export `pub mod perf;`.
3. Implement:
   - `PerfScope`
   - `PerfSample`
   - `PerfCounter`
   - `PerfCounterSnapshot`
   - thread-local sample buffer/drain.
4. Make sample recording no-allocation in the hot path.
5. Add unit tests for:
   - counter accumulation
   - max update
   - scope finish idempotency
   - drain/reset behavior.
6. Wire instrumentation into:
   - optimizer scheduler per pass
   - wire encode/decode
   - runtime megakernel planning/dispatch
   - WGPU lowering/cache/compile/upload/dispatch/readback where compile allows.
7. Use stable phase names from `docs/VYRE_BENCH_META_HARNESS_PRD.md`.
8. Do not make `vyre-bench` dependency required by production crates.

### Phase 5: Integration With Worker Outputs

Concrete tasks:

1. Review WGPU worker output and integrate if not already present.
2. Review reference/conform worker output and integrate.
3. Ignore worker edits in `/home/mukund-thiru/Santh_work/...` unless copied into
   this workspace by the user; do not blindly import from a different tree.
4. Fix WGPU pipeline cache `.iter()` if still present.
5. Fix `HashmapLocalSnapshot` mismatch if reference worker has not fixed it.
6. Run targeted compile after each integration step.

### Phase 6: Final Scans

Concrete scans:

```bash
rg -n "TODO|FIXME|deferred|future|0\\.7|placeholder|stub|todo!|unimplemented!" \
  vyre-foundation vyre-driver vyre-driver-wgpu vyre-runtime vyre-reference \
  vyre-libs vyre-primitives vyre-intrinsics conform scripts contracts docs
```

```bash
rg -n "#\\[ignore\\]|skip|skipped|mock|fake|dummy" \
  vyre-reference conform vyre-driver-wgpu vyre-runtime vyre-libs vyre-primitives
```

```bash
rg -n "panic!|unreachable!|expect\\(|unwrap\\(" \
  vyre-foundation/src vyre-driver/src vyre-driver-wgpu/src vyre-runtime/src \
  vyre-reference/src vyre-libs/src vyre-primitives/src vyre-intrinsics/src
```

Every hit must be one of:

- fixed,
- test-only and valid,
- impossible to trigger from user input with a clear invariant,
- or converted to a typed error.

### Phase 7: Final Verification

Run in this order:

```bash
cargo metadata --no-deps --format-version 1
cargo check -p vyre-foundation --tests
cargo check -p vyre-reference --tests
cargo check -p vyre-runtime --features wgpu,megakernel-batch --tests
cargo check -p vyre-driver-wgpu --tests
cargo check -p vyre-libs --lib
cargo check -p vyre-primitives --lib
cargo check -p vyre-intrinsics --tests
cargo test -p vyre-foundation --test workspace_structure_contracts -- --nocapture
scripts/check_architectural_invariants.sh
scripts/check_ownership_boundaries.sh
scripts/check_crate_metadata_normalized.sh
scripts/check_layering.sh
scripts/check_repo_split_readiness.sh
scripts/check_safe_crates_forbid_unsafe.sh
scripts/check_no_deferred_work.sh
scripts/check_gpu_test_loudness.sh
scripts/check_max_file_size.sh
```

If a command fails, fix the failing code. Do not remove the command.

## Known Current Blockers To Close First

1. Benchmark agent is modifying/removing old bench surface; avoid those files.
2. `vyre-bench/` is partial and owned by benchmark agent.
3. WGPU pipeline cache iteration compile issue:
   - `vyre-driver-wgpu/src/runtime/cache/pipeline.rs`
   - `LruPipelineCache` missing `.iter()`.
4. Reference compile issue:
   - `vyre-reference/src/hashmap_interp/step.rs`
   - `HashmapLocalSnapshot` expected, `HashmapLocals` found.
5. Foundation structure gate still had oversized files:
   - `execution_plan/mod.rs`
   - `optimizer.rs`
   - `transform/autodiff/grad.rs`
   - `validate/nodes.rs`
   - `validate/expr_rules.rs`
   - `ir_inner/model/program/buffer_decl.rs`
   - `serial/wire/encode/to_wire.rs`
   - `execution_plan/policy.rs`.
6. `strength_reduce` has duplicate `BinOp::BitAnd` unreachable pattern.
7. Old `vyre-runtime` references must remain zero.
8. Cargo cycle risk:
   - `vyre-driver-wgpu`
   - `vyre-runtime`
   - WGPU dev-dependency feature configuration.

