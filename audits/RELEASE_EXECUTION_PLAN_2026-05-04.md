# Vyre major release execution plan — 2026-05-04

Scope: finish Vyre for the major release excluding the C parser lane, which is owned separately.

## Release target

Vyre ships as a GPU compute substrate with a serious persistent megakernel execution path, a GPU-cost-aware optimization pipeline, backend execution hooks, conformance coverage, and reproducible speed proof. The release story is no longer the old GPU bytecode rule condition engine.

## Owned scope

- Megakernel runtime, queueing, scheduler, telemetry, replay, and resident execution.
- GPU optimization stack across IR, descriptor, runtime, and backend layers.
- Backend execution paths for CUDA, wgpu, SPIR-V/reference where release-relevant.
- Conformance, parity, release gates, and signed certificate flow.
- Benchmarks proving speed and exposing bottlenecks.
- Release claim audit and docs outside parser-specific claims.

## Phase 1 — truth baseline

Goal: identify exactly where time is going.

Deliverables:

- Benchmark suite for many tiny jobs, medium batches, fused batches, cache-hot dispatch, cache-cold dispatch, readback-heavy jobs, and megakernel queue throughput.
- GPU probe gate using `nvidia-smi`, CUDA device count, and wgpu adapter enumeration.
- One release speed-proof benchmark command.
- Timing split for compile, pipeline cache lookup, dispatch submit, queue publication, GPU execution, readback, and host allocation/copy.
- No "skipped: no GPU" paths in release gates.

## Phase 2 — execution planner

Goal: make execution strategy explicit and optimizer-selectable.

Deliverables:

- Central execution plan object for runtime decisions.
- Strategy enum for direct dispatch, batched dispatch, persistent megakernel, AOT artifact, and reference shadow.
- Cost model inputs from `DeviceProfile`, program shape, batch size, buffer layout, readback needs, cache state, and queue pressure.
- Debug overrides for direct, megakernel, and backend selection.
- Structured explanation for strategy selection.

## Phase 3 — megakernel v2

Goal: make megakernel the high-throughput strategy most workloads naturally select.

Deliverables:

- Work queue packing that removes redundant slots before publication.
- Multi-op packed slots used by real dispatch paths.
- Persistent compiled megakernel reuse across batches.
- Resident buffer plan so repeated dispatches avoid re-upload and rebind churn.
- Queue telemetry for published, claimed, done, requeued, occupancy, pressure, and stalls.
- Priority and continuation path wired into runtime decisions.
- Sparse readback path for hit-like workloads.
- Differential replay log tied to dispatched slots.
- Bench proving persistent megakernel beats direct dispatch for tiny and medium batched workloads.

## Phase 4 — optimization stack upgrade

Goal: optimize for GPU economics, not CPU-style node count.

Deliverables:

- Pass ordering reviewed against GPU cost.
- Cost penalties for dispatch count, pipeline miss, readback, host copy, bind churn, divergence, global atomics, uncoalesced memory, and tiny kernels.
- Cross-region fusion with region metadata preserved for debugging.
- Descriptor passes for coalescing, shared memory, branch flattening, dead stores, load forwarding, constant buffers, and tail masks.
- Runtime-level optimization over batches and megakernel queues.
- Optimization telemetry: op count, dispatch count, readback bytes, binding count, queue slots, and waste score.

## Phase 5 — backend execution cleanup

Goal: CUDA proves speed, wgpu proves portability, reference proves semantics.

Deliverables:

- CUDA resident allocation/upload/dispatch/readback/pipeline reuse audit and fixes.
- wgpu bind-group cache, buffer pool, and readback staging reuse audit and fixes.
- SPIR-V emission/parity kept honest where applicable.
- Reference remains byte oracle only.
- Backend selection is explicit, inventory-driven, and failure-loud.
- Unsupported backend features return structured actionable errors.

## Phase 6 — conformance and parity

Goal: speed changes must preserve semantics.

Deliverables:

- Conformance runner covers megakernel-visible work item paths where possible.
- CPU reference parity for optimized vs unoptimized programs.
- Backend parity for CUDA/wgpu/reference on the release corpus.
- ULP contracts explicit for float ops.
- No universal skip/exemption path.
- Signed cert generation works after runtime changes.
- Every new optimization has semantic preservation tests or a bounded invariant.

## Phase 7 — release blockers

Goal: remove anything that makes the release claim false.

Deliverables:

- Zero stubs in touched megakernel, optimization, and backend paths.
- No hidden CPU fallback in GPU-marked paths.
- No unbounded queues, caches, reads, or allocation loops.
- Every touched public error has actionable `Fix:` text.
- Every changed public surface is documented or intentionally private.
- Existing warnings in touched release paths fixed when they intersect scope.

## Phase 8 — docs and migration story

Goal: stop describing old Vyre.

Deliverables:

- README reflects current substrate/runtime architecture.
- Release notes explain the jump from bytecode engine to IR/runtime stack.
- Execution strategy docs explain when megakernel is selected.
- Optimization pipeline docs explain cost model and hooks.
- Benchmark doc has reproducible hardware and commands.
- Parser claims coordinated with the parser lane owner.

## Immediate execution order

1. Build the benchmark truth harness around current megakernel dispatch.
2. Identify largest fixed overheads.
3. Convert the highest-cost megakernel path from rebuild/republish/readback-heavy to reuse/pack/resident/telemetry.
4. Run narrow tests after each substantial change, then full touched-crate tests.

## Execution notes - 2026-05-04

- Added megakernel truth benchmark `runtime.megakernel.truth.1024` with queue-plan, queue-publish, backend-dispatch, lineage, processed, remaining, published, and deduped metrics.
- Fixed WGPU dispatcher to publish all requested work after fusion planning; fusion selection is no longer allowed to filter the queue.
- Added runtime-safe duplicate WorkItem pruning and exact logical completion accounting for retained redundant work.
- Capped inline lineage/fusion graph construction for hot queues above 256 items to prevent O(n^2) host planning on release paths.
- Cached one-shot sharded empty megakernel Program templates behind shared Arcs.
- Replaced per-slot WGPU publication with `Megakernel::encode_work_items_ring_into`, a validated bulk WorkItem ring encoder.
- Added dense-output fast negative dedupe path; `megakernel_queue_plan_ns` dropped from ~228 us p50 to ~10.6 us p50 on the 1024-item truth benchmark.
- `megakernel_queue_publish_ns` dropped from ~446 us p50 to ~272 us p50 on the same benchmark after bulk publication.
- Final 30-sample truth run passed: GPU p50 803,214 ns, p99 1,238,875 ns. Top-line runtime is now dominated by WGPU backend dispatch/readback latency, not host queue planning.
- Validation passed: `vyre-foundation --lib`, `vyre-runtime --lib`, `vyre-driver-wgpu --lib`, and `vyre-bench --lib`.
