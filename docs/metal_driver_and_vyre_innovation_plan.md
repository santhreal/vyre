# Vyre Metal Driver, Testing, Performance, and Innovation Plan

This document is the source of truth for making Vyre substantially stronger, starting with a first-class native Metal driver and continuing through correctness, performance, GPU execution depth, and novel capability work across the whole stack.

The goal is not to add another nominal backend. The goal is to make Vyre a serious heterogeneous execution system where Apple GPUs, NVIDIA GPUs, CPU reference execution, parsing, matching, graph/dataflow, and higher-level security workloads share one truth model and one optimization loop.

## Current state

Vyre already has the right architectural boundary for this work.

- `vyre-driver` owns the backend contract, registry, diagnostics, resident-resource API, cache interfaces, and backend capability model.
- `vyre-lower` owns the substrate-neutral lowering path from `Program` to `KernelDescriptor`.
- `vyre-emit-naga` owns Naga module construction for GPU-oriented targets.
- `vyre-emit-ptx` proves that a native backend-specific emitter can exist without contaminating the core model.
- `vyre-driver-wgpu` proves live GPU dispatch, but Apple GPU execution through this path is still mediated by WGPU.
- `vyre-driver-cuda` proves native-driver ownership, JIT/cache/device-specific planning, and direct runtime integration.
- `vyre-driver-reference` is the byte-truth oracle and must remain the correctness baseline.
- `vyre-emit-metal` owns the native Metal artifact seam: `KernelDescriptor -> vyre-emit-naga -> naga::back::msl -> structured native_module JSON`.
- `xtask compile --to native_module` now emits a Metal artifact through `vyre-emit-metal` instead of the historical placeholder error.

The native `metal` backend now compiles emitted MSL with Metal, manages borrowed and resident buffers, dispatches kernels, proves byte parity through the shared conformance runner, emits benchmark/telemetry artifacts, and supports compiled resident execution and zero-copy resource-output chaining.

## Implemented evidence ledger

Implemented slice:

1. `vyre-emit-metal` crate added to the workspace.
2. Workspace Naga features include `msl-out`.
3. `vyre-emit-metal` emits MSL through the shared `vyre-emit-naga` Lego path instead of duplicating lowering.
4. `vyre-emit-metal` emits structured `native_module` JSON artifacts with schema, target, emitter, MSL version, translated Metal entry point, descriptor hash, MSL hash, workgroup size, binding metadata, Naga `_buffer_sizes` sidecar metadata, and source.
5. `xtask compile --to native_module` routes through `vyre-emit-metal`.
6. `docs/targets.md` identifies `native_module` as artifact emission and keeps runtime dispatch separate from target artifacts.
7. `vyre-driver-metal` crate added to the workspace.
8. `vyre-driver-metal` owns the `metal` backend ID and submits inventory registration only on Apple targets.
9. Non-Apple `vyre_driver_metal::acquire()` fails actionably instead of fabricating a backend.
10. Apple `MetalBackend` dispatch path is routed through `BindingPlan`, `output_binding_layouts`, `enforce_actual_output_budget`, `vyre-lower`, and `vyre-emit-metal` instead of a duplicated Metal-specific ABI planner.
11. `vyre-primitives::graph::dominator_tree` exposes its CHK initialization, depth-recompute, and predecessor-intersection phases as child regions so LegoGate, `print-composition`, and composition discipline tests can inspect the real phase boundaries.
12. `vyre-emit-metal` records the actual translated MSL function name instead of assuming the logical Naga entry point survives unchanged.
13. `vyre-driver-metal` binds Naga's Metal `_buffer_sizes` sidecar from shared artifact metadata so bounds-checked Metal kernels can compile and run.
14. Apple-side `vyre-driver-metal` tests now prove native Metal acquisition, backend registration, dispatch capability, precedence, and one real output-buffer dispatch.

Validation evidence:

1. `./cargo_full test -p vyre-emit-metal` passed: 6 unit tests, 0 failures.
2. `./cargo_full test -p xtask` passed: 345 unit tests, 0 failures.
3. `xtask::compile::tests::native_module_target_emits_metal_artifact_json` proves the `native_module` target emits structured Metal JSON from an actual `Program`.
4. `vyre-emit-metal` tests prove deterministic artifacts, MSL emission, binding metadata, optimized emission stats, actionable missing-entry errors, and Metal slot-limit rejection.
5. `vyre-driver-metal` non-Apple tests prove unsupported acquisition is actionable and no fake `metal` backend is registered on non-Apple builds.
6. `./cargo_full test -p vyre-primitives dominator_tree` passed: 20 focused dominator tests, including the CHK phase-region regression test.
7. `./cargo_full run -p xtask --bin xtask -- gate1` passed: 607 ops audited, 0 failures.
8. `./cargo_full run -p xtask --bin xtask -- print-composition vyre-primitives::graph::dominator_tree` shows `init_state`, `recompute_depth`, and `intersect_predecessors` child regions.
9. `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-conform-enforce --test composition_discipline` passed: 7 tests, 0 failures.
10. `scripts/check_no_shader_assets.sh` passed: no shader asset files under `src/ops/**` or `src/dialect/**`.
11. `./cargo_full test -p vyre-driver` passed: 575 unit tests, integration tests, and 6 doc tests.
12. `ssh tt-macbook 'cd /Users/thiruthangarathinam/Santh/libs/performance/matching/vyre && CARGO_TARGET_DIR="$HOME/cargo-target" ./cargo_full test -p vyre-driver-metal'` passed on Darwin arm64: 2 Apple tests, 0 failures, including real Metal dispatch of a one-store `u32` program.
13. `./cargo_full test -p xtask native_module_target_emits_metal_artifact_json` passed after schema 2 update, proving the artifact exposes the translated entry point and `sizes_buffer_index`.

## Non-negotiable outcomes

1. Native Metal exists as a real `VyreBackend` implementation.
2. `metal` successful outputs are byte-identical to `cpu-ref`.
3. `native_module` emits a real Metal artifact bundle instead of an error.
4. Repeated workloads use resident buffers and persistent pipeline state.
5. MacBook validation runs through one scripted gate.
6. WGPU-on-Metal and native Metal are benchmarked against the same programs.
7. Vyre gains repeated correctness and performance gates across all active backends.
8. Innovations are wired into executable tests, benchmarks, or operator-visible behavior.

## Backend naming and target contract

The runtime backend ID should be `metal`.

The artifact target name should remain `native_module` because existing docs and CLI target naming already point there. This split keeps operator language clean:

- `--backend metal` means live native Metal execution.
- `--to native_module` means emit a native Metal artifact bundle.

The Metal backend should register through the existing `inventory` registry, with backend precedence above `wgpu` for Apple-native workloads once parity and resident execution pass.

## LegoGate source read and application

Source read for this plan:

1. `docs/lego-block-rule.md` is the canonical LegoGate source in this checkout.
2. The referenced `docs/LEGO_BLOCK_PHILOSOPHY.md` path is stale in this checkout; do not cite it as an active source of truth unless the file is restored.
3. `vyre-libs/tests/SKILL.md` is the concrete testing example for public `Program` producers, backend dispatch parity, matching decision tables, cache persistence, region/span dedup, and fixture reuse.
4. `vyre-libs/tests/skill_md_examples.rs` is the runnable example proving the `SKILL.md` decision-table, dispatch-helper, dedup, cache, rule-pipeline, regex-set, substring, and Aho-Corasick examples stay truthful.
5. `.internals/skills/testing/SKILL.md` is restored as the shared testing doctrine that crate-local `tests/SKILL.md` files inherit.
6. `.internals/skills/testing/property.md` is restored as the shared property-test contract referenced by `vyre-foundation/tests/SKILL.md`.

Actual source facts read from those files:

1. `docs/lego-block-rule.md` requires a search of `vyre-primitives/src/<domain>/` and `vyre-libs/src/{math,nn,hash,matching,parsing,text,security,logical}` before inventing any sub-op.
2. Gate 1 is reuse-driven. A large op split into private chunks is still a failure when the child regions are not registered reusable primitives.
3. The `attention` example names the correct shape: compose registered `matmul`, `softmax_step`, a second `matmul`, and `layer_norm_step` primitives instead of hiding the kernel behind local `attention_part_*` helpers.
4. The Molten visual example names the correct abstraction pressure: proposed `visual::separable_conv` became `math::conv1d`, pixel pack/unpack stayed Tier-1 bit operations, color interpolation stayed Tier-1 arithmetic, and the rounded-rect SDF stayed private because it had one caller.
5. `vyre-libs/tests/SKILL.md` requires every public `Program` producer to validate, round-trip through wire format, dispatch on every linked dispatch-capable backend, and match `cpu-ref` bytes for deterministic outputs.
6. The matching decision tables in `vyre-libs/tests/SKILL.md` are the testing model for other domains: tests must prove the router selects the simplest correct block, then escalates only when constraints require a richer engine.
7. The cache, persistence, region/span dedup, and fixture tables in `vyre-libs/tests/SKILL.md` are not examples to copy by hand; they identify shared production helpers that tests must exercise directly.
8. `vyre-libs/tests/skill_md_examples.rs` turns those documentation rows into executable contracts, so any plan section that cites a decision table must also name the runnable proof that keeps the table from drifting.
9. `vyre-libs/tests/skill_md_examples.rs` does not reimplement the documented helpers. It calls production `pack_haystack_u32`, `pack_u32_slice`, `scan_guard`, `unpack_match_triples`, `cached_load_or_compile`, `engine_cache_path`, `dedup_regions_reference`, `dedup_regions_inplace`, `build_rule_pipeline`, `compile_regex_set`, `substring_search`, and `aho_corasick`.
10. The executable example asserts both behavior and routing identity: helper outputs match exact bytes/spans, warm cache avoids recompilation, and generated `Program` regions expose the expected generator IDs.

The canonical LegoGate rule is not "make smaller files" and it is not "hide loops behind private helpers." The actual rule in `docs/lego-block-rule.md` is:

1. Before inventing any sub-op, search `vyre-primitives/src/<domain>/` and `vyre-libs/src/{math,nn,hash,matching,parsing,text,security,logical}` for an existing primitive.
2. Compose existing primitives first.
3. Promote a new primitive only when it has at least two real consumers, a stable API, and one domain-neutral concern.
4. Gate 1 is satisfied by registered child-region composition, not by bespoke private splitting.
5. A Tier-3 domain may keep one-caller helpers private; Tier 2.5 exists for reusable substrate, not domain glue.

The actual examples matter:

1. The `attention` example says the right answer is not `attention_part_a` and `attention_part_b`. The right answer is visible composition through registered `matmul`, `softmax_step`, and `layer_norm_step` primitives so `print-composition`, Gate 1, and optimizer fusion can all see the structure.
2. The `visual` example says domain language often lies. Proposed visual primitives dissolved into `math::conv1d`, Tier-1 bit/arith expressions, and one private SDF helper because only one caller needed it. Domain framing must be reduced to the underlying operation before adding a primitive.
3. The `vyre-libs/tests/SKILL.md` example says tests must prove valid programs, wire round-trip, dispatch on linked dispatch-capable backends, and byte equality with CPU reference. It also gives concrete matching decision tables for choosing the simplest fitting engine instead of building a new matcher.

Applied to this plan:

1. Metal must reuse `vyre-lower`, `vyre-emit-naga`, `vyre-emit-metal`, `BindingPlan`, `output_binding_layouts`, existing backend registry types, and existing output-budget enforcement. A Metal-only ABI planner is duplication unless the shared ABI contract cannot express a required Metal fact.
2. Any new optimizer or runtime helper must name the shared primitive it composes. If no primitive fits, the first implementation stays at the narrow owner boundary until a second real consumer exists.
3. Backend code may be split by responsibility, but those splits do not count as LegoGate success. The success criterion is visible shared substrate reuse and testable behavior.
4. New parsing, matching, graph, security, or ML capability must route through Tier 2.5 when reusable, and through Tier 3 composition when domain-specific.
5. Documentation, tests, and `print-composition` output must agree about which primitive owns each operation.

## Files and crates to add

Metal emitter crate:

- `vyre-emit-metal`
- Public API accepts `vyre_lower::KernelDescriptor`.
- Public API returns Metal source plus reflection metadata.
- The current implementation reuses `vyre-emit-naga` plus Naga MSL output.
- Direct MSL emission can be added behind the same API for constructs where Naga blocks coverage, performance, or exact control.

Add a native Metal driver crate:

- `vyre-driver-metal`
- Depends on `vyre-driver`, `vyre-lower`, `vyre-emit-metal`, and the Rust Metal bindings.
- Compiles and runs only on Apple platforms through target `cfg`.
- Exposes no fake success path on non-Apple hosts.

Wire the workspace:

- `vyre-emit-metal` is in the Vyre workspace.
- `native_module` emission is wired in `xtask`.
- The native `metal` backend crate must be added without creating a fake successful backend on non-Apple hosts.
- Add backend feature gates consistently with existing driver crates when the runtime backend lands.
- Add conformance entries for the `metal` backend when dispatch exists.

## Metal driver architecture

The driver should mirror the mature WGPU/CUDA split while remaining native to Metal.

Core types:

- `MetalBackend`
- `MetalDevice`
- `MetalDeviceProfile`
- `MetalPipeline`
- `MetalPipelineCache`
- `MetalBufferPool`
- `MetalResidentBuffer`
- `MetalDispatchPlan`
- `MetalTiming`
- `MetalArtifact`
- `MetalErrorContext`

Core flow:

1. Acquire `MTLDevice`.
2. Build `MetalDeviceProfile` from device name, registry identity when available, OS version, feature support, memory limits, max buffer size, max threadgroup memory, max threads per threadgroup, and counter/timestamp support.
3. Lower `Program` through `vyre-lower`.
4. Emit Metal source and reflection metadata through `vyre-emit-metal`.
5. Compile a `MTLLibrary`.
6. Resolve a `MTLFunction`.
7. Create a `MTLComputePipelineState`.
8. Build bind/resource metadata from reflection.
9. Allocate or reuse buffers.
10. Encode dispatch into a command buffer.
11. Commit, wait, and read back.
12. Normalize timing and diagnostics.
13. Return byte-exact outputs.

## Basic dispatch slice

The first executable path must run real programs:

1. Single input buffer, single output buffer.
2. Multiple input buffers.
3. Multiple output buffers.
4. Zero-length buffer handling.
5. Non-multiple-of-threadgroup-size lengths.
6. Bounds-protected writes.
7. Structured backend errors for unsupported programs.
8. Byte parity against `cpu-ref`.

The first slice is not complete until it can dispatch through the public `VyreBackend::dispatch` method and is discoverable through the registry as `metal`.

## Resident execution slice

Resident execution is required because repeated allocation and readback would hide most of the benefit of a native backend.

Required API coverage:

1. `allocate_resident`.
2. `upload_resident`.
3. `download_resident`.
4. Ranged resident download.
5. Batch resident upload.
6. Batch resident download.
7. Resident dispatch.
8. Borrowed dispatch into caller-provided buffers where the API supports it.
9. Cache invalidation on device loss or compilation option changes.
10. Deterministic cleanup for handles.

The resident path should become the primary benchmark path for Apple GPUs.

## Metal artifact bundle

`native_module` should emit a structured artifact, not just raw source.

Artifact contents:

1. MSL source.
2. Kernel entry name.
3. ABI metadata.
4. Binding map.
5. Expected input/output layout.
6. Workgroup shape.
7. Target feature assumptions.
8. Vyre program hash.
9. Lowered descriptor hash.
10. Emitter version.
11. Optional compiled library bytes if the platform API supports stable capture in the chosen format.

The artifact must be deterministic for the same program, emitter version, and target options.

## Validation on MacBook

MacBook validation should be scripted through environment variables rather than ad hoc shell history.

Expected environment:

```bash
VYRE_MACBOOK_SSH=macbook
VYRE_MACBOOK_VYRE_ROOT=/path/to/Santh/libs/performance/matching/vyre
```

Correctness gate:

```bash
scripts/check_metal_macbook.sh driver
```

Conformance gate:

```bash
scripts/check_metal_macbook.sh conformance
```

Benchmark gate:

```bash
scripts/check_metal_macbook.sh benchmark
```

Complete gate:

```bash
scripts/check_metal_macbook.sh all
```

If `cargo_full(workspace)` is unavailable on the MacBook, the fix is to install or expose the workspace-approved command there, not to silently switch the plan to an unapproved raw build path.

## Repeated correctness and performance loop

Each meaningful batch should land progress on at least three axes:

1. Detection truth.
2. Performance.
3. Test depth.
4. Dogfood or operator UX.
5. Organization and deduplication.
6. Architecture.

The loop:

1. Choose one real workload family.
2. Add or strengthen byte-truth tests against `cpu-ref`.
3. Run the workload through every available backend.
4. Measure compile time, enqueue time, device time, readback time, allocation count, cache hit rate, and output parity.
5. Remove one avoidable copy, allocation, synchronization, branch, or cache miss.
6. Add the regression test or benchmark that would catch the same mistake.
7. Record the backend-specific limitation as a typed capability, not prose.

This loop applies to Metal, CUDA, WGPU, SPIR-V, CPU reference, Weir, Surge, parsing, matching, graph execution, and any higher-level consumer of Vyre.

## Testing expansion plan

The testing part of this plan is a product requirement, not a polish pass. Every implemented slice needs a proving gate that exercises the real public boundary and makes duplication or fake wiring visible.

The testing plan is derived from the actual LegoGate and `vyre-libs/tests/SKILL.md` examples, not from a generic coverage checklist. A test only counts when it proves one of the product contracts below through a real seam.

Testing is now a required field for every implementation item in this plan. Each selected item must name:

1. The owning source contract: LegoGate rule, public backend contract, emitter artifact contract, resident-resource contract, benchmark schema, or CLI/operator contract.
2. The positive truth test that proves the real behavior.
3. The negative twin that proves fail-closed behavior.
4. The seam path under test, such as public `Program -> lower -> emit -> backend -> cpu-ref parity`, artifact write/read/validate, resident upload/dispatch/download, or benchmark report/validator.
5. The LegoGate proof when the item adds or changes a reusable primitive, including `gate1`, `print-composition`, duplicate-body checks, or caller tests naming the registered child primitive.
6. The production helper being exercised when the item touches packing, cache, content hashes, dedup, layout reflection, dispatch config, output sizing, backend timing, or benchmark reporting.
7. The benchmark or dogfood gate when the item claims speed, allocation reduction, cache behavior, backend selection improvement, or workload capability.

Testing has five jobs:

1. Prove byte truth against `cpu-ref`.
2. Prove the public seam is wired end to end.
3. Prove the implementation obeys the Lego-block rule instead of hiding duplicated work behind private helpers.
4. Prove helper selection follows the decision table: choose the simplest existing primitive that satisfies the workload before escalating to a richer engine.
5. Prove any cache, fixture, packing, or dedup helper used by tests is the production helper, not a test-only clone.

The actual Lego-block examples make the testing bar stricter:

1. An op that passes only because a large body was split into local chunks is still a failure. Tests must inspect registered `Region` composition through `gate1` or `print-composition` whenever a high-level op grows.
2. The `attention` pattern must be tested as composition over registered primitives such as `matmul`, `softmax_step`, and `layer_norm_step`, not as private `attention_part_a` or `attention_part_b` bodies.
3. The `visual` pattern must be tested by reducing domain language to underlying primitives. If a proposed visual, parsing, matching, graph, security, or backend helper is really bit ops, arithmetic, convolution, scan, hash, layout reflection, or buffer planning, the test should prove reuse of that lower block.
4. Primitive promotion must include evidence of at least two real consumers or stay private to the one owning Tier-3 composition. Tests should name those consumers when promotion is claimed.
5. Composition tests must catch drift: caller-visible behavior, CPU reference bytes, `Program` wire round-trip, and region-chain output must all agree.
6. The matching `SKILL.md` decision-table pattern must be copied as a method, not as a matching-only artifact: each domain should test that routing chooses the smallest correct block, then only moves to a heavier block when constraints require it.
7. The cache and persistence examples must become a seam test pattern: cache keys, wire magic, content hashes, and disk paths must be generated by shared production code and then exercised through a real caller.
8. The region/span dedup examples must become a seam test pattern: owned and in-place compaction must share one identity model and produce byte-equivalent reports.

Required testing ladder:

1. **Primitive/LegoGate tests.** New reusable work must have tests at the owning primitive layer, plus a caller-level test proving the Tier-3 op composes the registered primitive instead of reimplementing it.
2. **Lowering tests.** Each supported `Program` shape must prove `Program -> lower_for_emit -> KernelDescriptor` preserves binding roles, output layout, dispatch dimensions, and deterministic hashes.
3. **Emitter tests.** Each emitter must prove deterministic output, entry-point correctness, binding metadata, missing-entry errors, unsupported-shape errors, and optimizer-stat wiring.
4. **Artifact tests.** `native_module` artifacts must assert schema, target, MSL version, entry point, workgroup size, binding slots, layout hashes, and stable output bytes where the artifact contract requires exact bytes.
5. **Driver unit tests.** Each backend crate must test registration behavior, non-fake platform gating, capability reporting, input validation, output-budget enforcement, and structured error messages.
6. **Backend parity tests.** Every dispatch-capable backend must compare output bytes against `cpu-ref` for representative programs and edge cases.
7. **Resident-resource tests.** Resident allocation, upload, ranged download, batch upload, batch download, dispatch, cleanup, stale-handle rejection, and cache invalidation must be asserted through public APIs.
8. **Cross-crate integration tests.** Real calls must flow from a public `vyre-libs` program through lower, emit, backend dispatch, report, and CPU-reference comparison.
9. **Adversarial tests.** Malformed programs, duplicate bindings, wrong buffer counts, zero-length buffers, extreme dimensions, invalid metadata, corrupted artifacts, and unsupported ops must fail closed with actionable fixes.
10. **Property tests.** Algebraic primitives, byte pack/unpack helpers, region dedup, matching spans, and deterministic artifact builders must run property suites at risk-appropriate scale.
11. **Performance tests.** Cold compile, warm compile, cache hit, cold dispatch, warm dispatch, resident dispatch, upload bandwidth, download bandwidth, ranged transfer, and backend-selection overhead must be measured.
12. **MacBook live tests.** Native Metal correctness, conformance, and benchmarks must run on the Apple GPU path through the scripted SSH gate.
13. **Composition audit tests.** `cargo xtask gate1`, `print-composition`, and duplicate-body checks must stay part of the gate whenever a slice changes `vyre-libs`, `vyre-primitives`, emitters, or backend ABI.
14. **Operator-visible tests.** CLI targets, JSON fields, README claims, docs/targets rows, exit codes, and error text must be covered when a user-visible surface changes.

Concrete test obligations from the actual examples:

1. **Attention-style composition obligation.** If an op looks like a high-level algorithm, tests must assert the expected registered primitive chain instead of accepting private helper decomposition.
2. **Visual-style abstraction obligation.** If a new domain helper is proposed, tests must prove why the work is not already Tier-1 IR, `math`, `text`, `matching`, `hash`, `graph`, `parsing`, `security`, or `logical` substrate.
3. **Public `Program` obligation.** Each public constructor must pass validation, wire round-trip, region-generator identity, backend dispatch, and `cpu-ref` byte comparison.
4. **Decision-table obligation.** Each router or planner must have table-driven tests that prove it picks the smallest sufficient primitive for the workload and records the reason when a heavier primitive is selected.
5. **Shared-helper obligation.** Tests for packing, cache paths, wire magic, content hashes, region/span dedup, dispatch config, output layout, and resident handles must call production helpers, not test-local clones.
6. **Promotion obligation.** A new Tier 2.5 primitive must have owning primitive tests plus at least two caller tests proving real reuse; a one-caller helper stays private and is tested only at the owner boundary.
7. **Backend seam obligation.** Metal tests must use the real `VyreBackend` public API and shared conformance runner where possible, then add driver-local tests only for platform-specific behavior that the shared runner cannot observe.
8. **Failure obligation.** Every validation test needs a negative twin for wrong binding counts, stale handles, unsupported operations, malformed metadata, or wrong abstraction layer, with actionable `Fix:` text when the error crosses a user-visible boundary.
9. **Executable-document obligation.** Any decision table, helper table, route table, or benchmark schema added to docs must have a matching runnable test like `vyre-libs/tests/skill_md_examples.rs` that calls the production symbols and proves the documented rows do not drift.
10. **Testing-skill reference obligation.** The restored `.internals/skills/testing/SKILL.md` must remain the higher-level test doctrine for crate-local `tests/SKILL.md` files, and `.internals/skills/testing/property.md` must remain the shared property-test contract for wire/property surfaces.

LegoGate-specific gates:

1. `./cargo_full run -p xtask --bin xtask -- gate1` must pass for touched primitive and library surfaces.
2. `./cargo_full run -p xtask --bin xtask -- print-composition <op_id>` must show real registered child regions for any op relying on composition to satisfy the budget.
3. `composition_discipline` duplicate-body tests must fail if an op reimplements an existing registered primitive instead of calling it.
4. New Tier 2.5 primitives must have direct primitive tests plus at least two caller-level tests proving real reuse.
5. Single-caller helpers must have owner-level behavior tests, but must not be documented or tested as shared primitives.
6. Any backend helper promoted to shared code must have one test from Metal and one test from another backend, emitter, driver, or harness consumer.
7. Any artifact/reflection/cache/timing schema moved to a shared seam must have a round-trip test plus one real driver or emitter integration test.
8. Any operation dissolved into Tier 1 IR must have a behavior test at the owning composition and no fake primitive registration.

Minimum gates for the Metal runtime slice:

1. `./cargo_full test -p vyre-driver-metal`
2. `./cargo_full test -p vyre-emit-metal`
3. `./cargo_full test -p xtask`
4. `./cargo_full test -p vyre-driver`
5. `scripts/check_no_shader_assets.sh`
6. `cargo xtask gate1`
7. MacBook: `VYRE_BACKEND=metal cargo_full(workspace) test -p vyre-conform-runner`
8. MacBook: `VYRE_BACKENDS=cpu-ref,wgpu,metal cargo_full(workspace) bench -p vyre-bench`

Correctness test families:

1. Positive truth tests for every emitted operation.
2. Negative twins for unsupported shapes and invalid inputs.
3. Cross-backend byte parity tests.
4. Cross-file integration tests from source programs through lower, emit, dispatch, and report.
5. Boundary tests for empty buffers, one-byte buffers, alignment edges, max threadgroup sizes, max buffer lengths, and non-divisible lengths.
6. Adversarial tests for malformed programs, corrupted metadata, wrong binding counts, duplicate bindings, and mismatched output layouts.
7. Property tests for algebraic primitives where exact integer semantics apply.
8. Differential tests against CPU reference and existing GPU backends.
9. Scale corpus tests for large buffers and repeated resident runs.
10. Error-path tests asserting actionable messages and fix text.

Performance test families:

1. Compile-only cost.
2. Pipeline-cache hit cost.
3. Cold dispatch cost.
4. Warm dispatch cost.
5. Resident dispatch cost.
6. Upload bandwidth.
7. Download bandwidth.
8. Ranged download cost.
9. Batch resident operation cost.
10. End-to-end workload latency.
11. Throughput per watt when host APIs expose enough data.
12. Backend selection overhead.
13. Shape prediction accuracy.
14. Command submission overhead.
15. Device synchronization overhead.

Dogfood workloads:

1. Literal matching.
2. Multi-pattern matching.
3. Regex-adjacent automata.
4. Decode pipelines.
5. Hashing pipelines.
6. Parser tokenization.
7. Dataflow propagation.
8. Graph reachability.
9. Fixpoint iteration.
10. Security-rule prefiltering.
11. Large repository text scanning.
12. Mixed CPU/GPU workload scheduling.

## Performance improvement plan

Immediate performance targets:

1. Remove per-dispatch allocation in hot paths.
2. Reuse command queues.
3. Reuse pipeline states.
4. Reuse staging buffers.
5. Reuse bind/resource layouts.
6. Use resident buffers for repeated workloads.
7. Avoid full-buffer readback when only slices are needed.
8. Batch small transfers.
9. Cache lowering and emission artifacts.
10. Cache backend compilation artifacts.
11. Separate compile time from execution time in every benchmark.
12. Separate device time from host wait time where the backend can measure it.
13. Make backend selection capability-aware instead of rank-only.
14. Add typed limits so lowering can choose legal shapes before runtime failure.
15. Make every backend report cache hit/miss counters.

Medium-size architecture upgrades:

1. Shared pipeline-cache key schema across native drivers.
2. Shared backend timing schema across all drivers.
3. Shared resident handle semantics across all drivers.
4. Shared error taxonomy across all drivers.
5. Shared capability schema that covers memory, barriers, atomics, subgroups, native integer widths, timestamp quality, and max dispatch dimensions.
6. Shared benchmarking harness that always compares against `cpu-ref`.
7. Shared output-layout reflection so emitters do not reinvent ABI metadata.
8. Shared adapter fingerprinting primitives.
9. Shared workload corpus.
10. Shared backend conformance manifest.

## Deduplication targets inside Vyre

Deduplication should focus on primitives that are already repeated or are likely to become repeated with Metal.

Targets:

1. Backend device fingerprint creation.
2. Pipeline cache key creation.
3. Program hash normalization.
4. Lowered descriptor hash normalization.
5. Output layout reflection.
6. Binding map reflection.
7. Workgroup shape selection.
8. Buffer alignment logic.
9. Resident handle validation.
10. Backend timing result normalization.
11. Backend error construction.
12. Capability flag definitions.
13. Unsupported-operation diagnostics.
14. Dispatch shape validation.
15. Test corpus loading.
16. Cross-backend parity assertions.
17. Benchmark metric field names.
18. Artifact bundle metadata.
19. Cache invalidation rules.
20. Device-lost or device-reset recovery decisions.

## Seam, boundary, and Lego-block upgrade plan

Vyre should be built from clean Lego blocks: each block has one job, one public contract, one owner, and one test harness. If two blocks need the same primitive, the primitive moves to a shared crate instead of being copied.

Boundary rules:

1. Domain logic must not import CLI, docs, tests, transport, or UI.
2. Drivers must not own semantic lowering.
3. Emitters must not own backend runtime state.
4. Runtime backends must not invent private ABI metadata when shared reflection can express it.
5. Tests must not duplicate production parsers, hashers, layout calculators, or backend selection logic.
6. Benchmarks must use the same dispatch paths that production code uses.
7. Artifact readers and artifact writers must share one schema.
8. Capability fields must be typed data, not prose or string matching.
9. Backend-specific code belongs behind one backend boundary.
10. Cross-backend code belongs in `vyre-driver`, `vyre-lower`, shared emit support, or a dedicated shared crate.

Required seam upgrades:

1. **Backend capability seam** - one shared capability schema for Metal, CUDA, WGPU, SPIR-V, and CPU reference.
2. **Device fingerprint seam** - one shared fingerprint constructor with backend-specific extension fields.
3. **Pipeline-cache seam** - one shared key schema covering program hash, descriptor hash, emitter hash, target options, and device fingerprint.
4. **Artifact seam** - one artifact metadata schema reused by `native_module`, PTX, SPIR-V, and any reloadable runtime bundle.
5. **Reflection seam** - one output-layout and binding-layout reflection model shared by emitters and drivers.
6. **Timing seam** - one timing result schema covering compile, enqueue, device, sync, transfer, allocation, cache hit, and timing quality.
7. **Resident-resource seam** - one handle validation model shared by all drivers.
8. **Error seam** - one backend error taxonomy with actionable `Fix:` text and backend context.
9. **Conformance seam** - one parity runner that can test any backend against `cpu-ref`.
10. **Benchmark seam** - one workload scorecard so backend numbers compare the same fields.
11. **Corpus seam** - one corpus loader for correctness, perf, fuzz, and dogfood workloads.
12. **Rewrite seam** - one optimizer rule registry with legality constraints, tests, and proof hooks.
13. **Lowering seam** - one substrate-neutral `KernelDescriptor` path, with backend-specific choices represented as constraints and options.
14. **Emitter seam** - each emitter consumes the same descriptor and produces source/artifact plus shared reflection.
15. **Scheduler seam** - backend selection uses typed capability and measured cost, not scattered backend priority checks.
16. **Validation seam** - MacBook, desktop, and santhserver validation use the same runner with different device fingerprints.
17. **Dataflow seam** - Weir/Surge/static-analysis consumers pass typed programs and buffers to Vyre instead of embedding custom GPU logic.
18. **Parsing seam** - GPU token/skeleton/parsing helpers expose reusable primitives rather than scanner-specific kernels.
19. **Finding seam** - security/static-analysis findings use one provenance and dedup identity model.
20. **Config seam** - operational knobs live in Tier A config, strategy/rules/corpora live in Tier B data.

Dedup audit targets:

1. Search for duplicate hash construction across drivers, emitters, cache code, tests, and artifact code.
2. Search for duplicate layout and binding metadata construction.
3. Search for duplicate backend capability enums and string constants.
4. Search for duplicate device/profile/fingerprint structs.
5. Search for duplicate timing structs and benchmark JSON fields.
6. Search for duplicate resident-handle validation.
7. Search for duplicate output buffer sizing logic.
8. Search for duplicate workgroup/block/threadgroup shape calculations.
9. Search for duplicate backend error formatting.
10. Search for duplicate conformance fixtures.
11. Search for duplicate corpus loading and generated input helpers.
12. Search for duplicate `Program` canonicalization and hashing.
13. Search for duplicate artifact metadata structs.
14. Search for duplicate capability checks in CLI, tests, docs, and runtime dispatch.
15. Search for duplicate scheduler/rank logic.
16. Search for duplicate cache invalidation rules.
17. Search for duplicate backend-supported-op declarations.
18. Search for duplicate byte-parity assertions.
19. Search for duplicate decode/match/token helper kernels.
20. Search for duplicate report/provenance models.

Every new helper requires a repo search first. If the operation already exists, import it. If the existing location is awkward, move it to a proper shared block instead of copying it.

## Seam test plan

Each seam needs a proving test that uses real adjacent blocks.

1. Source `Program` to lowerer.
2. Lowerer to emitter.
3. Emitter to artifact schema.
4. Artifact schema to runtime backend.
5. Runtime backend to resident resource manager.
6. Resident resource manager to dispatch path.
7. Dispatch path to output-layout reflection.
8. Output-layout reflection to byte-parity assertion.
9. Backend capability schema to scheduler.
10. Scheduler to concrete backend selection.
11. Benchmark harness to real dispatch path.
12. Conformance runner to real backend registry.
13. Corpus loader to fuzz/property/dogfood tests.
14. Error taxonomy to CLI/operator output.
15. Device fingerprint to cache key.
16. Cache key to pipeline cache hit/miss behavior.
17. Rewrite registry to lowering output.
18. Rewrite legality to backend capability.
19. Finding provenance to stable dedup identity.
20. Validation runner to remote MacBook execution.

## Canonical LegoGate doctrine for this plan

This plan follows `docs/lego-block-rule.md`, `docs/primitives-tier.md`, and the `vyre-libs` test skill examples. The rule is not just "dedup when convenient." The rule is composition-first architecture.

Before adding any new sub-op, helper, emitter helper, test fixture, backend adapter, or scanner primitive:

1. Search `vyre-primitives/src/<domain>/`.
2. Search `vyre-libs/src/{math,nn,hash,matching,parsing,text,security,logical}`.
3. Search by name, operation phrase, op id, and sibling region chain.
4. Run the Gate 1 composition lens: if the new work is really a composition of existing blocks, call the blocks.
5. Only invent a new primitive when nothing existing maps and at least two real consumers justify promotion.
6. If there is only one consumer, keep the helper private in the Tier 3 composition until a second consumer exists.
7. If an existing helper is in the wrong layer, move it to the correct Lego block rather than copying it.

The actual examples change how Vyre work should be judged:

1. The `attention` example is not solved by splitting one monolith into private helper chunks. It is solved by composing existing `matmul`, `softmax_step`, second `matmul`, and `layer_norm_step` primitives so the region chain exposes real reuse.
2. The `visual` example is not solved by creating a new visual primitive domain. The correct answer dissolves most proposed visual primitives into existing `math` or Tier 1 IR operations: convolution becomes `math::conv1d`, pixel packing is bit operations, color interpolation is arithmetic, and one-caller SDF stays private.
3. The matching decision table is the model for operator choice: pick the simplest block whose constraints fit, then escalate to richer engines only when the workload requires it.

Applied to Metal:

1. Metal must not create private copies of buffer packing, output reflection, cache keys, device fingerprints, timing records, resident handles, dispatch config math, or conformance fixtures.
2. Metal-specific capability facts belong in a shared capability model with backend extensions, not scattered strings.
3. Metal artifact metadata must reuse the shared artifact/reflection seam rather than inventing a new bundle schema.
4. Metal tests must call the real backend through the registry and shared conformance runner where possible.
5. Metal micro-optimizations must either compose existing lower-level blocks or become reusable primitives with at least two consumers.

Applied to research innovations:

1. A research item that only benefits one kernel remains a private optimization until a second real consumer exists.
2. A research item that creates a new abstraction must name its layer: Tier 1 contract, Tier 2 intrinsic, Tier 2.5 primitive, Tier 3 composition, backend runtime, emitter, scheduler, or test harness.
3. A research item that duplicates an existing primitive is rejected even if it has a better name.
4. A research item that needs a new primitive must include the discovery evidence proving existing blocks do not map.
5. A research item that cannot be tested through a real seam is not ready to land.

## Full testing plan

Testing is a product surface for Vyre. Every backend, emitter, primitive, scheduler, artifact, and optimization must prove behavior against real adjacent blocks.

### Universal public-function contract

Every public function that returns a `Program` must prove:

1. The program validates.
2. The program round-trips through the wire format.
3. The program wraps its body in a region whose generator identifies the owning module or primitive.
4. The program dispatches through every linked dispatch-capable backend that advertises support.
5. Every deterministic backend output matches `cpu-ref` byte-for-byte.
6. Edge cases fail with structured errors, not panics or empty success.
7. Any emitted artifact can be reloaded or rejected with a typed error.
8. Any cache key is deterministic for canonical-equivalent inputs.

### LegoGate tests

Required tests and gates:

1. Gate 1 composition tests for new or modified high-level ops.
2. `lego-audit` for no-reinvention, depth of composition, primitive coverage, and region-chain coverage.
3. Region-chain tests proving Tier 3 compositions call registered Tier 2.5 primitives rather than private clones.
4. Raw IR construction lints for dialect crates that should compose Lego blocks.
5. API-index tests proving public tables point to real symbols.
6. Decision-table tests proving routing picks the simplest valid primitive.
7. Duplicate-helper tests for buffer packing, region/span dedup, dispatch config, cache paths, and fixtures.
8. Promotion tests requiring two real consumers before a private helper becomes a Tier 2.5 primitive.
9. Anti-pattern tests for private split helpers that only hide loop/node budget violations.
10. Regression tests for every helper moved into a shared Lego block.

### Metal backend tests

Metal-specific tests must be layered:

1. Emitter unit tests for MSL source shape, ABI reflection, binding order, output layout, and source maps.
2. Artifact tests for deterministic `native_module` metadata.
3. Compile-failure tests for invalid MSL and unsupported capability combinations.
4. Runtime acquisition tests on macOS and fail-closed unsupported-platform tests elsewhere.
5. Basic dispatch parity tests against `cpu-ref`.
6. Multi-input and multi-output parity tests.
7. Zero-length, one-byte, unaligned, max-ish, and non-divisible length boundary tests.
8. Resident upload, resident dispatch, ranged download, and batch transfer tests.
9. Cache hit/miss tests explaining the exact key component that changed.
10. Timing-quality tests distinguishing host timing from device timing.
11. Device-fingerprint tests proving backend capabilities become typed facts.
12. WGPU-on-Metal versus native Metal differential tests for the same descriptor.
13. MacBook live conformance through the shared runner.
14. Wrong-output replay capsule tests.
15. Counterexample shrinker tests once the operation family is supported by the shrinker.

### Cross-backend conformance tests

Every backend-capable operation family should run through:

1. `cpu-ref` as oracle.
2. WGPU where available.
3. CUDA where available.
4. Metal where available.
5. SPIR-V/artifact validation where relevant.
6. Borrowed dispatch path.
7. Resident dispatch path.
8. Timed dispatch path.
9. Artifact emission path.
10. Artifact reload path where supported.

Failures must identify backend ID, device fingerprint, operation family, program hash, descriptor hash, artifact hash, input corpus ID, and the first differing byte range.

### Property, fuzz, and adversarial tests

Required families:

1. Legal random `Program` generation with byte-parity differential execution.
2. Malformed artifact fuzzing.
3. Invalid binding/layout fuzzing.
4. Resident-handle misuse fuzzing.
5. Buffer size, alignment, overflow, and underflow adversarial cases.
6. Operation-specific algebraic property tests where exact integer semantics apply.
7. Floating-point tests separated by exact, approximate, and intentionally unsupported semantics.
8. Corpus mutation tests for parsing, matching, graph, and decode workloads.
9. Backend capability fuzzing to prove unsupported paths fail before dispatch.
10. Error-message tests requiring context plus `Fix:` guidance.

### Performance tests

Performance tests must measure the real production path, not microbench-only shortcuts.

Metrics:

1. Compile time.
2. Lowering time.
3. Emission time.
4. Backend compile/JIT time.
5. Pipeline-cache hit cost.
6. Enqueue cost.
7. Device execution time where trustworthy.
8. Host wait/sync time.
9. Upload bytes.
10. Download bytes.
11. Avoided readback bytes.
12. Allocation count.
13. Copy count.
14. Cache hit/miss reason.
15. Resident handle reuse count.
16. Workgroup/threadgroup shape.
17. Occupancy or counter data where available.
18. Thermal/device-state annotation for MacBook runs.
19. Backend selection decision and alternatives rejected.
20. End-to-end workload latency.

Perf gates should compare cold dispatch, warm dispatch, resident dispatch, and artifact-cache dispatch separately. Native Metal is not considered strong because it compiles once; it is strong when warm resident workloads beat WGPU-on-Metal or explain why they cannot.

### Research innovation tests

Each selected research item must declare:

1. Novelty label.
2. Owning Lego block.
3. Existing primitives searched.
4. New seam, if any.
5. Truth test.
6. Negative test.
7. Adversarial test.
8. Cross-backend differential test.
9. Benchmark.
10. Corpus or workload where it matters.
11. Failure mode.
12. Operator-visible output.

If these cannot be named, the item stays as research inventory and does not reshape architecture.

## Innovation novelty labels

Innovation items must be labeled honestly before implementation:

1. **Absorbed research** - established technique that Vyre should implement cleanly.
2. **Novel integration** - known ingredients combined in a Vyre-specific way.
3. **Frontier bet** - a research-level idea where success requires real experiments, measurements, and failure analysis.
4. **Moonshot** - a high-risk capability that needs a reduced prototype and hard evidence before it becomes core architecture.

No item should be described as novel only because it has a new name. The test is whether Vyre gains a measurable capability, correctness property, performance win, or system boundary that did not exist before.

## Research anchors

This catalog is grounded in specific systems and papers, then translated into Vyre-native work. The point is not to copy any one system. The point is to make Vyre combine compiler truth, GPU-resident execution, code-analysis workloads, automata, parsing, graph/dataflow, and backend specialization in one stack.

Primary anchors:

1. [Apple Metal argument buffers](https://developer.apple.com/documentation/metal/buffers/improving_cpu_performance_by_using_argument_buffers) and [argument buffers with resource heaps](https://developer.apple.com/documentation/metal/using-argument-buffers-with-resource-heaps?changes=la_9): resource binding and heap residency should become first-class runtime concepts in `vyre-driver-metal`.
2. [Apple Metal counter sampling](https://developer.apple.com/documentation/metal/gpu_counters_and_counter_sample_buffers/sampling_gpu_data_into_counter_sample_buffers?changes=_2.) and [Metal feature set tables](https://developer.apple.com/metal/capabilities/): backend capability fingerprints and timing quality must come from device facts, not rank guesses.
3. [Gunrock GPU graph analytics](https://arxiv.org/abs/1701.01170): frontier-centric GPU graph execution maps directly to code property graphs, reachability, taint, and fixpoint work.
4. [GraphBLAST](https://arxiv.org/abs/1908.01407), [GraphBLAS](https://graphblas.org/), and [GBTL-CUDA](https://www.sei.cmu.edu/library/gbtl-cuda-graph-algorithms-and-primitives-for-gpus/): sparse linear algebra and semirings are a serious path for GPU-native graph/dataflow primitives.
5. [PFAC and GPU Aho-Corasick research](https://www.mdpi.com/2079-9292/8/3/270), [iNFAnt NFA pattern matching](https://ccr.sigcomm.org/online/files/p21-2v40n5d2-cascaranoA.pdf), and [GPU regex automata representation work](https://researchwith.stevens.edu/en/publications/exploring-different-automata-representations-for-efficient-regula/): Vyre matching should choose automata representations by measured density, memory pressure, and branch behavior.
6. [Hyperscan](https://www.usenix.org/conference/nsdi19/presentation/wang-xiang), [Hyperscan internals](https://www.intel.com/content/www/us/en/collections/libraries/hyperscan/regular-expression-match.html), and [Hyperscan logical combinations](https://www.intel.com/content/www/us/en/developer/articles/technical/logical-combinations-of-regular-expressions-in-hyperscan.html): large pattern sets require decomposition, logical composition, SIMD/bit-parallel engines, and benchmarkable corpora.
7. [GPU Gems scan/prefix-sum](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda) and stream-compaction research: scan, compact, partition, and histogram are core Vyre primitives, not utility kernels.
8. [ParPaRaw GPU raw-data parsing](https://arxiv.org/abs/1905.13415) and [GPU CKY parsing](https://aclanthology.org/W11-2921/): parsing can use GPU skeletonization, token classification, and bulk grammar checks even when full AST construction remains CPU-owned.
9. [GPU static data-flow analysis for Android vetting](https://www.anl.gov/argonne-scientific-publications/pub/162586), [GPU points-to analysis](https://www.jstage.jst.go.jp/article/jssst/29/3/29_3_70/_article/-char/en), and [GPU-accelerated fixpoint algorithms](https://cris.fau.de/publications/210729454/): naive worklists underutilize GPUs, so Vyre needs data-parallel fact propagation, frontier compaction, and sparse relation kernels.
10. [Souffle Datalog for static analysis](https://souffle-lang.github.io/cav-paper) and [multi-node multi-GPU Datalog](https://www-new.evl.uic.edu/news/2025/2025-06-08-2930/): large static-analysis relations require specialized joins, dedup, materialization control, and relation-aware scheduling.
11. [rNdN fast GPU query compilation](https://dblp.uni-trier.de/rec/journals/taco/KrolikVH23.html), [ReSQL low-latency query compilation](https://dbis.cs.tu-dortmund.de/en/publications/2022/low-latency-query-compilation/), [Pyper GPU query efficiency](https://www.vldb.org/pvldb/vol14/p202-paul.pdf), and [TQP++](https://www.microsoft.com/en-us/research/publication/tqp-bridging-ml-compilers-and-analytical-query-processing-on-gpus/): short GPU workloads die from compilation overhead and materialization unless the compiler/runtime cooperate.
12. [Equality saturation](https://www.cs.cornell.edu/~ross/publications/eqsat/), [egg](https://arxiv.org/abs/2004.03082), [STOKE](https://www.microsoft.com/en-us/research/publication/stochastic-superoptimization/), [Alive2](https://github.com/AliveToolkit/alive2), and [Csmith](https://web.stanford.edu/class/cs343/resources/finding-bugs-compilers.pdf): Vyre needs optimizer search, translation validation, randomized program generation, shrinking, and wrong-output hunting as core infrastructure.

## Research-level innovation catalog

Each item below is named intentionally. Every item must become one of: a crate/API, a compiler pass, a backend feature, a conformance gate, a benchmark, a corpus entry, a capability field, or an operator-visible report.

1. **Metal ABI Truth Certificates** - Emit a per-kernel certificate covering binding layout, output layout, workgroup shape, MSL hash, device fingerprint, and byte-parity status against `cpu-ref`.
2. **Unified-Memory Slice Return** - On Apple GPUs, plan outputs as resident slices and read back only requested byte ranges instead of whole buffers.
3. **Heap-Resident Vyre Arena** - Use Metal heaps as a backend-owned arena for long-lived buffers so repeated workloads avoid per-dispatch allocation churn.
4. **Argument-Buffer Megabind ABI** - Collapse large binding sets into Metal argument buffers with stable Vyre reflection metadata and Tier 1/Tier 2 capability checks.
5. **Function-Constant Specialization Bank** - Compile one MSL template with Metal function constants for lengths, flags, and strategy choices instead of regenerating full source for every shape.
6. **Metal Counter Quality Model** - Classify Metal timing and counter data by sampling support, resolution, boundary type, and measurement distortion.
7. **Apple Thermal Benchmark Normalizer** - Record thermal state, power mode, and repeated-run drift so MacBook benchmarks do not lie about backend performance.
8. **MSL Source Operation Map** - Attach every generated MSL line to the originating Vyre operation for debug, minimization, and wrong-output reports.
9. **Native-vs-WGPU Metal Duel** - Run the same lowered descriptor through native Metal and WGPU-on-Metal, then report compile, enqueue, execution, sync, transfer, and parity differences.
10. **Metal Feature-Family Planner** - Make lowering choose legal barriers, atomics, argument buffers, indirect dispatch, and SIMD-group paths from Metal feature-family facts.
11. **Cross-Backend Counterexample Shrinker** - When `metal`, `cuda`, `wgpu`, and `cpu-ref` disagree, shrink the `Program`, inputs, and layout to the smallest reproducing case.
12. **Byte-Truth Backend Passport** - Store a backend/device passport listing exactly which operation families passed parity on that physical device.
13. **Bounded Vyre Translation Validator** - Use an Alive2-inspired bounded checker for selected Vyre rewrites so optimizer wins carry proof obligations.
14. **Emitter Differential Comparator** - Compare Naga-generated MSL, direct MSL, PTX, SPIR-V, and CPU reference semantics at the descriptor level before runtime dispatch.
15. **Semantic Program Hashing** - Hash canonical operation semantics rather than source order accidents so equivalent kernels share cache entries.
16. **Legal Program Fuzz Forge** - Generate random but valid Vyre programs with defined semantics, then run differential backend testing at scale.
17. **Malformed Artifact Fuzz Forge** - Generate corrupted `native_module`, PTX, SPIR-V, and metadata bundles to prove fail-closed artifact loading.
18. **Resident Handle Model Checker** - Explore stale handles, cross-backend handles, overlapping ranges, double frees, and range abuse against resident APIs.
19. **Deterministic Replay Capsule** - Capture program, inputs, backend fingerprint, compiler options, emitted source, artifact hashes, and outputs for every wrong-output bug.
20. **Optimization Bisect Ledger** - Record every applied rewrite and pass decision so wrong-output minimization can binary-search optimizer responsibility.
21. **Vyre E-Graph Optimizer** - Add an equality-saturation optimizer over `KernelDescriptor` for algebraic, layout, and memory-access rewrites.
22. **Costed Equality Extraction Solver** - Extract from the e-graph using backend-specific costs for register pressure, memory coalescing, barriers, and readback volume.
23. **Multi-Backend Pareto Extractor** - Choose rewrites that are Pareto-good across Metal, CUDA, WGPU, and CPU instead of optimizing for one backend blindly.
24. **Rewrite Certificate Bundles** - Ship each rewrite with positive, negative, adversarial, and differential tests plus a bounded semantic check when feasible.
25. **Learned Rewrite Admission** - Admit new rewrite choices only when benchmark history shows a stable win for a workload/device class.
26. **Relational E-Matching for Dataflow** - Apply equality saturation to relation algebra and semiring expressions used by graph/dataflow workloads.
27. **Descriptor Equivalence Cache** - Cache proof that two lowered descriptors are semantically identical, allowing compile-cache reuse across syntactic variants.
28. **Constraint-Tagged Rewrite Rules** - Attach backend legality constraints directly to rewrite rules so illegal Metal/CUDA choices never reach codegen.
29. **Rewrite Heatmap Dogfood Report** - Report which rewrites actually fired on real corpora and which are dead weight.
30. **Superoptimized Microkernel Library** - Use STOKE-style search and exhaustive differential tests for tiny integer, scan, bitmap, and offset kernels.
31. **Hybrid PFAC-NFA Matcher** - Choose per-pattern cluster between failureless Aho-Corasick, bit-parallel NFA, compressed DFA, and CPU fallback.
32. **Automata Density Router** - Route regex/pattern groups by measured NFA active-state density and DFA memory pressure instead of pattern-count heuristics.
33. **Hyperscan-Style Literal Rose Extractor** - Decompose complex rules into literal roses, accelerable prefilters, and confirmation fragments.
34. **GPU Candidate Window Emitter** - Return compact candidate windows from GPU matching, then confirm only suspicious windows on CPU or a second kernel.
35. **Logical Pattern Combiner Kernel** - Evaluate AND/OR/NOT combinations of pattern hits on device so large rule packs avoid CPU-side bitmap stitching.
36. **Streaming Boundary State Tiles** - Preserve automata state across chunk boundaries for large corpus scanning without rescanning overlap blindly.
37. **Bit-Parallel NFA Word Packs** - Pack NFA state into machine words/SIMD groups and use GPU bit operations for epsilon-closure-style transitions where legal.
38. **Automata Compression Autotuner** - Select table compression, transition encoding, and state layout using measured bandwidth, cache behavior, and divergence.
39. **Two-Phase Long-Match Balancer** - Detect long-running match lanes, compact them, and process them in a second phase to reduce warp/SIMD-group imbalance.
40. **Multi-Decode Automata Fusion** - Fuse raw, URL-decoded, HTML-decoded, JSON-unescaped, and base64-decoded candidate streams into one provenance-preserving scan plan.
41. **GPU Delimiter Skeletonizer** - Build delimiter, quote, escape, bracket, and newline skeletons on GPU before parser-specific CPU structure building.
42. **Tree-Sitter Token Prefilter** - Use GPU token classification to reduce CPU tree-sitter work to syntactically interesting spans.
43. **Parallel Bracket Balance Spine** - Compute bracket/brace/paren balance and error candidates with scan primitives across huge files.
44. **Syntax Island Indexer** - Identify string literals, comments, template blocks, SQL fragments, JS-in-HTML, and shell-in-YAML islands as GPU-generated interval sets.
45. **Incremental Chunk Identity Cache** - Use content-defined chunks so unchanged source ranges preserve token, match, and dataflow artifacts across scans.
46. **Decode-Provenance Offset Map** - Track decoded bytes back to original file offsets through GPU decode pipelines for accurate findings.
47. **Parser Ambiguity Frontier Queue** - Store ambiguous parse frontier candidates as compact GPU queues for languages with embedded or partial syntax.
48. **GPU Token Predicate Bank** - Evaluate thousands of lexical predicates over token streams as bitsets before semantic passes run.
49. **Cross-Language Lexeme Schema** - Normalize identifiers, calls, literals, member accesses, imports, and comments into one GPU-friendly token table.
50. **Structural Hash AST Overlay** - Overlay CPU AST nodes with GPU-computed structural hashes for fast dedup, incremental invalidation, and repeated scans.
51. **Frontier Dataflow Engine** - Model static-analysis propagation as Gunrock-style frontiers rather than scalar worklists.
52. **Datalog Join Accelerator** - Implement GPU joins, semijoins, antijoins, projection, dedup, and materialization control for static-analysis relations.
53. **Semiring Taint Propagator** - Express taint, reachability, and confidence accumulation as GraphBLAS-style semiring operations.
54. **Sparse CPG Matrix Backend** - Store code property graph slices as sparse matrices for GPU BFS, reachability, dominator approximations, and slice queries.
55. **Dynamic Frontier Worklist** - Use compacted frontiers and active-set bitmaps so fixpoint iterations do not underutilize the GPU.
56. **GPU SCC Condenser** - Condense strongly connected components on device for call graphs, import graphs, and dependency graphs before higher-level analysis.
57. **Field-Sensitive Fact Bitsets** - Encode object-field facts as compressed bitsets to reduce relation explosion in points-to and taint analyses.
58. **Interprocedural Summary Cache** - Cache procedure summaries by structural hash and invalidate only affected callgraph regions.
59. **Flow-Sensitivity Delta Engine** - Propagate only fact deltas between iterations, with GPU-side duplicate suppression.
60. **Pointer-Fact Candidate Filter** - Use GPU prefiltering to discard impossible alias/source/sink pairs before expensive precise analysis.
61. **Truth-Weighted Backend Scheduler** - Choose backends by capability, recent correctness passport, workload shape, and measured speed, not fixed precedence alone.
62. **Cache Admission Oracle** - Admit compiled kernels to cache only when compile cost, expected reuse, and memory pressure justify it.
63. **Compile-Latency Governor** - Route short workloads through interpreter, cached template, or precompiled path when full compilation would dominate runtime.
64. **Warm-Resident Classifier** - Detect workloads that will repeat and move them to resident-buffer plans automatically.
65. **Fusion Profit Model** - Fuse kernels only when it reduces materialization, dispatch overhead, or memory traffic without causing register/divergence blowups.
66. **Batch Fusion DAG** - Represent a batch of Vyre kernels as a DAG that can be fused, split, or scheduled across devices.
67. **Readback Avoidance Planner** - Keep intermediate facts, match bitmaps, and candidate windows on GPU until an operator-visible output requires CPU bytes.
68. **Checksum-First Validator** - For huge outputs, compare GPU-generated checksums before expensive full readback during conformance runs.
69. **Host-GPU Overlap Pipeline** - Overlap input loading, upload, dispatch, download, and CPU confirmation using explicit runtime stages.
70. **Multi-Device Consensus Runner** - Run high-value kernels on two different backends/devices and require agreement before trusting novel optimizer paths.
71. **Workgroup Genetic Tuner** - Search workgroup sizes and shapes per kernel family/device using benchmark-guided mutations.
72. **Occupancy-Register Budgeter** - Reject rewrites that increase registers enough to reduce occupancy below the measured profit threshold.
73. **Threadgroup Memory Tradeoff Solver** - Choose when to use Metal threadgroup memory/CUDA shared memory based on reuse distance and bank-conflict risk.
74. **Vector-Width Bandit** - Learn per-device vector width choices for literal matching, decode, hashing, and bitmap operations.
75. **Backend Shape Pair Learner** - Learn shape mappings where Metal and CUDA need different optimal threadgroup/block choices for the same descriptor.
76. **Allocator Churn Profiler** - Report allocations per dispatch path and fail perf gates when hot paths allocate unexpectedly.
77. **Copy-Elimination Proof Pass** - Trace buffer ownership and prove which host/device copies are unnecessary, then assert the count in benchmarks.
78. **Divergence Heat Mapper** - Instrument branch-heavy kernels to report where lanes diverge across automata, parsing, and graph workloads.
79. **Cache Miss Explainer** - Explain why a pipeline/resident/artifact cache missed, including which part of the key changed.
80. **Performance Regression Corpus Bisection** - Bisect corpus, kernel, pass, and backend changes to find the smallest workload causing a perf regression.
81. **Rule Selectivity Learner** - Learn which static/security rules are highly selective and schedule them as early GPU prefilters.
82. **Finding Candidate GPU Sieve** - Emit only candidate finding tuples that survive cheap GPU checks before CPU semantic validation.
83. **Sink-Source Pair Generator** - Generate source/sink candidate pairs on GPU from token, AST overlay, and dataflow bitsets.
84. **Path-Feasibility Bitmap Explorer** - Use GPU bitmaps to cheaply approximate path feasibility before deeper symbolic or LLM reasoning.
85. **Vulnerability Primitive Lattice** - Represent SSRF, authz, injection, deserialization, path traversal, and secret-flow primitives as composable dataflow operators.
86. **Attack Surface Corpus Index** - Build a resident GPU index of routes, handlers, parameters, schemas, and auth checks for repeated analysis.
87. **On-Device Finding Dedup** - Deduplicate finding candidates by stable hashes on GPU before reporting or CPU confirmation.
88. **Evidence Provenance Ledger** - Attach backend, kernel, corpus chunk, decode chain, rule, and dataflow path provenance to every generated candidate.
89. **Rule Collision Simulator** - Detect when multiple rules repeatedly fire on the same primitive and consolidate them into one stronger operator.
90. **Exploitability Prior Ranker** - Rank candidate findings using graph distance to trust boundaries, auth context, input controllability, and sink impact.
91. **Device Farm Fingerprinter** - Treat desktop, santhserver, and MacBook as typed devices with reproducible capability fingerprints.
92. **MacBook Native Metal Oracle** - Make the MacBook the canonical native Metal validation host with scripted conformance and perf capture.
93. **Santhserver CUDA Cross-Oracle** - Use santhserver CUDA as a second native backend oracle for cross-architecture disagreement hunting.
94. **Cross-Architecture Repro Capsule** - Store enough data to replay a bug on Apple Metal, CUDA, WGPU, and CPU reference.
95. **Remote Benchmark Thermostat** - Normalize remote benchmark runs by device load, thermal state, driver/runtime version, and warmup profile.
96. **Multi-Host Corpus Sharder** - Shard massive corpora across devices while preserving deterministic result merge and dedup.
97. **Confidence Consensus Engine** - Compute result confidence from backend agreement, conformance passport, and operation coverage.
98. **Backend Quarantine Policy** - Automatically stop selecting a backend for a capability family after repeated wrong-output or device-failure evidence.
99. **Fleet Capability Registry** - Maintain a TOML-backed registry of devices, capabilities, counters, memory limits, and validated backends.
100. **Workload Placement Optimizer** - Place parsing, automata, graph, dataflow, and confirmation stages on the best device based on measured costs.
101. **Flounder-Style Tiny Vyre IR** - Add a low-latency, close-to-kernel IR for short workloads where heavyweight lowering/emission costs more than execution.
102. **Map-Reduce Fusion Schema** - Borrow GPU query-processing fusion ideas to express scan/filter/project/reduce workloads without intermediate materialization.
103. **Multi-Gated Execution Graph** - Switch algorithms at runtime based on observed selectivity, match density, frontier size, and transfer pressure.
104. **Runtime Characteristic Switcher** - Collect cheap runtime statistics and choose between dense, sparse, branchy, and resident execution variants.
105. **Interpreter-Compiler Hybrid Path** - Execute tiny/rare programs through a low-overhead path and promote hot programs to compiled native kernels.
106. **Kernel Template Instantiator** - Prebuild templates for common primitive families and specialize by constants, avoiding full codegen on hot short jobs.
107. **Compile Budget Optimizer** - Give each workload a compile-time budget and choose optimization depth by expected execution savings.
108. **Operator Pipeline Fusion** - Fuse decode, match, filter, project, and compact into one kernel when data movement dominates.
109. **Segment-Shuffle Divergence Splitter** - Split divergent query/static-analysis operators into segment and shuffle phases to improve GPU utilization.
110. **Columnar Corpus Operator Set** - Store token, finding, relation, and route corpora in columnar GPU-friendly buffers for analytical-style kernels.
111. **Lowering Decision Journal** - Persist every lowering decision with input facts, rejected alternatives, and measured result so decisions become auditable.
112. **Proof-Carrying Benchmark Result** - Pair every benchmark number with exact inputs, backend passport, binary/artifact hashes, and warmup policy.
113. **Counterfactual Compiler Replay** - Replay a workload with selected passes disabled to quantify each pass contribution to time and correctness risk.
114. **Workload Ecology Dashboard** - Show which real workloads dominate runtime, compile time, memory, transfers, cache misses, and wrong-output risk.
115. **Capability-Limit Synthesizer** - Generate tiny kernels that empirically discover backend/device limits not exposed cleanly by APIs.
116. **Adversarial Program Lab** - Maintain a corpus of pathological kernels for OOM, divergence, atomics, barriers, alignment, oversized buffers, and invalid metadata.
117. **Bug-Report-Guided Mutator** - Convert historical Vyre/backend bugs into fuzzer biases, inspired by compiler bug-report-guided generation.
118. **Rule-to-Kernel Trace Map** - Link high-level security/static-analysis rules to the exact generated kernels and data buffers that evaluated them.
119. **Backend Contract Litmus Suite** - Define small litmus tests for every backend contract: resident safety, errors, timing, cache, artifact loading, and byte truth.
120. **Vyre Research Scorecard** - Score each change by novelty, correctness evidence, measured speed, backend coverage, corpus impact, and security-analysis value.

## Metal-specific innovation targets

Apple GPUs are not just another CUDA target. Native Metal should exploit Apple-specific properties where they are real and measured.

1. Unified-memory-aware transfer planning.
2. Metal heap-backed buffer pooling.
3. Argument buffers for large resource sets.
4. Function constants for cheap specialization.
5. MSL source mapping to Vyre operations.
6. SIMD-group abstraction where feature support is present.
7. Counter/timestamp quality detection.
8. Thermal-state annotation for repeatable laptop benchmarks.
9. Metal binary/archive cache if the API surface and reproducibility requirements allow it.
10. Native comparison against WGPU-on-Metal for every representative workload.

## Backend conformance matrix

Every backend should report capability and truth status for these categories:

1. Scalar integer operations.
2. Scalar floating operations where exact semantics are promised.
3. Buffer reads.
4. Buffer writes.
5. Multi-output kernels.
6. Bounds checks.
7. Barriers.
8. Atomics.
9. Subgroup/SIMD-group operations.
10. Resident buffers.
11. Ranged downloads.
12. Batch transfers.
13. Timed dispatch.
14. Cancellation or timeout behavior.
15. Device-loss behavior.
16. Artifact emission.
17. Artifact reload.
18. Error message quality.
19. Cache behavior.
20. Cross-backend parity.

## Workload scorecard

Each completed optimization should be scored against real Vyre workloads:

1. Does it reduce compile time?
2. Does it reduce first dispatch latency?
3. Does it reduce warm dispatch latency?
4. Does it reduce allocation count?
5. Does it reduce CPU copies?
6. Does it reduce readback bytes?
7. Does it increase cache hit rate?
8. Does it increase backend coverage?
9. Does it increase correctness confidence?
10. Does it make a security or code-analysis workload more capable?

## Implementation order

1. Add `vyre-emit-metal`.
2. Add `vyre-driver-metal`.
3. Wire workspace features and target `cfg`.
4. Register backend ID `metal`.
5. Implement basic MSL emission through the shared lowering path.
6. Implement basic Metal dispatch.
7. Add byte-parity tests against `cpu-ref`.
8. Add MacBook live validation command.
9. Wire `native_module` artifact emission.
10. Add resident buffers.
11. Add pipeline cache.
12. Add benchmark comparisons for `cpu-ref`, `wgpu`, and `metal`.
13. Add capability matrix reporting.
14. Add backend truth certificates.
15. Add workload scorecard gates.
16. Start consuming the innovation backlog by choosing items that improve real workloads and can be proven by tests or benchmarks.
17. Run the dedup audit before adding new Metal-specific helpers.
18. Move duplicated helpers into shared Lego blocks.
19. Add seam tests for every moved shared block.
20. Label each selected innovation by novelty tier.
21. Reject innovation work that cannot name its proving test, benchmark, or operator-visible behavior.
22. Keep backend-specific code inside the backend crate unless the shared contract has been upgraded first.
23. Keep emitter-specific code inside the emitter crate unless reflection/artifact schema needs a shared upgrade.
24. Keep security/static-analysis consumers on typed Vyre APIs instead of scanner-specific GPU shortcuts.
25. Treat every duplicate primitive found during Metal work as part of the Metal work, not unrelated cleanup.

## Acceptance gates

The Metal driver is real when these gates pass:

1. `metal` appears in the backend registry on macOS.
2. `metal` does not appear as a fake successful backend on unsupported platforms.
3. At least one real `Program` dispatches through native Metal.
4. Outputs match `cpu-ref` byte-for-byte.
5. Multi-buffer dispatch works.
6. Error paths return actionable messages.
7. `native_module` emits an artifact.
8. Resident upload, dispatch, and download work.
9. Warm resident dispatch is measured separately from cold dispatch.
10. Native Metal is benchmarked against WGPU-on-Metal.
11. The MacBook validation gate is scriptable through SSH.
12. The docs, CLI target naming, backend registry, tests, and benchmark labels agree.
13. Metal uses shared capability, timing, artifact, reflection, cache-key, error, and resident-resource seams.
14. No duplicated Metal-only helper exists where a shared Vyre primitive should own the behavior.
15. Every newly shared seam has a real adjacent-module test.
16. Every selected innovation has a novelty label and a proving artifact.
17. Backend selection uses typed facts and measured costs rather than scattered priority strings.
18. Performance gates report allocation count, copy count, readback bytes, and cache hit/miss reason.
19. Wrong-output failures produce replay capsules and minimized counterexamples where the shrinker supports the operation family.
20. The implementation leaves fewer duplicate primitives than it found in the touched area.

## First concrete build slice

The first build slice should touch only the minimum needed for a real native path:

1. `vyre-emit-metal` crate.
2. Workspace manifest.
3. `xtask` target emission wiring.
4. `vyre-driver-metal` crate.
5. A small conformance test set comparing `metal` to `cpu-ref`.
6. `scripts/check_metal_macbook.sh` for MacBook driver, conformance, benchmark, and complete Metal gates.
7. Shared capability/fingerprint/cache-key/timing/reflection seams needed by the driver.
8. Dedup moves required to avoid copying existing helper logic into the Metal path.

The result must not be a placeholder. It must execute at least one real kernel through Metal and prove byte parity.

## Implementation evidence - 2026-06-07

Metal driver work completed in this batch:

1. `vyre-emit-metal` artifact schema is now `3` and records dense Metal buffer indices separately from descriptor slots.
2. Workgroup/shared memory bindings no longer consume host Metal buffer slots.
3. Threadgroup memory metadata is emitted and the Metal runtime calls `set_threadgroup_memory_length` before dispatch.
4. Naga CAS result types are registered through the predeclared-type table so Metal MSL includes `naga_atomic_compare_exchange_weak_explicit`.
5. Metal runtime allocates backend-owned `__vyre_naga_trap_sidecar` storage when trap lowering inserts an artifact-only binding.
6. Metal runtime computes every fallible sidecar, threadgroup-memory, and grid value before creating a command encoder, preventing uncaught Objective-C aborts on ordinary backend errors.
7. Metal dispatch grid inference uses the shared `dispatch_element_count_for_program` primitive instead of a Metal-local output-trim heuristic.
8. Native Metal now advertises subgroup support and proves it with an Apple `SubgroupSize` dispatch test.
9. `vyre-libs::parsing::c11_build_cfg_and_gotos` is split into label-insert and goto-lookup phases with a uniform barrier so GPU execution does not race label writers against goto readers.
10. Conformance selector tests now force-link Metal inventory and honor `VYRE_BACKEND=metal` before backend acquisition.
11. `parity_matrix` filters registered backends by `VYRE_BACKEND` before probing factories, preventing CUDA probing on the Mac Metal gate.
12. `vyre-driver-metal` has a release coverage floor in `scripts/check_test_coverage_per_crate.sh`.

Tests and validation completed:

1. Local: `./cargo_full test -p vyre-emit-naga -p vyre-emit-metal`.
2. Local: `./cargo_full test -p xtask native_module_target_emits_metal_artifact_json`.
3. Local: `./cargo_full test -p vyre-driver-metal`.
4. Local: `./cargo_full test -p vyre-conform-runner --features gpu --test lens_parity --test ulp_audit --no-run`.
5. Local: `./cargo_full test -p vyre-conform-runner --features gpu --test parity_matrix --no-run`.
6. Local: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts concrete_driver_coverage_floors_are_nonzero_release_gates`.
7. Mac: `./cargo_full test -p vyre-driver-metal` passed 6 Apple runtime tests.
8. Mac: `prove --backend metal --ops vyre-primitives::reduce::workgroup_sum_u32` passed.
9. Mac: `prove --backend metal --ops vyre-libs::nn::cross_entropy` passed.
10. Mac: `prove --backend metal --ops vyre-primitives::decode::inflate_stored` passed.
11. Mac: `prove --backend metal --ops vyre-primitives::bitset::select1_query` passed.
12. Mac: `prove --backend metal --ops vyre-libs::parsing::c11_build_cfg_and_gotos` passed.
13. Mac: `VYRE_BACKEND=metal ./cargo_full test -p vyre-conform-runner --features gpu --test cert_artifact` passed 11 tests.
14. Mac: `VYRE_BACKEND=metal ./cargo_full test -p vyre-conform-runner --features gpu --test lens_parity` passed 4 tests.
15. Mac: `VYRE_BACKEND=metal ./cargo_full test -p vyre-conform-runner --features gpu --test parity_matrix` passed 5 tests, including the catalog-scale parity case.
16. Mac: `VYRE_BACKEND=metal ./cargo_full test -p vyre-conform-runner --features gpu --test ulp_audit` passed 4 tests.
17. Mac: `VYRE_BACKEND=metal ./cargo_full test -p vyre-conform-runner --features gpu` passed the full package, including cert, gap cert, lens, parity matrix, release gates, ULP audit, and doctests.

## Implementation evidence - 2026-06-07 resident execution slice

Implemented slice:

1. `vyre-driver-metal::MetalBackend` now owns a deterministic resident resource table guarded by a mutex and monotonic nonzero resident handles.
2. Native Metal implements `allocate_resident`, `upload_resident`, `upload_resident_many`, `upload_resident_at`, `upload_resident_at_many`, `download_resident_into`, `download_resident_range_into`, `download_resident_ranges_into`, `free_resident`, and `dispatch_resident_timed` through the public `VyreBackend` contract.
3. Metal resident dispatch resolves resources in binding order, supports mixed borrowed/resident input resources, rejects stale handles, rejects wrong resource counts, rejects undersized output handles, and preserves resident output buffers for explicit readback after dispatch.
4. Metal resident dispatch reuses `BindingPlan`, `output_binding_layouts`, `dispatch_element_count_for_program`, `infer_dispatch_grid_for_count`, `vyre-lower`, `vyre-emit-metal`, Metal artifact binding metadata, `_buffer_sizes` sidecar planning, internal trap sidecar allocation, and shared output-budget enforcement.
5. Borrowed and resident Metal dispatch now share one command-encoding path for pipeline binding, threadgroup memory binding, `_buffer_sizes` binding, dispatch sizing, command-buffer status checking, output collection, and timing capture.
6. Resident full upload accepts bounded payloads and zero-pads unwritten logical allocation bytes; ranged and batch transfers validate all ranges against logical resident allocation size before touching Metal memory.
7. Resident downloads return logical allocation bytes even when Metal needs a nonzero physical allocation for zero-length safety.
8. Apple-only tests prove full upload, zero padding, ranged upload, ranged download, batch upload, ranged batch upload, ranged batch download, stale-handle rejection after free, resident dispatch, returned output bytes, persistent resident output contents, and host timing fields.

Validation evidence:

1. Local Linux: `./cargo_full test -p vyre-driver-metal` passed: 2 non-Apple tests, 0 failures.
2. MacBook Apple GPU: `ssh -o BatchMode=yes -o ConnectTimeout=8 tt-macbook 'cd /Users/thiruthangarathinam/Santh/libs/performance/matching/vyre && CARGO_TARGET_DIR="$HOME/cargo-target-metal-fresh" ./cargo_full test -p vyre-driver-metal'` passed: 8 Apple tests, 0 failures.
3. MacBook Apple GPU: `ssh -o BatchMode=yes -o ConnectTimeout=8 tt-macbook 'cd /Users/thiruthangarathinam/Santh/libs/performance/matching/vyre && CARGO_TARGET_DIR="$HOME/cargo-target-metal-fresh" VYRE_BACKEND=metal ./cargo_full test -p vyre-conform-runner --features gpu'` passed all `vyre-conform-runner` unit, cert, dispatch-grid, gap-cert, lens, parity-matrix, release-gate, ULP-audit, and doctest sections with 0 failures.
4. The full Mac conformance run specifically revalidated the shared Metal dispatch path after resident command encoding was deduplicated with borrowed dispatch.

## Implementation evidence - 2026-06-07 Metal pipeline cache slice

Implemented slice:

1. `vyre-driver-metal::MetalBackend` now keeps an in-memory compiled Metal pipeline cache for MSL artifacts and `MTLComputePipelineState` objects.
2. Metal pipeline cache keys reuse shared backend-neutral cache primitives: `try_normalized_program_cache_digest`, `dispatch_policy_cache_digest`, `hex_encode`, and `PipelineDeviceFingerprint`.
3. The cache key includes normalized Program identity, dispatch policy, Metal artifact schema, MSL version, Metal driver crate version, and Metal device name, so policy or compilation-contract changes miss instead of reusing stale pipelines.
4. `compile_pipeline` now checks the cache before lowering, artifact emission, Metal library compilation, and pipeline creation. Hits return cloned Metal artifact/pipeline handles; misses compile once and insert the compiled entry.
5. `pipeline_cache_snapshot` exposes honest Metal cache hit/miss counters through the existing `VyreBackend` public telemetry seam.
6. `shutdown` clears both resident resources and cached Metal pipelines, giving deterministic backend-owned cleanup through the public lifecycle hook.
7. Apple-only tests prove first dispatch records a miss, second identical dispatch records a hit, miss counters do not increment on the hit, and both dispatch outputs stay byte-correct.

Validation evidence:

1. Local Linux: `./cargo_full test -p vyre-driver-metal` passed: 2 non-Apple tests, 0 failures.
2. MacBook Apple GPU: `ssh -o BatchMode=yes -o ConnectTimeout=8 tt-macbook 'cd /Users/thiruthangarathinam/Santh/libs/performance/matching/vyre && CARGO_TARGET_DIR="$HOME/cargo-target-metal-fresh" ./cargo_full test -p vyre-driver-metal'` passed: 9 Apple tests, 0 failures.
3. MacBook Apple GPU after pipeline caching: `ssh -o BatchMode=yes -o ConnectTimeout=8 tt-macbook 'cd /Users/thiruthangarathinam/Santh/libs/performance/matching/vyre && CARGO_TARGET_DIR="$HOME/cargo-target-metal-fresh" VYRE_BACKEND=metal ./cargo_full test -p vyre-conform-runner --features gpu'` passed all `vyre-conform-runner` unit, cert, dispatch-grid, gap-cert, lens, parity-matrix, release-gate, ULP-audit, and doctest sections with 0 failures.
4. The conformance rerun specifically validated that cached Metal pipelines do not break byte parity, dispatch-grid contracts, release-gate expectations, or f32 ULP audit behavior.

## Implementation evidence - 2026-06-07 Metal capability/profile slice

Implemented slice:

1. `vyre-driver-metal::MetalBackend` now reports live Metal workgroup dimensions from `MTLDevice::maxThreadsPerThreadgroup` instead of a hardcoded `[1024, 1, 1]` profile.
2. Metal now overrides `max_compute_invocations_per_workgroup` so scheduler admission does not derive an invalid total-thread product from per-axis Metal limits.
3. Metal now reports `max_storage_buffer_bytes` from `MTLDevice::maxBufferLength`.
4. Metal now emits a typed `DeviceProfile` containing backend id, subgroup support, subgroup size, workgroup limits, max invocation count, max threadgroup-memory bytes, shared-memory availability, max storage-buffer binding size, and transfer-rate-derived memory bandwidth when Metal reports it.
5. Capability booleans remain honest: Metal does not advertise specialization constants or indirect dispatch until those code paths are actually implemented and tested.
6. `DeviceProfile` projections now carry Metal shared-memory and storage-buffer limits into validation and optimizer capability surfaces through the existing shared `DeviceProfile` methods.
7. `shutdown` behavior is tested through the public lifecycle hook: resident handles become stale after backend-owned resources are cleared.

Validation evidence:

1. Local Linux: `./cargo_full test -p vyre-driver-metal` passed: 2 non-Apple tests, 0 failures.
2. MacBook Apple GPU: `ssh -o BatchMode=yes -o ConnectTimeout=8 tt-macbook 'cd /Users/thiruthangarathinam/Santh/libs/performance/matching/vyre && CARGO_TARGET_DIR="$HOME/cargo-target-metal-fresh" ./cargo_full test -p vyre-driver-metal'` passed: 11 Apple tests, 0 failures.
3. MacBook Apple GPU after capability/profile changes: `ssh -o BatchMode=yes -o ConnectTimeout=8 tt-macbook 'cd /Users/thiruthangarathinam/Santh/libs/performance/matching/vyre && CARGO_TARGET_DIR="$HOME/cargo-target-metal-fresh" VYRE_BACKEND=metal ./cargo_full test -p vyre-conform-runner --features gpu'` passed all `vyre-conform-runner` unit, cert, dispatch-grid, gap-cert, lens, parity-matrix, release-gate, ULP-audit, and doctest sections with 0 failures.
4. The Apple profile test proves live nonzero workgroup, invocation, storage-buffer, and threadgroup-memory limits, and proves unsupported features stay false until implemented.

## Implementation evidence - 2026-06-07 Metal timing seam slice

Implemented slice:

1. `vyre-driver-metal::MetalBackend::dispatch_borrowed_timed` now uses the native Metal command path instead of inheriting the backend default host-only timer.
2. Borrowed timed dispatch reports host wall time plus Metal enqueue and wait timings through the public `TimedDispatchResult` fields.
3. Metal still reports `device_ns: None` until real Metal counter/timestamp support is implemented; the backend does not fabricate device timing.
4. `dispatch_borrowed` now delegates to `dispatch_borrowed_timed`, so borrowed dispatch and timed borrowed dispatch share one compile, bind, command-encode, wait, status-check, output-collection, and cache path.
5. Borrowed and resident dispatch now share one `validate_metal_dispatch_config` helper for cooperative dispatch rejection, zero-iteration rejection, and unsupported repeated-dispatch diagnostics.
6. Apple-only tests prove borrowed timed dispatch returns correct bytes, nonzero wall timing, populated enqueue/wait timing, and no fake device timing.

Validation evidence:

1. Local Linux: `./cargo_full test -p vyre-driver-metal` passed: 2 non-Apple tests, 0 failures.
2. MacBook Apple GPU: `ssh -o BatchMode=yes -o ConnectTimeout=8 tt-macbook 'cd /Users/thiruthangarathinam/Santh/libs/performance/matching/vyre && CARGO_TARGET_DIR="$HOME/cargo-target-metal-fresh" ./cargo_full test -p vyre-driver-metal'` passed: 12 Apple tests, 0 failures.
3. MacBook Apple GPU after borrowed timing path change: `ssh -o BatchMode=yes -o ConnectTimeout=8 tt-macbook 'cd /Users/thiruthangarathinam/Santh/libs/performance/matching/vyre && CARGO_TARGET_DIR="$HOME/cargo-target-metal-fresh" VYRE_BACKEND=metal ./cargo_full test -p vyre-conform-runner --features gpu'` passed all `vyre-conform-runner` unit, cert, dispatch-grid, gap-cert, lens, parity-matrix, release-gate, ULP-audit, and doctest sections with 0 failures.
4. The conformance rerun specifically validated that routing `dispatch_borrowed` through the timed path did not change byte truth or release-gate behavior.

## Implementation evidence - 2026-06-07 MacBook validation script slice

Implemented slice:

1. `scripts/check_metal_macbook.sh` is now the scripted MacBook validation gate required by the Metal plan.
2. The script is driven by `VYRE_MACBOOK_SSH` and `VYRE_MACBOOK_VYRE_ROOT`, with `VYRE_MACBOOK_CARGO_TARGET_DIR` and `VYRE_MACBOOK_CONNECT_TIMEOUT` as operational knobs.
3. The script executes commands on the remote Apple GPU checkout through SSH with `BatchMode=yes` and a bounded connect timeout.
4. The script sources `scripts/lib/cargo_runner.sh` on the remote checkout and runs `vyre_select_cargo_runner`, so the Mac gate uses the workspace-approved cargo runner instead of hardcoded raw cargo or host-specific command text.
5. The script exposes `driver`, `conformance`, `benchmark`, and `all` modes.
6. `driver` runs `"$CARGO_RUNNER" test -p vyre-driver-metal`.
7. `conformance` runs `VYRE_BACKEND=metal "$CARGO_RUNNER" test -p vyre-conform-runner --features gpu`.
8. `benchmark` builds `vyre-bench`, lists the benchmark registry as JSON, then runs `foundation.elementwise.add.1m` through explicit `--backend cpu-ref`, `--backend wgpu`, and `--backend metal` smoke executions.
9. `docs/metal_driver_and_vyre_innovation_plan.md` now points the MacBook correctness, conformance, benchmark, and complete gates at `scripts/check_metal_macbook.sh` instead of ad hoc SSH snippets.
10. `conform/vyre-conform-runner/tests/release_gate_contracts.rs` now has a contract test that pins the script to the documented environment variables, shared cargo runner, SSH behavior, Metal driver gate, Metal conformance gate, benchmark gate, and supported modes.

Validation evidence:

1. Local syntax: `bash -n scripts/check_metal_macbook.sh` passed.
2. Local focused contract: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts metal_macbook_gate_is_scripted_through_env_and_shared_runner` passed: 1 test, 0 failures.
3. Local release-gate contracts: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts` passed: 13 tests, 0 failures.
4. Scripted MacBook driver gate: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh scripts/check_metal_macbook.sh driver` passed: 12 Apple `vyre-driver-metal` tests, 0 failures.
5. Scripted MacBook conformance gate: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh scripts/check_metal_macbook.sh conformance` passed all `vyre-conform-runner` unit, cert, dispatch-grid, gap-cert, lens, parity-matrix, release-gate, ULP-audit, and doctest sections with 0 failures.

## Implementation evidence - 2026-06-07 Metal benchmark gate slice

Implemented slice:

1. `scripts/check_metal_macbook.sh benchmark` now proves concrete backend execution instead of relying on a broad environment variable.
2. The benchmark gate builds `vyre-bench`, verifies the case registry with `list --format json`, and runs `foundation.elementwise.add.1m` through `cpu-ref`, `wgpu`, and `metal` by explicit `--backend` selection.
3. The benchmark gate is bounded as smoke evidence with `VYRE_ALLOW_FEW_SAMPLES=1` and `--measured-samples 3`; release-grade benchmark evidence still keeps the normal sample floor.
4. `vyre-bench` now links `vyre-driver-metal` so the benchmark binary can acquire the native Metal backend on Apple hosts.
5. `vyre-bench` target-gates `vyre-driver-cuda` out of macOS and `vyre-frontend-rust` target-gates its transitive CUDA dependency out of macOS, removing `cudarc` from the Mac benchmark binary graph.
6. `vyre-bench::link_benchmark_backend_registrations` explicitly retains reference, WGPU, Metal, SPIR-V, and non-macOS CUDA backend registration crates before the CLI queries the backend registry.
7. `vyre-bench::cases::nvme_gpu_ingest` is Linux-only at the module boundary because its runtime primitive is Linux-only `io_uring`.
8. Benchmark environment capture now treats unavailable `nvidia-smi` as environment telemetry instead of a fatal process error, while CUDA backend acquisition remains the fail-closed path for CUDA-specific runs.
9. Per-sample NVML telemetry now runs only for the CUDA backend; CPU reference, WGPU, and Metal runs do not fail because NVIDIA telemetry is absent.
10. `conform/vyre-conform-runner/tests/release_gate_contracts.rs` pins the benchmark script to explicit backend smoke runs, bounded sample count, Metal bench linkage, macOS CUDA dependency gating, Rust frontend CUDA dependency gating, backend-retention wiring, and shared cargo-runner usage.

Validation evidence:

1. Local syntax: `bash -n scripts/check_metal_macbook.sh` passed.
2. Local macOS dependency graph: `./cargo_full tree -p vyre-bench --target aarch64-apple-darwin -i cudarc` reported no dependency path.
3. Local focused release contract: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts metal_macbook_gate_is_scripted_through_env_and_shared_runner` passed: 1 test, 0 failures.
4. Local benchmark release contracts: `./cargo_full test -p vyre-bench --test release_matrix_contracts` passed: 16 tests, 0 failures.
5. Local Linux NVMe ingest contract after Linux-only module gating: `./cargo_full test -p vyre-bench --test nvme_gpu_ingest_telemetry` passed: 5 tests, 0 failures.
6. Scripted MacBook benchmark gate: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh scripts/check_metal_macbook.sh benchmark` passed with explicit `cpu-ref`, `wgpu`, and `metal` smoke executions.

## Implementation evidence - 2026-06-07 complete MacBook gate slice

Validation evidence:

1. Local full release-gate contracts after benchmark script, macOS dependency, backend-retention, and NVML changes: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts` passed: 13 tests, 0 failures.
2. Scripted complete MacBook gate: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh scripts/check_metal_macbook.sh all` passed end to end.
3. The `all` gate driver section passed `vyre-driver-metal`: 12 Apple runtime tests, 0 failures, plus 0 doctest failures.
4. The `all` gate conformance section passed `vyre-conform-runner --features gpu`, including unit tests, main tests, compute pins, cert artifact, cert regression pin, dispatch-grid contracts, gap cert artifact, lens parity, parity matrix, release-gate contracts, ULP audit, and doctests.
5. The `all` gate parity matrix section completed the catalog-scale `parity_matrix_across_all_registered_ops` case successfully.
6. The `all` gate benchmark section passed the native Metal/WGPU/reference benchmark smoke gate after rebuilding `vyre-bench`.

## Implementation evidence - 2026-06-07 benchmark report artifact slice

Implemented slice:

1. `scripts/check_metal_macbook.sh benchmark` now writes one JSON report per backend instead of relying only on process exit status.
2. The benchmark report directory is controlled by `VYRE_MACBOOK_BENCH_OUTPUT_DIR` and defaults to `${CARGO_TARGET_DIR:-target}/vyre-metal-benchmark-smoke` on the remote Apple host.
3. The script asserts every backend report is nonempty, names the expected `selected_backend`, records exactly one smoke case, and records zero failed cases.
4. The report-producing path still uses the same shared cargo-runner selection and bounded `VYRE_ALLOW_FEW_SAMPLES=1` smoke policy.
5. The release-gate contract pins the documented benchmark-output environment variable, `--output "$output"` usage, and report assertions for selected backend, total case count, and zero failures.

Validation evidence:

1. Local syntax: `bash -n scripts/check_metal_macbook.sh` passed.
2. Local focused release contract: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts metal_macbook_gate_is_scripted_through_env_and_shared_runner` passed: 1 test, 0 failures.
3. Local full release-gate contracts: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts` passed: 13 tests, 0 failures.
4. Scripted MacBook benchmark gate with explicit report directory: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh VYRE_MACBOOK_BENCH_OUTPUT_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-benchmark-smoke-reports scripts/check_metal_macbook.sh benchmark` passed and wrote reports to `/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-benchmark-smoke-reports`.

## Implementation evidence - 2026-06-07 native Metal versus WGPU differential slice

Implemented slice:

1. `vyre-driver-metal` now has an Apple-only dev dependency on `vyre-driver-wgpu` for direct native Metal versus WGPU-on-Metal differential testing.
2. `apple_native_metal_matches_wgpu_on_same_program_bytes` builds one multi-input `u32` `Program`, dispatches it through native Metal and WGPU, checks each backend against an explicit byte oracle, and then checks the two backend outputs for byte identity.
3. The test exercises the public `VyreBackend::dispatch` seam for both backends, not private runtime internals.
4. The test covers the plan requirement for WGPU-on-Metal versus native Metal differential testing on the same workload.

Validation evidence:

1. Local Linux: `./cargo_full test -p vyre-driver-metal` passed: 2 non-Apple tests, 0 failures.
2. Scripted MacBook driver gate: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh scripts/check_metal_macbook.sh driver` passed: 13 Apple `vyre-driver-metal` tests, 0 failures, including `apple_native_metal_matches_wgpu_on_same_program_bytes`.

## Implementation evidence - 2026-06-07 Metal output-boundary slice

Implemented slice:

1. `apple_dispatch_handles_empty_and_unaligned_output_ranges` covers two output-boundary cases in one native Metal dispatch.
2. The test declares one zero-element output with `output_byte_range(0..0)` and proves Metal returns an empty output slot instead of dropping the slot or crashing.
3. The test declares one `u32` output with the unaligned one-byte range `1..2`, stores `0x11223344`, and proves Metal output collection returns the trimmed byte `0x33`.
4. The test exercises the shared output-layout planning contract through native Metal dispatch rather than a Metal-only layout helper.

Validation evidence:

1. Local Linux: `./cargo_full test -p vyre-driver-metal` passed: 2 non-Apple tests, 0 failures.
2. Scripted MacBook driver gate: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh scripts/check_metal_macbook.sh driver` passed: 14 Apple `vyre-driver-metal` tests, 0 failures, including `apple_dispatch_handles_empty_and_unaligned_output_ranges`.

## Implementation evidence - 2026-06-07 Metal negative-dispatch slice

Implemented slice:

1. `apple_dispatch_config_errors_are_actionable` covers public native Metal dispatch failures for unsupported cooperative dispatch and invalid explicit zero fixpoint iterations.
2. The cooperative-dispatch check asserts the error names `Metal cooperative grid dispatch` and the `metal` backend.
3. The zero-fixpoint check asserts the error names `fixpoint_iterations=0` and includes actionable `Fix:` guidance.
4. The test exercises `VyreBackend::dispatch` directly, proving the validation path runs before Metal command encoding.

Validation evidence:

1. Local Linux: `./cargo_full test -p vyre-driver-metal` passed: 2 non-Apple tests, 0 failures.
2. Scripted MacBook driver gate: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh scripts/check_metal_macbook.sh driver` passed: 15 Apple `vyre-driver-metal` tests, 0 failures, including `apple_dispatch_config_errors_are_actionable`.

## Implementation evidence - 2026-06-07 post-boundary complete MacBook gate

Validation evidence:

1. Scripted complete MacBook gate after native-vs-WGPU differential, empty/unaligned output boundary, and negative-dispatch additions: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh scripts/check_metal_macbook.sh all` passed end to end.
2. The `all` gate driver section passed `vyre-driver-metal`: 15 Apple tests, 0 failures, plus 0 doctest failures.
3. The `all` gate conformance section passed `vyre-conform-runner --features gpu`, including cert artifact, gap cert artifact, lens parity, catalog-scale parity matrix, release-gate contracts, ULP audit, and doctests.
4. The `all` gate benchmark section passed and wrote per-backend smoke reports to `/Users/thiruthangarathinam/cargo-target-metal-fresh/vyre-metal-benchmark-smoke`.

## Implementation evidence - 2026-06-07 WGPU versus native Metal benchmark comparison slice

Implemented slice:

1. `scripts/check_metal_macbook.sh benchmark` now emits `wgpu-vs-metal.txt` next to the per-backend smoke JSON reports.
2. The comparison artifact records `baseline_backend=wgpu`, `candidate_backend=metal`, the `vyre-bench compare` table for the shared `foundation.elementwise.add.1m` case, and `compare_exit_code`.
3. The script keeps benchmark comparison as evidence, not a hard speed gate; if `vyre-bench compare` reports a regression exit code, the artifact records it and the smoke gate still proves report creation, backend identity, case count, and zero failed cases.
4. The script asserts the comparison artifact is nonempty, names both compared backends, and includes the shared workload id.
5. The release-gate contract pins the comparison artifact name, `vyre-bench compare` invocation, WGPU baseline report, Metal candidate report, and comparison exit-code recording.

Validation evidence:

1. Local syntax: `bash -n scripts/check_metal_macbook.sh` passed.
2. Local focused release contract: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts metal_macbook_gate_is_scripted_through_env_and_shared_runner` passed: 1 test, 0 failures.
3. Local full release-gate contracts: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts` passed: 13 tests, 0 failures.
4. Scripted MacBook benchmark gate with explicit comparison directory: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh VYRE_MACBOOK_BENCH_OUTPUT_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-benchmark-compare-reports scripts/check_metal_macbook.sh benchmark` passed and wrote reports plus `wgpu-vs-metal.txt` to `/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-benchmark-compare-reports`.

## Implementation evidence - 2026-06-07 shared timing-quality profile slice

Implemented slice:

1. `vyre-driver::DeviceProfile` now carries a typed `DeviceTimingQuality` plus explicit `supports_device_timestamps` and `supports_hardware_counters` booleans.
2. The timing-quality ladder is shared across backends: `HostOnly`, `HostEnqueueWait`, `DeviceTimestamps`, and `HardwareCounters`.
3. Conservative and default backend profiles report `HostOnly` with no timestamp or hardware-counter support.
4. Native Metal reports `HostEnqueueWait` because its timed path currently measures host wall, enqueue, and wait phases without fabricating device timestamps or counters.
5. WGPU reports `DeviceTimestamps` only when both timestamp query features are negotiated; otherwise it reports `HostEnqueueWait`.
6. CUDA reports `DeviceTimestamps` because its native timing path can expose device-timestamp timing; it does not claim hardware counters through this profile field.
7. SPIR-V artifact validation reports `HostOnly` because it does not own live runtime timing.
8. `vyre-driver-metal` now asserts the live Apple Metal profile exposes honest timing quality, no fake device timestamps, and no fake hardware counters.
9. This closes the plan's timing seam requirement with typed capability facts instead of prose or backend-specific strings.

Validation evidence:

1. Local shared profile test: `./cargo_full test -p vyre-driver device_profile` passed.
2. Local Metal crate test: `./cargo_full test -p vyre-driver-metal` passed: 2 non-Apple tests, 0 failures.
3. Local WGPU focused compile/test: `./cargo_full test -p vyre-driver-wgpu adapter_caps` passed the three adapter-capability tests and compiled the filtered WGPU test targets.
4. Local CUDA focused compile: `./cargo_full test -p vyre-driver-cuda device_profile` completed successfully; no exact `device_profile` test names were present, so the value was compile coverage across CUDA test targets under the filter.
5. Local SPIR-V focused compile: `./cargo_full test -p vyre-driver-spirv device_profile` completed successfully; no exact `device_profile` test names were present, so the value was compile coverage across SPIR-V test targets under the filter.
6. Scripted MacBook driver gate after timing-quality changes: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh scripts/check_metal_macbook.sh driver` passed: 15 Apple `vyre-driver-metal` tests, 0 failures, including `apple_device_profile_reports_live_metal_limits`.
7. Scripted complete MacBook gate after timing-quality changes: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh scripts/check_metal_macbook.sh all` passed end to end.
8. The `all` gate driver section passed `vyre-driver-metal`: 15 Apple tests, 0 failures, plus 0 doctest failures.
9. The `all` gate conformance section passed `vyre-conform-runner --features gpu`, including cert artifact, cert regression pin, dispatch-grid contracts, gap cert artifact, lens parity, catalog-scale parity matrix, release-gate contracts, ULP audit, and doctests.
10. The `all` gate benchmark section passed and wrote per-backend smoke reports plus comparison artifacts to `/Users/thiruthangarathinam/cargo-target-metal-fresh/vyre-metal-benchmark-smoke`.

## Implementation evidence - 2026-06-07 benchmark backend-profile artifact slice

Implemented slice:

1. `DeviceTimingQuality` now exposes stable report strings through `as_str()`: `host_only`, `host_enqueue_wait`, `device_timestamps`, and `hardware_counters`.
2. `vyre-bench` result JSON now carries an optional top-level `backend_profile` derived from the selected backend's shared `DeviceProfile`.
3. New benchmark profile fields include backend id, timing quality, device timestamp support, hardware counter support, subgroup support, indirect-dispatch support, workgroup limits, shared-memory limit, storage-buffer binding limit, subgroup size, compute units, and memory bandwidth.
4. `execute_suite` captures the first acquired benchmark backend profile once and writes it into the report beside `selected_backend`.
5. Existing report readers remain compatible because `backend_profile` has a serde default.
6. `scripts/check_metal_macbook.sh benchmark` now rejects any per-backend smoke report that lacks `backend_profile`, the matching profile backend id, a valid timing-quality string, or explicit timestamp/counter support fields.
7. `conform/vyre-conform-runner` release-gate contracts now pin the MacBook script to the backend-profile and timing-quality report checks.
8. This makes the timing seam operator-visible in the same benchmark artifacts that already compare `cpu-ref`, `wgpu`, and native `metal`.

Validation evidence:

1. Local shared timing string test: `./cargo_full test -p vyre-driver device_profile` passed: 2 tests, 0 failures.
2. Local benchmark schema test: `./cargo_full test -p vyre-bench result_schema` passed and confirmed benchmark execution populates `backend_profile`.
3. Local benchmark profile projection test: `./cargo_full test -p vyre-bench backend_profile_projects_timing_quality_for_reports` passed.
4. Local script syntax: `bash -n scripts/check_metal_macbook.sh` passed.
5. Local focused MacBook release contract: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts metal_macbook_gate_is_scripted_through_env_and_shared_runner` passed.
6. Local real CLI smoke report: `VYRE_ALLOW_FEW_SAMPLES=1 /mnt/FlareTraining/santh-archive/cargo-target/debug/vyre-bench run --suite smoke --format json --backend cpu-ref --case foundation.elementwise.add.1m --warmup-samples 0 --measured-samples 3 --sample-timeout-secs 30 --determinism-runs 1 --output /tmp/vyre-bench-profile-cpu-ref.json` passed, and grep checks proved `selected_backend=cpu-ref`, `backend_profile.backend=cpu-ref`, `timing_quality=host_only`, and false timestamp/counter support.
7. Scripted MacBook benchmark gate with profile-enforced reports: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh VYRE_MACBOOK_BENCH_OUTPUT_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-benchmark-profile-reports scripts/check_metal_macbook.sh benchmark` passed and wrote validated reports to `/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-benchmark-profile-reports`.
8. Local full release-gate contracts: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts` passed: 13 tests, 0 failures.
9. Local benchmark release matrix and constructor tests: `./cargo_full test -p vyre-bench --test suite_completeness --test release_matrix_contracts` passed: 17 tests, 0 failures.
10. Scripted complete MacBook gate with profile-enforced benchmark reports: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh VYRE_MACBOOK_BENCH_OUTPUT_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-benchmark-profile-all-reports scripts/check_metal_macbook.sh all` passed end to end, including 15 Apple Metal driver tests, full Metal conformance, catalog-scale parity matrix, ULP audit, and benchmark reports under `/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-benchmark-profile-all-reports`.

## Implementation evidence - 2026-06-07 structured benchmark report validation slice

Implemented slice:

1. `vyre-bench` now exposes `validate-report --path <report.json>` as a first-class CLI command.
2. `validate-report` reuses the existing bounded report loader, serde `ReportSchema` parser, summary-evidence validation, and blocker-evidence validation instead of duplicating JSON parsing in shell.
3. `validate-report` accepts `--backend`, `--total-cases`, and `--failed` expectations for gate scripts that need exact artifact identity.
4. `ReportSchema::validate_backend_profile_evidence` validates that expected backend, `selected_backend`, and `backend_profile.backend` agree.
5. The backend-profile validator rejects missing backend profiles for expected-backend checks, invalid timing-quality strings, zero workgroup dimensions, and zero invocation limits.
6. `vyre-bench compare` now prints baseline/candidate selected backend, profile backend, and timing quality before the timing table, so comparison artifacts preserve backend-profile context.
7. `scripts/check_metal_macbook.sh benchmark` now calls `"$bench_bin" validate-report` for each `cpu-ref`, `wgpu`, and `metal` smoke report instead of using repeated JSON regex checks.
8. The WGPU-vs-native-Metal comparison artifact now has enforced `baseline_profile_backend=wgpu`, `candidate_profile_backend=metal`, `baseline_timing_quality=...`, and `candidate_timing_quality=...` lines from the benchmark harness itself.
9. `conform/vyre-conform-runner` release-gate contracts pin the MacBook script to the structured report validator and comparison-profile checks.

Validation evidence:

1. Local script syntax: `bash -n scripts/check_metal_macbook.sh` passed.
2. Local CLI validator tests: `./cargo_full test -p vyre-bench validate_report` passed: `validate_report_command_accepts_backend_profile_contract`, `validate_report_expectations_rejects_missing_backend_profile`, and `validate_report_expectations_rejects_profile_backend_drift`.
3. Local backend-profile tests: `./cargo_full test -p vyre-bench backend_profile` passed, including the report projection and validator tests.
4. Local focused MacBook script contract: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts metal_macbook_gate_is_scripted_through_env_and_shared_runner` passed.
5. Scripted MacBook benchmark gate with structured report validation: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh VYRE_MACBOOK_BENCH_OUTPUT_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-benchmark-validate-report-reports scripts/check_metal_macbook.sh benchmark` passed and wrote validated reports to `/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-benchmark-validate-report-reports`.
6. Scripted complete MacBook gate with structured report validation: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh VYRE_MACBOOK_BENCH_OUTPUT_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-benchmark-validate-report-all-reports scripts/check_metal_macbook.sh all` passed end to end, including 15 Apple Metal driver tests, full Metal conformance, catalog-scale parity matrix, ULP audit, and structured benchmark report validation under `/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-benchmark-validate-report-all-reports`.

## Implementation evidence - 2026-06-07 structured WGPU-vs-Metal comparison artifact slice

Implemented slice:

1. `vyre-bench compare` now accepts `--output <path>` and writes a structured JSON comparison artifact in addition to the existing human-readable table.
2. The comparison artifact schema is `vyre-bench.compare.v1`.
3. The artifact records baseline and candidate run id, suite, selected backend, profile backend, timing quality, source fingerprints, total case count, and failed count.
4. Each comparison case records case id, baseline/candidate p50 wall time, baseline/candidate mean wall time, delta fraction, delta percent, p-value, verdict, and per-case regression status.
5. The top-level `regressed` flag is derived from per-case regression flags.
6. `compare` writes the JSON artifact before returning a regression error, so a regressed comparison still leaves machine-readable evidence.
7. `vyre-bench validate-comparison` reads the artifact through the same bounded report-input path, validates schema, expected baseline backend, expected candidate backend, valid timing-quality strings, nonempty case list, expected case ids, and top-level regression consistency.
8. `scripts/check_metal_macbook.sh benchmark` now writes both `wgpu-vs-metal.txt` and `wgpu-vs-metal.json`, then validates the JSON artifact through `"$bench_bin" validate-comparison`.
9. The Mac benchmark gate still preserves human-readable comparison text while making the comparison evidence machine-checkable.
10. `conform/vyre-conform-runner` release-gate contracts pin the JSON comparison artifact, `compare --output`, and `validate-comparison` wiring.

Validation evidence:

1. Local script syntax: `bash -n scripts/check_metal_macbook.sh` passed.
2. Local comparison artifact test: `./cargo_full test -p vyre-bench compare` passed, including `compare_writes_structured_profile_artifact`.
3. Local comparison validator test: `./cargo_full test -p vyre-bench validate_comparison` passed, including `validate_comparison_rejects_candidate_backend_drift`.
4. Local focused MacBook script contract: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts metal_macbook_gate_is_scripted_through_env_and_shared_runner` passed.
5. Local full release-gate contracts: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts` passed: 13 tests, 0 failures.
6. Scripted MacBook benchmark gate with structured comparison JSON: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh VYRE_MACBOOK_BENCH_OUTPUT_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-benchmark-compare-json-reports scripts/check_metal_macbook.sh benchmark` passed and wrote reports plus comparison artifacts to `/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-benchmark-compare-json-reports`.
7. Scripted complete MacBook gate with structured comparison JSON: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh VYRE_MACBOOK_BENCH_OUTPUT_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-benchmark-compare-json-all-reports scripts/check_metal_macbook.sh all` passed end to end, including 15 Apple Metal driver tests, full Metal conformance, catalog-scale parity matrix, ULP audit, and structured comparison validation under `/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-benchmark-compare-json-all-reports`.

## Implementation evidence - 2026-06-07 MacBook benchmark bundle validation slice

Implemented slice:

1. `vyre-bench` now exposes `validate-benchmark-bundle --dir <path>` as a first-class CLI command for the MacBook benchmark output directory.
2. The bundle validator reuses `load_report`, `validate_report_expectations`, `load_comparison_artifact`, and `validate_comparison_expectations` instead of duplicating JSON parsing in shell.
3. The bundle contract requires `cpu-ref.json`, `wgpu.json`, `metal.json`, `wgpu-vs-metal.json`, and `wgpu-vs-metal.txt`.
4. Each backend report must validate as the expected backend, exactly one case, zero failures, matching `selected_backend`, and matching `backend_profile.backend`.
5. The JSON comparison artifact must validate as WGPU baseline, Metal candidate, include `foundation.elementwise.add.1m`, carry valid timing-quality strings, and have a top-level regression flag matching per-case evidence.
6. The text comparison artifact must be nonempty and include baseline/candidate backend lines, baseline/candidate profile backend lines, baseline/candidate timing-quality lines, and the shared benchmark case id.
7. Missing bundle artifacts now produce contextual errors naming the expected path and the fix, instead of a bare OS error.
8. `scripts/check_metal_macbook.sh benchmark` now runs the bundle validator after per-file report validation and comparison validation.
9. `conform/vyre-conform-runner` release-gate contracts pin the bundle validator invocation in the MacBook script.

Validation evidence:

1. Local script syntax: `bash -n scripts/check_metal_macbook.sh` passed.
2. Local bundle validator tests: `./cargo_full test -p vyre-bench validate_benchmark_bundle` passed, including `validate_benchmark_bundle_accepts_complete_mac_gate_artifacts` and `validate_benchmark_bundle_rejects_missing_comparison_json`.
3. Local comparison validator test: `./cargo_full test -p vyre-bench validate_comparison` passed after the bundle validator changes.
4. Local focused MacBook script contract: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts metal_macbook_gate_is_scripted_through_env_and_shared_runner` passed.
5. Local full release-gate contracts: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts` passed: 13 tests, 0 failures.
6. Scripted MacBook benchmark gate with bundle validation: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh VYRE_MACBOOK_BENCH_OUTPUT_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-benchmark-bundle-reports scripts/check_metal_macbook.sh benchmark` passed and wrote validated artifacts to `/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-benchmark-bundle-reports`.
7. Scripted complete MacBook gate with bundle validation: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh VYRE_MACBOOK_BENCH_OUTPUT_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-benchmark-bundle-all-reports scripts/check_metal_macbook.sh all` passed end to end, including 15 Apple Metal driver tests, full Metal conformance, catalog-scale parity matrix, ULP audit, and benchmark bundle validation under `/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-benchmark-bundle-all-reports`.

## Implementation evidence - 2026-06-07 benchmark bundle manifest slice

Implemented slice:

1. `vyre-bench validate-benchmark-bundle` now emits a content-addressed `vyre-bench.bundle.v1` manifest when `--manifest-output` is provided.
2. The manifest covers the exact five Mac benchmark bundle artifacts: `cpu-ref.json`, `wgpu.json`, `metal.json`, `wgpu-vs-metal.json`, and `wgpu-vs-metal.txt`.
3. Each manifest artifact records relative path, artifact kind, byte length, and BLAKE3 digest.
4. The top-level `bundle_blake3` is computed from sorted artifact metadata and is independent of the output directory path.
5. The bundle validator parses reports and comparison artifacts from the same bounded bytes it hashes, reusing the existing report and comparison validation contracts instead of adding a shell-side verifier.
6. `scripts/check_metal_macbook.sh benchmark` writes `bundle-manifest.json`, asserts schema, artifact count, bundle hash field, and the Metal report entry.
7. `release_gate_contracts::metal_macbook_gate_is_scripted_through_env_and_shared_runner` pins the manifest requirement so the Mac benchmark gate cannot silently stop producing content-addressed evidence.

Validation evidence:

1. Local: `bash -n scripts/check_metal_macbook.sh` passed.
2. Local: `./cargo_full test -p vyre-bench validate_benchmark_bundle` passed: 2 focused tests, 0 failures.
3. Local: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts metal_macbook_gate_is_scripted_through_env_and_shared_runner` passed: 1 focused test, 0 failures.
4. Local: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts` passed: 13 tests, 0 failures.
5. MacBook benchmark: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh VYRE_MACBOOK_BENCH_OUTPUT_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-benchmark-manifest-reports scripts/check_metal_macbook.sh benchmark` passed and wrote the manifest-backed bundle.
6. MacBook complete gate: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh VYRE_MACBOOK_BENCH_OUTPUT_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-benchmark-manifest-all-reports scripts/check_metal_macbook.sh all` passed native Metal driver tests, full Metal conformance, release contracts, ULP audit, and manifest-backed benchmark output.

## Implementation evidence - 2026-06-07 benchmark bundle manifest replay slice

Implemented slice:

1. `vyre-bench validate-benchmark-bundle` now accepts `--manifest-input` and verifies a previously written `vyre-bench.bundle.v1` manifest against the current benchmark artifacts.
2. Manifest replay validates schema, artifact count, relative artifact paths, artifact kinds, per-artifact BLAKE3 shape, normalized top-level `bundle_blake3`, and exact manifest metadata equality against freshly computed artifact evidence.
3. The replay path rejects artifact drift after manifest creation and reports a bundle hash mismatch with actionable fix text.
4. The Mac benchmark gate now runs both manifest creation and manifest replay through `vyre-bench`; shell only asserts that the replayed manifest file exists and carries expected schema fields.
5. The release gate contract pins `--manifest-input "$bundle_manifest"` so replay validation remains load-bearing in the scripted Apple GPU benchmark gate.

Validation evidence:

1. Local: `bash -n scripts/check_metal_macbook.sh` passed.
2. Local: `./cargo_full test -p vyre-bench validate_benchmark_bundle` passed: 3 focused tests, 0 failures, including manifest replay and artifact drift rejection.
3. Local: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts metal_macbook_gate_is_scripted_through_env_and_shared_runner` passed: 1 focused test, 0 failures.
4. Local: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts` passed: 13 tests, 0 failures.
5. MacBook benchmark: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh VYRE_MACBOOK_BENCH_OUTPUT_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-benchmark-manifest-replay-reports scripts/check_metal_macbook.sh benchmark` passed and replay-validated the manifest-backed bundle.

## Implementation evidence - 2026-06-07 benchmark bundle manifest provenance slice

Implemented slice:

1. `vyre-bench.bundle.v1` manifests now include a `provenance` block recording validator command, validator crate version, suite, benchmark case, report backends, baseline backend, and candidate backend.
2. The top-level `bundle_blake3` now includes provenance plus sorted artifact metadata, so provenance drift changes the manifest hash contract instead of remaining unaudited prose.
3. Manifest replay rejects provenance drift independently from artifact-byte drift and reports an actionable provenance mismatch.
4. `scripts/check_metal_macbook.sh benchmark` asserts the stable provenance fields after manifest creation and replay.
5. `release_gate_contracts::metal_macbook_gate_is_scripted_through_env_and_shared_runner` pins the provenance checks in the Mac benchmark gate.

Validation evidence:

1. Local: `bash -n scripts/check_metal_macbook.sh` passed.
2. Local: `./cargo_full test -p vyre-bench validate_benchmark_bundle` passed: 4 focused tests, 0 failures, including artifact drift and provenance drift rejection.
3. Local: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts metal_macbook_gate_is_scripted_through_env_and_shared_runner` passed: 1 focused test, 0 failures.
4. Local: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts` passed: 13 tests, 0 failures.
5. MacBook benchmark: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh VYRE_MACBOOK_BENCH_OUTPUT_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-benchmark-manifest-provenance-reports scripts/check_metal_macbook.sh benchmark` passed and validated the provenance-bearing manifest on the Apple GPU host.

## Implementation evidence - 2026-06-07 benchmark bundle derived provenance slice

Implemented slice:

1. `vyre-bench.bundle.v1` manifest provenance is now derived from validated benchmark reports and the structured comparison artifact instead of being hardcoded by the manifest builder.
2. The Mac-specific expected case/backend constants are now only validation expectations for `scripts/check_metal_macbook.sh benchmark`; the manifest records the suite, case, report backends, baseline backend, and candidate backend observed in the artifacts.
3. Manifest integrity validation normalizes against the manifest's recorded provenance, validates provenance shape, and still rejects edited provenance through the normalized `bundle_blake3` contract.
4. Unit coverage now proves both Mac bundle replay and a custom report/comparison set whose provenance derives as `custom-suite`, `custom.case`, `alpha`, and `beta` rather than the Mac constants.

Validation evidence:

1. Local: `bash -n scripts/check_metal_macbook.sh` passed.
2. Local: `./cargo_full test -p vyre-bench benchmark_bundle` passed: 5 focused tests, 0 failures.
3. Local: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts metal_macbook_gate_is_scripted_through_env_and_shared_runner` passed: 1 focused test, 0 failures.
4. Local: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts` passed: 13 tests, 0 failures.
5. MacBook benchmark: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh VYRE_MACBOOK_BENCH_OUTPUT_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-benchmark-manifest-derived-provenance-reports scripts/check_metal_macbook.sh benchmark` passed with manifest creation, replay, and provenance checks on the Apple GPU host.

## Implementation evidence - 2026-06-07 benchmark bundle source-consistency slice

Implemented slice:

1. `vyre-bench.bundle.v1` provenance now records `source_fingerprint` and `source_tree_fingerprint` derived from validated backend reports and the structured WGPU-vs-Metal comparison artifact.
2. Bundle provenance derivation now rejects mixed-source bundles before manifest output when backend reports or comparison sides disagree on source fingerprint or source-tree fingerprint.
3. Manifest replay validates the source provenance fields as part of the normalized `bundle_blake3` contract, so source metadata edits invalidate the manifest.
4. The Mac benchmark gate asserts the manifest carries both source fingerprint fields after manifest creation and replay.
5. Unit coverage now includes a custom derived-provenance case and an explicit mixed-source-tree rejection case.

Validation evidence:

1. Local: `bash -n scripts/check_metal_macbook.sh` passed.
2. Local: `./cargo_full test -p vyre-bench benchmark_bundle` passed: 6 focused tests, 0 failures.
3. Local: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts metal_macbook_gate_is_scripted_through_env_and_shared_runner` passed: 1 focused test, 0 failures.
4. Local: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts` passed: 13 tests, 0 failures.
5. MacBook benchmark: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh VYRE_MACBOOK_BENCH_OUTPUT_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-benchmark-manifest-source-consistency-reports scripts/check_metal_macbook.sh benchmark` passed with source-consistent manifest creation and replay on the Apple GPU host.

## Implementation evidence - 2026-06-07 benchmark bundle comparison-consistency slice

Implemented slice:

1. `validate-benchmark-bundle` now recomputes the expected WGPU-vs-Metal comparison artifact from the bundled `wgpu.json` and `metal.json` reports using the same `build_comparison_artifact` and `comparison_side` code path used by `vyre-bench compare`.
2. Bundle validation rejects stale or mismatched `wgpu-vs-metal.json` even when backend names, case ID, source fingerprints, and manifest hashes would otherwise look plausible.
3. `ComparisonArtifact`, `ComparisonSide`, and `ComparisonCase` now implement equality for exact replay comparison of structured benchmark evidence.
4. Unit coverage now mutates `wgpu.json` after comparison generation and proves the bundle validator rejects the stale comparison artifact before manifest output.

Validation evidence:

1. Local: `bash -n scripts/check_metal_macbook.sh` passed.
2. Local: `./cargo_full test -p vyre-bench benchmark_bundle` passed: 7 focused tests, 0 failures.
3. Local: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts metal_macbook_gate_is_scripted_through_env_and_shared_runner` passed: 1 focused test, 0 failures.
4. Local: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts` passed: 13 tests, 0 failures.
5. MacBook benchmark: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh VYRE_MACBOOK_BENCH_OUTPUT_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-benchmark-comparison-consistency-reports scripts/check_metal_macbook.sh benchmark` passed with comparison/report consistency validation on the Apple GPU host.

## Implementation evidence - 2026-06-07 benchmark bundle compare-exit-code slice

Implemented slice:

1. `validate-benchmark-bundle` now parses `compare_exit_code=` from `wgpu-vs-metal.txt` and checks it against the structured `wgpu-vs-metal.json` `regressed` verdict.
2. A non-regressed structured comparison now requires `compare_exit_code=0`; a regressed structured comparison requires a nonzero captured compare exit code.
3. Invalid, missing, negative, or contradictory compare exit-code text fails with actionable fix text before manifest output.
4. The Mac benchmark gate explicitly checks that `wgpu-vs-metal.txt` contains `compare_exit_code=` before bundle validation.
5. Unit coverage now mutates the comparison text to `compare_exit_code=7` while JSON says `regressed=false` and proves the bundle validator rejects the contradiction.

Validation evidence:

1. Local: `bash -n scripts/check_metal_macbook.sh` passed.
2. Local: `./cargo_full test -p vyre-bench benchmark_bundle` passed: 8 focused tests, 0 failures.
3. Local: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts metal_macbook_gate_is_scripted_through_env_and_shared_runner` passed: 1 focused test, 0 failures.
4. Local: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts` passed: 13 tests, 0 failures.
5. MacBook benchmark: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh VYRE_MACBOOK_BENCH_OUTPUT_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-benchmark-compare-exit-code-reports scripts/check_metal_macbook.sh benchmark` passed with text exit-code and structured comparison consistency validation on the Apple GPU host.

## Implementation evidence - 2026-06-07 benchmark bundle manifest artifact-set slice

Implemented slice:

1. `vyre-bench.bundle.v1` manifest replay now enforces the exact required artifact set: `cpu-ref.json`, `wgpu.json`, `metal.json`, `wgpu-vs-metal.json`, and `wgpu-vs-metal.txt`.
2. Manifest integrity validation rejects missing required artifacts, duplicate artifact path/kind pairs, unknown artifact entries, and mislabeled artifact kinds even when the top-level `bundle_blake3` has been recomputed.
3. The required artifact-set check runs before normalized manifest hash comparison so structural schema errors are reported directly instead of being hidden as generic hash drift.
4. Unit coverage now mutates manifests with missing, duplicate, unknown, and mislabeled artifact entries, recomputes their bundle hash, and proves replay rejects the schema violation.

Validation evidence:

1. Local: `bash -n scripts/check_metal_macbook.sh` passed.
2. Local: `./cargo_full test -p vyre-bench benchmark_bundle` passed: 9 focused tests, 0 failures.
3. Local: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts metal_macbook_gate_is_scripted_through_env_and_shared_runner` passed: 1 focused test, 0 failures.
4. Local: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts` passed: 13 tests, 0 failures.
5. MacBook benchmark: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh VYRE_MACBOOK_BENCH_OUTPUT_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-benchmark-manifest-artifact-set-reports scripts/check_metal_macbook.sh benchmark` passed with exact artifact-set manifest validation on the Apple GPU host.

## Implementation evidence - 2026-06-07 benchmark bundle CLI contract slice

Implemented slice:

1. Added direct CLI coverage for `vyre-bench validate-benchmark-bundle --dir <bundle> --manifest-output <path>` writing a replayable manifest for a complete benchmark bundle.
2. Added direct CLI coverage for `vyre-bench validate-benchmark-bundle --dir <bundle> --manifest-input <path>` replaying a freshly written manifest through the user-visible command path.
3. Added direct CLI rejection coverage for contradictory comparison text evidence, proving the command returns an actionable error when `compare_exit_code` contradicts structured comparison JSON.
4. The CLI tests reuse the same production bundle validator and synthetic Mac benchmark bundle fixture instead of duplicating validation logic in test-only helpers.

Validation evidence:

1. Local: `bash -n scripts/check_metal_macbook.sh` passed.
2. Local: `./cargo_full test -p vyre-bench benchmark_bundle` passed: 11 focused tests, 0 failures.
3. Local: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts metal_macbook_gate_is_scripted_through_env_and_shared_runner` passed: 1 focused test, 0 failures.
4. Local: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts` passed: 13 tests, 0 failures.
5. MacBook benchmark: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh VYRE_MACBOOK_BENCH_OUTPUT_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-benchmark-cli-bundle-reports scripts/check_metal_macbook.sh benchmark` passed through the CLI manifest write/replay path on the Apple GPU host.

## Implementation evidence - 2026-06-07 benchmark bundle CPU-reference comparison slice

Implemented slice:

1. The Mac benchmark bundle now includes two structured comparison pairs: `wgpu-vs-metal` for native Metal versus WGPU-on-Metal and `cpu-ref-vs-metal` for native Metal versus the reference baseline.
2. `validate-benchmark-bundle` now validates both comparison JSON artifacts, both comparison text artifacts, both text exit-code contracts, and both comparison-to-report consistency relationships.
3. `vyre-bench.bundle.v1` manifests now require seven exact artifacts: three backend reports plus JSON/text artifacts for both comparison pairs.
4. Manifest provenance now records all comparison pairs, while preserving the existing primary WGPU-vs-Metal baseline/candidate fields for script compatibility.
5. The Mac benchmark gate now writes and validates `cpu-ref-vs-metal.json` and `cpu-ref-vs-metal.txt`, and the release contract pins those artifacts.

Validation evidence:

1. Local: `bash -n scripts/check_metal_macbook.sh` passed.
2. Local: `./cargo_full test -p vyre-bench benchmark_bundle` passed: 11 focused tests, 0 failures.
3. Local: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts metal_macbook_gate_is_scripted_through_env_and_shared_runner` passed: 1 focused test, 0 failures.
4. Local: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts` passed: 13 tests, 0 failures.
5. MacBook benchmark: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh VYRE_MACBOOK_BENCH_OUTPUT_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-benchmark-cperef-comparison-reports scripts/check_metal_macbook.sh benchmark` passed with both WGPU-vs-Metal and CPU-ref-vs-Metal comparison artifacts on the Apple GPU host.

## Implementation evidence - 2026-06-07 expanded benchmark bundle full-gate slice

Implemented slice:

1. The first full Mac `all` run after adding `cpu-ref-vs-metal` surfaced a real bundle-validator issue: strict equality over recomputed floating comparison fields rejected a semantically consistent `cpu-ref-vs-metal.json` after JSON round-trip.
2. `validate_comparison_matches_bundle_reports` now checks comparison schema, side metadata, case IDs, p50 values, verdicts, and regression booleans exactly, while validating derived floating fields with a tight relative tolerance.
3. Stale comparison detection remains load-bearing because side metadata, p50 values, verdicts, case order/count, and regression state still reject drift; existing stale-report tests continue to pass.
4. The expanded seven-artifact benchmark bundle now passes through the full `scripts/check_metal_macbook.sh all` path, not only benchmark mode.

Validation evidence:

1. Local: `bash -n scripts/check_metal_macbook.sh` passed.
2. Local: `./cargo_full test -p vyre-bench benchmark_bundle` passed: 11 focused tests, 0 failures.
3. Local: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts metal_macbook_gate_is_scripted_through_env_and_shared_runner` passed: 1 focused test, 0 failures.
4. Local: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts` passed: 13 tests, 0 failures.
5. MacBook complete gate: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh VYRE_MACBOOK_BENCH_OUTPUT_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-benchmark-cperef-comparison-all-rerun-reports scripts/check_metal_macbook.sh all` passed native Metal driver tests, full Metal conformance, release contracts, ULP audit, and expanded seven-artifact benchmark bundle validation on the Apple GPU host.

## Implementation evidence - 2026-06-07 benchmark bundle comparison-pairs gate slice

Implemented slice:

1. The Mac benchmark gate now asserts that `bundle-manifest.json` contains the `comparison_pairs` provenance field.
2. The gate checks both expected comparison relationships: `cpu-ref->metal` and `wgpu->metal`.
3. The release contract pins these manifest checks so the expanded seven-artifact bundle cannot silently lose comparison-pair provenance.

Validation evidence:

1. Local: `bash -n scripts/check_metal_macbook.sh` passed.
2. Local: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts metal_macbook_gate_is_scripted_through_env_and_shared_runner` passed: 1 focused test, 0 failures.
3. Local: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts` passed: 13 tests, 0 failures.
4. MacBook benchmark: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh VYRE_MACBOOK_BENCH_OUTPUT_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-benchmark-comparison-pairs-reports scripts/check_metal_macbook.sh benchmark` passed with manifest comparison-pair provenance checks on the Apple GPU host.

## Testing contract - LegoGate-backed proof model

Testing for this plan must be source-bound, not shape-bound. The test suite must prove the actual contracts in `docs/lego-block-rule.md`, `.internals/skills/testing/SKILL.md`, `vyre-libs/tests/SKILL.md`, and `vyre-libs/tests/skill_md_examples.rs`.

Required testing spine for every implementation slice:

1. Name the owning contract before writing the test: LegoGate, backend dispatch, artifact schema, resident-resource lifecycle, benchmark telemetry, CLI/operator behavior, or public `Program` producer.
2. Prove production behavior through the real seam, not a mocked duplicate path.
3. Add the positive proof, the negative twin, and at least one boundary/adversarial case for every user-visible behavior.
4. For public `Program` producers, prove `validate`, canonical wire round-trip, dispatch on every linked dispatch-capable backend, and byte equality with `cpu-ref` for deterministic outputs.
5. For reusable primitives, prove the owning primitive directly and prove the caller composes it through registered child regions visible to `print-composition` and Gate 1.
6. For backend/runtime work, prove unsupported-platform errors, real native dispatch, resident upload/download/range behavior, borrowed-output behavior, cache hit/miss accounting, cache invalidation, and CPU-reference parity.
7. For performance work, require a benchmark artifact that exposes the metric by name and a release contract pinning that metric so telemetry cannot silently disappear.
8. For documentation tables, require executable examples like `vyre-libs/tests/skill_md_examples.rs` so a table row failing in code becomes a test failure instead of stale prose.

Actual LegoGate examples applied to testing:

1. The `attention` example means a test must reject private `attention_part_*` splits as proof. The passing proof is visible composition through registered `matmul`, `softmax_step`, second `matmul`, and `layer_norm_step` child regions.
2. The Molten visual example means a test must force domain-language reduction before primitive creation. `visual::separable_conv` collapses to `math::conv1d`, pixel packing stays Tier-1 bit operations, color interpolation stays Tier-1 arithmetic, and one-caller SDF logic stays private.
3. The matching decision-table example means router tests must prove the simplest correct block is selected first, then only escalate to richer engines when constraints require it.
4. The SKILL examples mean tests must call production helpers such as `pack_haystack_u32`, `pack_u32_slice`, `scan_guard`, `cached_load_or_compile`, `dedup_regions_inplace`, `compile_regex_set`, `substring_search`, and `aho_corasick` instead of reimplementing expected behavior inside tests.

This plan treats tests as the proof that LegoGate composition is real. A section is not complete because the code is split into smaller files; it is complete when production behavior, routing identity, byte parity, composition visibility, and operator-facing gates all agree.

Source basis read for this contract:

1. `docs/lego-block-rule.md` says reuse is the Gate 1 mechanism. Private helper splitting is not enough.
2. `docs/lego-block-rule.md` uses `attention` to show the required shape: registered `matmul`, `softmax_step`, second `matmul`, and `layer_norm_step` child regions, not `attention_part_*` helpers.
3. `docs/lego-block-rule.md` uses the Molten visual example to show abstraction reduction: proposed visual primitives collapse into `math::conv1d`, Tier-1 bit/arith expressions, and a private one-caller SDF helper.
4. `vyre-libs/tests/SKILL.md` requires public `Program` producers to validate, wire round-trip, dispatch on linked dispatch-capable backends, and match `cpu-ref` bytes for deterministic outputs.
5. `vyre-libs/tests/SKILL.md` provides the matching decision-table model: select the simplest correct block, then escalate only when workload constraints require a richer engine.
6. `vyre-libs/tests/skill_md_examples.rs` proves the documentation rows by calling production helpers directly and asserting behavior plus routing identity.

Required testing shape for every Metal, optimizer, primitive, matching, parsing, graph, and benchmark change in this plan:

1. `Program` truth: generated programs validate, expose the expected generator/region chain, and round-trip through canonical wire bytes.
2. CPU oracle truth: deterministic outputs match `cpu-ref` exactly, including file/line/case identity where the harness has that metadata.
3. Backend parity truth: every dispatch-capable linked backend either returns byte-identical output or an actionable structured unsupported error.
4. LegoGate truth: `print-composition` and Gate 1 expose registered child-region reuse for composed operations; private helper splits do not count.
5. Decision-table truth: routing tables must have runnable tests equivalent to `skill_md_examples.rs`, proving the selected primitive, helper, cache path, dedup path, and escalation behavior.
6. Adversarial truth: tests cover zero-length buffers, boundary workgroup sizes, oversized dimensions, malformed artifacts, missing metadata, duplicate spans, bad cache bytes, and unsupported backend capabilities.
7. Persistence truth: cache and resident-buffer paths prove cold compile/upload, warm reuse without recompilation, invalidation on option/device/source drift, ranged downloads, and deterministic cleanup.
8. Benchmark truth: reports, comparison JSON, comparison text, manifests, source fingerprints, artifact BLAKE3, and compare-exit-code evidence must be replay-validatable from bytes on disk.
9. Mac truth: `scripts/check_metal_macbook.sh all` remains the Apple GPU release gate for native Metal driver tests, conformance, ULP audit, and benchmark bundle validation.
10. Dedup truth: before adding a helper, tests or review evidence name the existing primitive searched and reused, or prove the new helper has a single narrow owner until a second real consumer exists.

Concrete gates attached to this plan:

1. `./cargo_full test -p vyre-emit-metal` for artifact schema, MSL emission, binding metadata, deterministic hashes, and actionable emitter failures.
2. `./cargo_full test -p vyre-driver-metal` on the MacBook for native acquisition, registration, capability reporting, dispatch, resident execution, and parity slices.
3. `./cargo_full test -p vyre-driver` for backend contract stability and cross-driver behavior.
4. `./cargo_full test -p vyre-libs --test skill_md_examples` for documented decision-table truth where matching helpers are touched.
5. `./cargo_full run -p xtask --bin xtask -- gate1` for workspace composition pressure.
6. `./cargo_full run -p xtask --bin xtask -- print-composition <op_id>` for every newly composed public op in this plan.
7. `./cargo_full test -p vyre-conform-runner --test release_gate_contracts` for scripted release-gate drift.
8. `scripts/check_metal_macbook.sh all` with explicit `VYRE_MACBOOK_*` paths for full Apple GPU validation.

Implementation evidence - 2026-06-07 benchmark bundle comparison-pairs manifest slice:

Implemented slice:

1. `scripts/check_metal_macbook.sh benchmark` now asserts `comparison_pairs`, `cpu-ref->metal`, and `wgpu->metal` in `bundle-manifest.json`.
2. `release_gate_contracts::metal_macbook_gate_is_scripted_through_env_and_shared_runner` pins those manifest checks so expanded comparison-pair provenance remains load-bearing.
3. The expanded seven-artifact benchmark bundle now proves its manifest names both structured comparison relationships, not only the primary WGPU-vs-Metal baseline/candidate fields.

Validation evidence:

1. Local: `bash -n scripts/check_metal_macbook.sh` passed.
2. Local: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts metal_macbook_gate_is_scripted_through_env_and_shared_runner` passed: 1 focused test, 0 failures.
3. Local: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts` passed: 13 tests, 0 failures.
4. MacBook benchmark: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh VYRE_MACBOOK_BENCH_OUTPUT_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-benchmark-comparison-pairs-reports scripts/check_metal_macbook.sh benchmark` passed with comparison-pair manifest checks on the Apple GPU host.

## Implementation evidence - 2026-06-07 restored testing skill doctrine slice

Implemented slice:

1. Restored `.internals/skills/testing/SKILL.md` as the shared testing doctrine inherited by crate-local `tests/SKILL.md` files.
2. Restored `.internals/skills/testing/property.md` as the property-test contract referenced by `vyre-foundation/tests/SKILL.md`.
3. Updated this plan so it no longer treats the parent testing skill as absent and instead names the restored files as active doctrine.
4. The restored parent skill codifies production-path testing, exact observable assertions, positive/negative pairing, real seam tests, `Program` validation/wire/parity obligations, backend contract tests, LegoGate composition tests, and benchmark replay tests.

Validation evidence:

1. Local reference validation script over `*/tests/SKILL.md` passed: every `../../.internals/skills/testing/` reference resolves to an existing file.
2. Local executable-document gate: `./cargo_full test -p vyre-libs --test skill_md_examples` passed: 5 tests, 0 failures.

## Implementation evidence - 2026-06-07 Metal resident sequence API slice

Implemented slice:

1. Added Apple-only coverage for `VyreBackend::dispatch_resident_sequence_read_ranges_timed_into` through the native Metal backend.
2. The ordered sequence test dispatches two real programs over resident resources, proves the second step consumes the first step's resident handoff, reads the requested output range into caller-owned storage, and verifies host enqueue/wait timing evidence without faking device timing.
3. Added Apple-only coverage for `VyreBackend::dispatch_resident_repeated_sequence_read_ranges_into` through the native Metal backend.
4. The repeated sequence test mutates a live `ReadWrite` resident buffer across three public repeated dispatch iterations and proves the final state persists in the resident handle.
5. These tests exercise the shared resident sequence API surface instead of a Metal-private helper, matching the resident execution requirements in this plan.

Validation evidence:

1. Local non-Apple compatibility: `./cargo_full test -p vyre-driver-metal` passed: 2 tests, 0 failures, 0 doctest failures.
2. MacBook focused native Metal sequence gate: `ssh -o BatchMode=yes -o ConnectTimeout=8 tt-macbook 'cd /Users/thiruthangarathinam/Santh/libs/performance/matching/vyre && CARGO_TARGET_DIR="$HOME/cargo-target-metal-fresh" ./cargo_full test -p vyre-driver-metal resident_sequence'` passed: 2 tests, 0 failures.
3. MacBook native Metal driver package: `ssh -o BatchMode=yes -o ConnectTimeout=8 tt-macbook 'cd /Users/thiruthangarathinam/Santh/libs/performance/matching/vyre && CARGO_TARGET_DIR="$HOME/cargo-target-metal-fresh" ./cargo_full test -p vyre-driver-metal'` passed: 17 tests, 0 failures, 0 doctest failures.

Additional validation evidence - 2026-06-07 Metal resident sequence scripted driver gate:

1. Scripted MacBook driver gate: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh scripts/check_metal_macbook.sh driver` passed with 17 native Metal driver tests and 0 doctest failures.

## Implementation evidence - 2026-06-07 Metal borrowed-output API slice

Implemented slice:

1. Added Apple-only coverage for `VyreBackend::dispatch_borrowed_into` through the native Metal backend.
2. The test dispatches a real Metal kernel into caller-owned output storage and proves the output slot capacity is preserved for hot-loop reuse.
3. The test uses the shared public backend method instead of a Metal-private helper, covering the borrowed dispatch requirement in the resident execution slice.

Validation evidence:

1. Local non-Apple compatibility: `./cargo_full test -p vyre-driver-metal` passed: 2 tests, 0 failures, 0 doctest failures.
2. MacBook focused borrowed-output gate: `ssh -o BatchMode=yes -o ConnectTimeout=8 tt-macbook 'cd /Users/thiruthangarathinam/Santh/libs/performance/matching/vyre && CARGO_TARGET_DIR="$HOME/cargo-target-metal-fresh" ./cargo_full test -p vyre-driver-metal borrowed'` passed: 2 tests, 0 failures.
3. MacBook native Metal driver package: `ssh -o BatchMode=yes -o ConnectTimeout=8 tt-macbook 'cd /Users/thiruthangarathinam/Santh/libs/performance/matching/vyre && CARGO_TARGET_DIR="$HOME/cargo-target-metal-fresh" ./cargo_full test -p vyre-driver-metal'` passed: 18 tests, 0 failures, 0 doctest failures.
4. Scripted MacBook driver gate: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh scripts/check_metal_macbook.sh driver` passed with 18 native Metal driver tests and 0 doctest failures.

## Implementation evidence - 2026-06-07 Metal pipeline policy-cache partition slice

Implemented slice:

1. Added Apple-only coverage proving native Metal pipeline cache identity includes shared dispatch policy data.
2. The new cache test dispatches one program with the default policy, dispatches the same program with a `workgroup_override`, then dispatches the changed policy again.
3. The test proves the default policy records a miss, the changed workgroup policy records a separate miss, and the repeated changed policy records a cache hit without another miss.
4. This covers the resident execution requirement that compilation-option changes invalidate or partition cached pipeline state instead of silently reusing stale compiled state.

Validation evidence:

1. Local non-Apple compatibility: `./cargo_full test -p vyre-driver-metal` passed: 2 tests, 0 failures, 0 doctest failures.
2. MacBook focused cache gate: `ssh -o BatchMode=yes -o ConnectTimeout=8 tt-macbook 'cd /Users/thiruthangarathinam/Santh/libs/performance/matching/vyre && CARGO_TARGET_DIR="$HOME/cargo-target-metal-fresh" ./cargo_full test -p vyre-driver-metal cache'` passed: 2 tests, 0 failures.
3. MacBook native Metal driver package: `ssh -o BatchMode=yes -o ConnectTimeout=8 tt-macbook 'cd /Users/thiruthangarathinam/Santh/libs/performance/matching/vyre && CARGO_TARGET_DIR="$HOME/cargo-target-metal-fresh" ./cargo_full test -p vyre-driver-metal'` passed: 19 tests, 0 failures, 0 doctest failures.
4. Scripted MacBook driver gate: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh scripts/check_metal_macbook.sh driver` passed with 19 native Metal driver tests and 0 doctest failures.

## Implementation evidence - 2026-06-07 runtime API full MacBook gate

Validation evidence:

1. Local release contracts: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts` passed: 13 tests, 0 failures.
2. Scripted complete MacBook gate after resident sequence, borrowed-output, and pipeline policy-cache additions: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh VYRE_MACBOOK_BENCH_OUTPUT_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-plan-runtime-api-all-reports scripts/check_metal_macbook.sh all` passed end to end.
3. The `all` gate driver section passed 19 native Metal driver tests and 0 doctest failures.
4. The `all` gate conformance section passed `vyre-conform-runner --features gpu`, including unit tests, main tests, compute pins, certificate artifact, certificate regression pins, dispatch-grid contracts, gap certificate artifact, lens parity, catalog-scale parity matrix, release-gate contracts, ULP audit, and doctests.
5. The catalog-scale `parity_matrix_across_all_registered_ops` case passed.
6. The benchmark gate wrote validated reports to `/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-plan-runtime-api-all-reports`.

## Implementation evidence - 2026-06-07 Metal resident benchmark gate slice

Implemented slice:

1. `scripts/check_metal_macbook.sh benchmark` now runs an additional native Metal resident benchmark sidecar report at `metal-resident-queue-closure.json`.
2. The sidecar uses the existing `dataflow.ifds.skewed.queue_closure.1m` case, which exercises GPU-resident IFDS queue closure through resident sequence APIs instead of adding a new benchmark helper.
3. The resident sidecar is validated with `vyre-bench validate-report --backend metal --total-cases 1 --failed 0` and greps the resident case ID from the generated JSON.
4. The strict seven-artifact `foundation.elementwise.add.1m` bundle remains unchanged, so WGPU-vs-Metal and CPU-ref-vs-Metal bundle replay stays deterministic while the Mac benchmark gate gains resident-performance coverage.
5. `release_gate_contracts::metal_macbook_gate_is_scripted_through_env_and_shared_runner` now pins the resident benchmark report path, case ID, sample count, timeout, output path, and validation path.

Validation evidence:

1. Manual MacBook resident benchmark probe: `ssh -o BatchMode=yes -o ConnectTimeout=8 tt-macbook 'cd /Users/thiruthangarathinam/Santh/libs/performance/matching/vyre && CARGO_TARGET_DIR="$HOME/cargo-target-metal-fresh" VYRE_ALLOW_FEW_SAMPLES=1 "$HOME/cargo-target-metal-fresh/debug/vyre-bench" run --suite smoke --format json --backend metal --case dataflow.ifds.skewed.queue_closure.1m --warmup-samples 0 --measured-samples 1 --sample-timeout-secs 60 --determinism-runs 1 --output "$HOME/cargo-target-metal-fresh/metal-resident-smoke-probe.json" >/dev/null && "$HOME/cargo-target-metal-fresh/debug/vyre-bench" validate-report --path "$HOME/cargo-target-metal-fresh/metal-resident-smoke-probe.json" --backend metal --total-cases 1 --failed 0'` passed.
2. Local shell syntax: `bash -n scripts/check_metal_macbook.sh` passed.
3. Local focused release contract: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts metal_macbook_gate_is_scripted_through_env_and_shared_runner` passed: 1 test, 0 failures.
4. Local full release contracts: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts` passed: 13 tests, 0 failures.
5. Scripted MacBook benchmark gate: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh VYRE_MACBOOK_BENCH_OUTPUT_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-resident-benchmark-gate-reports scripts/check_metal_macbook.sh benchmark` passed and wrote benchmark reports under `/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-resident-benchmark-gate-reports`.
6. Scripted complete MacBook gate after resident benchmark wiring: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh VYRE_MACBOOK_BENCH_OUTPUT_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-resident-benchmark-all-reports scripts/check_metal_macbook.sh all` passed end to end, including 19 native Metal driver tests, full Metal conformance, catalog-scale parity matrix, release contracts, ULP audit, and benchmark reports under `/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-resident-benchmark-all-reports`.

Additional validation evidence - 2026-06-07 resident benchmark metric assertion slice:

1. `scripts/check_metal_macbook.sh benchmark` now asserts `dataflow_ifds_closure_resident_buffers` and `dataflow_ifds_closure_resident_reset_bytes` in `metal-resident-queue-closure.json`, making the resident sidecar prove resident-specific execution metrics rather than only case success.
2. Remote JSON inspection of `/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-resident-benchmark-gate-reports/metal-resident-queue-closure.json` confirmed the report contains `dataflow.ifds.skewed.queue_closure.1m` and resident IFDS closure metric keys.
3. Local shell syntax: `bash -n scripts/check_metal_macbook.sh` passed.
4. Local focused release contract: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts metal_macbook_gate_is_scripted_through_env_and_shared_runner` passed: 1 test, 0 failures.
5. Local full release contracts: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts` passed: 13 tests, 0 failures.
6. Scripted MacBook benchmark gate with resident metric assertions: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh VYRE_MACBOOK_BENCH_OUTPUT_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-resident-metric-gate-reports scripts/check_metal_macbook.sh benchmark` passed and wrote benchmark reports under `/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-resident-metric-gate-reports`.

## Implementation evidence - 2026-06-07 Metal shutdown cache-invalidation slice

Implemented slice:

1. Added Apple-only coverage proving `MetalBackend::shutdown()` invalidates native Metal compiled pipeline cache entries.
2. The test dispatches one program, proves the second dispatch hits the pipeline cache, calls `shutdown()`, then dispatches the same program again and requires a new cache miss instead of a stale hit.
3. The test proves historical cache counters remain observable across shutdown while cached pipeline entries are cleared and rebuilt on the next dispatch.
4. This completes the lifecycle side of the resident execution cache requirement alongside the earlier workgroup-policy cache partition test.

Validation evidence:

1. Local non-Apple compatibility: `./cargo_full test -p vyre-driver-metal` passed: 2 tests, 0 failures, 0 doctest failures.
2. MacBook focused cache gate: `ssh -o BatchMode=yes -o ConnectTimeout=8 tt-macbook 'cd /Users/thiruthangarathinam/Santh/libs/performance/matching/vyre && CARGO_TARGET_DIR="$HOME/cargo-target-metal-fresh" ./cargo_full test -p vyre-driver-metal cache'` passed: 3 tests, 0 failures.
3. MacBook native Metal driver package: `ssh -o BatchMode=yes -o ConnectTimeout=8 tt-macbook 'cd /Users/thiruthangarathinam/Santh/libs/performance/matching/vyre && CARGO_TARGET_DIR="$HOME/cargo-target-metal-fresh" ./cargo_full test -p vyre-driver-metal'` passed: 20 tests, 0 failures, 0 doctest failures.
4. Scripted MacBook driver gate: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh scripts/check_metal_macbook.sh driver` passed with 20 native Metal driver tests and 0 doctest failures.
5. Scripted complete MacBook gate: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh VYRE_MACBOOK_BENCH_OUTPUT_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-cache-shutdown-all-reports scripts/check_metal_macbook.sh all` passed end to end, including 20 native Metal driver tests, full Metal conformance, catalog-scale parity matrix, release contracts, ULP audit, and benchmark reports under `/Users/thiruthangarathinam/cargo-target-metal-fresh/metal-cache-shutdown-all-reports`.

## Implementation evidence - 2026-06-07 Metal resident transfer negative-boundary slice

Implemented slice:

1. Added Apple-only coverage for native Metal resident transfer range failures.
2. The test proves full resident upload larger than the allocation fails with an actionable range/allocation diagnostic.
3. The test proves ranged resident upload crossing the allocation end fails with the invalid byte range and allocation size.
4. The test proves ranged resident download crossing the allocation end fails with the invalid byte range and allocation size.
5. The test proves batched resident ranged download rejects range/output-count mismatches before touching caller output buffers.
6. This closes resident transfer fail-closed coverage beyond the existing happy-path full/ranged/batch transfer and stale-handle tests.

Validation evidence:

1. Local non-Apple compatibility: `./cargo_full test -p vyre-driver-metal` passed: 2 tests, 0 failures, 0 doctest failures.
2. MacBook focused resident-transfer gate: `ssh -o BatchMode=yes -o ConnectTimeout=8 tt-macbook 'cd /Users/thiruthangarathinam/Santh/libs/performance/matching/vyre && CARGO_TARGET_DIR="$HOME/cargo-target-metal-fresh" ./cargo_full test -p vyre-driver-metal resident_transfer'` passed: 2 tests, 0 failures.
3. MacBook native Metal driver package: `ssh -o BatchMode=yes -o ConnectTimeout=8 tt-macbook 'cd /Users/thiruthangarathinam/Santh/libs/performance/matching/vyre && CARGO_TARGET_DIR="$HOME/cargo-target-metal-fresh" ./cargo_full test -p vyre-driver-metal'` passed: 21 tests, 0 failures, 0 doctest failures.
4. Scripted MacBook driver gate: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh scripts/check_metal_macbook.sh driver` passed with 21 native Metal driver tests and 0 doctest failures.

## Implementation evidence - 2026-06-07 Metal resident dispatch negative-boundary slice

Implemented slice:

1. Added Apple-only coverage for native Metal resident dispatch resource validation failures.
2. The test proves resident dispatch rejects a missing output resource and reports expected versus received resource counts.
3. The test proves resident dispatch rejects stale output handles after `free_resident` and reports the handle lifetime problem.
4. The test proves resident dispatch rejects an undersized resident output allocation and reports required versus actual byte counts.
5. This closes fail-closed coverage for resident dispatch binding resources beyond the existing happy-path binding-order and persisted-output test.

Validation evidence:

1. Local non-Apple compatibility: `./cargo_full test -p vyre-driver-metal` passed: 2 tests, 0 failures, 0 doctest failures.
2. MacBook focused resident-dispatch gate: `ssh -o BatchMode=yes -o ConnectTimeout=8 tt-macbook 'cd /Users/thiruthangarathinam/Santh/libs/performance/matching/vyre && CARGO_TARGET_DIR="$HOME/cargo-target-metal-fresh" ./cargo_full test -p vyre-driver-metal resident_dispatch'` passed: 2 tests, 0 failures.
3. MacBook native Metal driver package: `ssh -o BatchMode=yes -o ConnectTimeout=8 tt-macbook 'cd /Users/thiruthangarathinam/Santh/libs/performance/matching/vyre && CARGO_TARGET_DIR="$HOME/cargo-target-metal-fresh" ./cargo_full test -p vyre-driver-metal'` passed: 22 tests, 0 failures, 0 doctest failures.
4. Scripted MacBook driver gate: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh scripts/check_metal_macbook.sh driver` passed with 22 native Metal driver tests and 0 doctest failures.

## Implementation evidence - 2026-06-07 Metal backend metric snapshot and testing-spine slice

Implemented slice:

1. The testing contract section now directly names the checked-in LegoGate and SKILL sources that govern this plan: `docs/lego-block-rule.md`, `.internals/skills/testing/SKILL.md`, `vyre-libs/tests/SKILL.md`, and `vyre-libs/tests/skill_md_examples.rs`.
2. The plan now requires every implementation slice to name its owning contract, prove production behavior through the real seam, cover positive/negative/adversarial cases, and pin any performance telemetry in release contracts.
3. The plan now records the actual LegoGate examples: `attention` must prove visible registered child-region reuse instead of private `attention_part_*` splits, and the Molten visual example must reduce domain-language primitives into existing math or Tier-1 IR when that is the real operation.
4. The Metal benchmark gate now requires native backend metric telemetry in both the foundation Metal report and resident queue-closure sidecar report: `metal_pipeline_cache_hits`, `metal_pipeline_cache_misses`, `metal_resident_buffer_count`, and `metal_resident_bytes`.
5. The release-gate contract pins those metric names so benchmark telemetry cannot silently disappear.

Validation evidence:

1. `bash -n scripts/check_metal_macbook.sh` passed locally.
2. `./cargo_full test -p vyre-conform-runner --test release_gate_contracts` passed locally: 13 tests, 0 failures.
3. `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh VYRE_MACBOOK_BENCH_OUTPUT_DIR=/Users/thiruthangarathinam/cargo-target-metal-metric-assertion-reports scripts/check_metal_macbook.sh benchmark` passed and wrote reports to `/Users/thiruthangarathinam/cargo-target-metal-metric-assertion-reports`.
4. `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh VYRE_MACBOOK_BENCH_OUTPUT_DIR=/Users/thiruthangarathinam/cargo-target-metal-metric-snapshot-all-reports scripts/check_metal_macbook.sh all` passed.
5. The full MacBook gate included native `vyre-driver-metal` tests: 23 tests, 0 failures, 0 doctest failures.
6. The full MacBook gate included `vyre-conform-runner` unit, integration, parity, release-contract, and ULP-audit tests, including the parity matrix across all registered ops and the Metal MacBook release-contract test.
7. The full MacBook gate wrote benchmark reports to `/Users/thiruthangarathinam/cargo-target-metal-metric-snapshot-all-reports`, exercising the scripted metric assertions as part of the operator-facing `all` command.

## Implementation evidence - 2026-06-07 Metal resident ranged batch readback fusion slice

Implemented slice:

1. `vyre-driver-metal` resident ranged batch download now consumes the shared backend-neutral `vyre_driver::resident_transfer_fusion` interval planner instead of looping one `download_resident_range_into` call per requested range.
2. Metal now validates every requested resident handle and byte range, builds the fused copy/view plan, reserves caller output capacity for every fused view, and only then materializes output bytes. A subsequent invalid range cannot partially mutate an earlier output slot.
3. The fused readback path preserves caller-owned output buffer capacity, clears stale bytes for zero-length views, and materializes overlapping/adjacent readback views from the fused output bytes.
4. The new source-contract test pins Metal to `fuse_resident_transfer_intervals`, `reserve_fused_resident_view_outputs`, and `copy_fused_resident_view_into` so the driver cannot regress to a hidden per-range loop while still claiming resident batch support.
5. The Apple live test proves overlapping views, zero-byte views, capacity preservation, and fail-closed preflight behavior against real Metal resident buffers.

Validation evidence:

1. Local non-Apple package gate: `./cargo_full test -p vyre-driver-metal` passed: 3 tests, 0 failures, 0 doctest failures.
2. MacBook focused fused readback gate: `ssh -o BatchMode=yes -o ConnectTimeout=8 tt-macbook 'cd /Users/thiruthangarathinam/Santh/libs/performance/matching/vyre && CARGO_TARGET_DIR="$HOME/cargo-target-metal-fresh" ./cargo_full test -p vyre-driver-metal resident_ranged_batch_download'` passed: 1 test, 0 failures.
3. Scripted complete MacBook gate: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh VYRE_MACBOOK_BENCH_OUTPUT_DIR=/Users/thiruthangarathinam/cargo-target-metal-fused-readback-all-reports scripts/check_metal_macbook.sh all` passed.
4. The full MacBook gate included 25 native Metal driver tests, 0 failures, and 0 doctest failures.
5. The full MacBook gate included Metal conformance, catalog-scale parity matrix across all registered ops, release contracts, ULP audit, and benchmark report generation.
6. The benchmark phase wrote validated reports to `/Users/thiruthangarathinam/cargo-target-metal-fused-readback-all-reports`.

## Implementation evidence - 2026-06-07 Metal compiled-pipeline API slice

Implemented slice:

1. `vyre-driver-metal` now overrides `VyreBackend::compile_native` and `VyreBackend::compile_native_shared` instead of inheriting the shared `Ok(None)` passthrough.
2. `compile_native` returns a real `Arc<dyn CompiledPipeline>` backed by `MetalPersistentPipeline`, the compiled Metal artifact, the native `MTLComputePipelineState`, and the Metal command queue.
3. Compiled Metal dispatch reuses the same `dispatch_planned_buffers_with_queue` command path as normal backend dispatch, so there is still one ABI planner, one output layout path, one Naga `_buffer_sizes` binding path, and one command-buffer execution path.
4. The compiled-pipeline path validates cooperative/repeated/zero-iteration config through the same `validate_metal_dispatch_config` helper used by borrowed and resident dispatch.
5. The Apple live test proves `compile_native` populates the real Metal pipeline cache once, compiled-pipeline dispatches do not re-enter backend lowering/compile cache counters, `dispatch_borrowed_into` preserves caller output slot capacity, and compiled output bytes match direct Metal dispatch bytes.
6. The local source-contract test pins `compile_native`, `compile_native_shared`, `MetalPersistentPipeline`, `impl CompiledPipeline`, the shared command helper, and `Ok(Some(Arc::new(MetalPersistentPipeline` so the backend cannot silently regress to pipeline-mode passthrough.

Validation evidence:

1. Local non-Apple package gate: `./cargo_full test -p vyre-driver-metal` passed: 4 tests, 0 failures, 0 doctest failures.
2. MacBook focused compiled-pipeline gate: `ssh -o BatchMode=yes -o ConnectTimeout=8 tt-macbook 'cd /Users/thiruthangarathinam/Santh/libs/performance/matching/vyre && CARGO_TARGET_DIR="$HOME/cargo-target-metal-fresh" ./cargo_full test -p vyre-driver-metal compile_native'` passed: 2 tests, 0 failures.
3. Scripted complete MacBook gate: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh VYRE_MACBOOK_BENCH_OUTPUT_DIR=/Users/thiruthangarathinam/cargo-target-metal-compile-native-all-reports scripts/check_metal_macbook.sh all` passed.
4. The full MacBook gate included 27 native Metal driver tests, 0 failures, and 0 doctest failures.
5. The full MacBook gate included Metal conformance, catalog-scale parity matrix across all registered ops, release contracts, ULP audit, and benchmark report generation.
6. The benchmark phase wrote validated reports to `/Users/thiruthangarathinam/cargo-target-metal-compile-native-all-reports`.

## Implementation evidence - 2026-06-07 Metal compiled resident-handle dispatch slice

Implemented slice:

1. `vyre-driver-metal` compiled pipelines now support `CompiledPipeline::dispatch_persistent_handles`, `dispatch_persistent_handles_timed`, and `dispatch_persistent_handles_into` for resident-handle inputs instead of returning `UnsupportedFeature`.
2. `MetalBackend` now shares its resident-buffer table with `MetalPersistentPipeline` through one `Arc<Mutex<...>>`, so compiled pipelines observe the same live, freed, and shutdown-cleared resident handles as the backend that created them.
3. Backend resident dispatch and compiled resident dispatch now share `resolve_resident_resources_from_table`, `resident_input_lengths`, `plan_resident_buffers`, `output_binding_layouts`, `metal_slot_map`, and `dispatch_planned_buffers_with_queue` instead of duplicating resident ABI or command encoding logic.
4. Compiled resident dispatch preserves host enqueue/wait timing evidence and keeps `device_ns = None` until real Metal device counters are wired.
5. The source-contract test pins the shared resident table, `Arc::clone`, `dispatch_persistent_handles_timed`, `resolve_resident_resources_from_table`, and `plan_resident_buffers` so this seam cannot silently regress to a compiled-pipeline resident `UnsupportedFeature` path.
6. The Apple live test proves compiled persistent resident dispatch writes the correct bytes, persists output bytes in the resident handle, preserves caller output slot capacity through `dispatch_persistent_handles_into`, does not re-enter backend lowering/compile cache counters, and rejects stale resident handles after free.

Validation evidence:

1. Local non-Apple package gate: `./cargo_full test -p vyre-driver-metal` passed: 5 tests, 0 failures, 0 doctest failures.
2. MacBook focused compiled resident gate: `ssh -o BatchMode=yes -o ConnectTimeout=8 tt-macbook 'cd /Users/thiruthangarathinam/Santh/libs/performance/matching/vyre && CARGO_TARGET_DIR="$HOME/cargo-target-metal-fresh" ./cargo_full test -p vyre-driver-metal compile_native_dispatches_persistent'` passed: 1 test, 0 failures.
3. Scripted complete MacBook gate: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh VYRE_MACBOOK_BENCH_OUTPUT_DIR=/Users/thiruthangarathinam/cargo-target-metal-compiled-resident-all-reports scripts/check_metal_macbook.sh all` passed.
4. The full MacBook gate included 29 native Metal driver tests, 0 failures, and 0 doctest failures.
5. The full MacBook gate included Metal conformance, catalog-scale parity matrix across all registered ops, release contracts, ULP audit, and benchmark report generation.
6. The benchmark phase wrote validated reports to `/Users/thiruthangarathinam/cargo-target-metal-compiled-resident-all-reports`.

## Implementation evidence - 2026-06-07 Metal compiled resident resource-output slice

Implemented slice:

1. `vyre-driver-metal` compiled pipelines now support `CompiledPipeline::dispatch_persistent_resource_outputs` for zero-copy resident output chaining.
2. The resource-output path derives ordered output handles from the shared `BindingPlan` through `resident_output_resources` and rejects borrowed output resources before command submission.
3. The resource-output path submits the compiled resident command through `submit_planned_buffers_with_queue` and returns resident handles without calling `collect_outputs`, so downstream compiled pipelines can consume returned handles without host readback.
4. Normal borrowed/resident dispatch still uses `dispatch_planned_buffers_with_queue`, which wraps the shared submission helper and performs host output collection plus output-budget enforcement.
5. The Apple live test proves a two-stage compiled Metal chain: stage 1 returns the resident `mid` output handle, stage 2 consumes that handle as input, and final bytes persist in the resident `out` handle.
6. The negative case proves borrowed output resources are rejected with an actionable zero-copy fix.

Validation evidence:

1. Local non-Apple package gate: `./cargo_full test -p vyre-driver-metal` passed: 6 tests, 0 failures, 0 doctest failures.
2. MacBook focused zero-copy chain gate: `ssh -o BatchMode=yes -o ConnectTimeout=8 tt-macbook 'cd /Users/thiruthangarathinam/Santh/libs/performance/matching/vyre && CARGO_TARGET_DIR="$HOME/cargo-target-metal-fresh" ./cargo_full test -p vyre-driver-metal zero_copy_chaining'` passed: 1 test, 0 failures.
3. Scripted complete MacBook gate: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh VYRE_MACBOOK_BENCH_OUTPUT_DIR=/Users/thiruthangarathinam/cargo-target-metal-resource-outputs-all-reports scripts/check_metal_macbook.sh all` passed.
4. The full MacBook gate included 31 native Metal driver tests, 0 failures, and 0 doctest failures.
5. The full MacBook gate included Metal conformance, catalog-scale parity matrix across all registered ops, release contracts, ULP audit, and benchmark report generation.
6. The benchmark phase wrote validated reports to `/Users/thiruthangarathinam/cargo-target-metal-resource-outputs-all-reports`.

## Implementation evidence - 2026-06-07 shared pipeline cache miss-reason telemetry slice

Implemented slice:

1. `vyre-driver::pipeline` now owns backend-neutral pipeline cache miss classification through `PipelineCacheMissEvidence` and `PipelineCacheMissReason` instead of leaving reason logic to each backend.
2. The shared classifier distinguishes empty cache, changed normalized Program digest, changed dispatch policy digest, changed device/runtime fingerprint, and fallback key-absent misses.
3. `vyre-driver-metal` keeps its fast raw digest lookup but stores `MetalPipelineCacheIdentity` beside each compiled pipeline, including normalized Program digest, dispatch policy digest, and shared `PipelineDeviceFingerprint`.
4. Metal cache lookup now classifies every miss before compilation and increments stable benchmark-visible reason counters.
5. `backend_metric_snapshot` now exposes `metal_pipeline_cache_miss_empty_cache`, `metal_pipeline_cache_miss_program_changed`, `metal_pipeline_cache_miss_dispatch_policy_changed`, `metal_pipeline_cache_miss_device_or_runtime_changed`, and `metal_pipeline_cache_miss_key_absent` alongside the existing hit/miss and resident-buffer counters.
6. The MacBook benchmark script now requires these reason counters in both the foundation Metal report and the resident queue-closure report, so performance gates report cache hit/miss reason instead of only totals.
7. The release-contract test pins the new benchmark metric names in `scripts/check_metal_macbook.sh` so operator-facing cache telemetry cannot silently regress.
8. The Metal source contract proves the backend uses the shared classifier and exposes stable reason metrics rather than embedding an untested private miss taxonomy.
9. The Apple live metric test proves real empty-cache, same-program dispatch-policy-change, and different-Program miss buckets through native Metal dispatches.

Validation evidence:

1. Local shared classifier gate: `./cargo_full test -p vyre-driver miss_reason` passed: 2 tests, 0 failures.
2. Local Metal source/non-Apple package gate after final import fix: `./cargo_full test -p vyre-driver-metal` passed: 7 tests, 0 failures, 0 doctest failures.
3. Local release-contract gate: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts metal_macbook_gate_is_scripted_through_env_and_shared_runner` passed: 1 test, 0 failures.
4. Local shell syntax gate: `bash -n scripts/check_metal_macbook.sh` passed.
5. Scripted complete MacBook gate: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh VYRE_MACBOOK_BENCH_OUTPUT_DIR=/Users/thiruthangarathinam/cargo-target-metal-cache-reasons-all-reports scripts/check_metal_macbook.sh all` passed.
6. The full MacBook gate included 32 native Metal driver tests, 0 failures, and 0 doctest failures.
7. The full MacBook gate included Metal conformance, catalog-scale parity matrix across all registered ops, release contracts, ULP audit, and benchmark report generation.
8. The benchmark phase wrote validated reports to `/Users/thiruthangarathinam/cargo-target-metal-cache-reasons-all-reports` and the script asserted the new cache-miss reason metric fields in the Metal JSON artifacts.

## Implementation evidence - 2026-06-07 shared pipeline cache identity seam slice

Implemented slice:

1. `vyre-driver::pipeline` now owns `PipelineCacheIdentity`, a shared compiled-pipeline identity object carrying the final lookup digest, normalized Program digest, dispatch policy digest, and `PipelineDeviceFingerprint`.
2. `PipelineCacheIdentity::from_parts` hashes Program, dispatch policy, and device/runtime fingerprint as separate tuple fields, so backends no longer need to smuggle policy into revision strings to get correct cache partitioning.
3. `PipelineCacheIdentity::try_from_program` centralizes Program digest and dispatch-policy digest construction for backends that already have a device/runtime fingerprint.
4. `PipelineCacheMissEvidence::from_identities` and `PipelineCacheMissReason::classify_identities` now provide a shared identity-comparison path for cache miss explainability.
5. `vyre-driver-metal` removed the private `MetalPipelineCacheIdentity` struct and now stores the shared `PipelineCacheIdentity` beside each compiled Metal pipeline.
6. Metal cache lookup now uses `PipelineCacheMissReason::classify_identities` instead of a Metal-local miss-evidence loop.
7. Metal's cache fingerprint revision text now describes Metal artifact schema, MSL version, driver version, and device name, while dispatch policy lives in the shared policy tuple field.
8. The Metal source contract rejects a private Metal cache-identity/classifier regression and requires the shared identity and classifier calls.

Validation evidence:

1. Local shared cache identity gate: `./cargo_full test -p vyre-driver cache_identity` passed: 5 tests, 0 failures.
2. Local shared miss-reason gate: `./cargo_full test -p vyre-driver miss_reason` passed: 3 tests, 0 failures.
3. Local Metal source/non-Apple package gate before and after import cleanup: `./cargo_full test -p vyre-driver-metal` passed: 7 tests, 0 failures, 0 doctest failures.
4. Local release-contract gate: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts metal_macbook_gate_is_scripted_through_env_and_shared_runner` passed: 1 test, 0 failures.
5. Local shell syntax gate: `bash -n scripts/check_metal_macbook.sh` passed.
6. Scripted complete MacBook gate: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh VYRE_MACBOOK_BENCH_OUTPUT_DIR=/Users/thiruthangarathinam/cargo-target-metal-shared-cache-identity-all-reports scripts/check_metal_macbook.sh all` passed.
7. The full MacBook gate included 32 native Metal driver tests, 0 failures, and 0 doctest failures.
8. The full MacBook gate included Metal conformance, catalog-scale parity matrix across all registered ops, release contracts, ULP audit, and benchmark report generation.
9. The benchmark phase wrote validated reports to `/Users/thiruthangarathinam/cargo-target-metal-shared-cache-identity-all-reports`.
10. Final MacBook native driver gate after import cleanup: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh scripts/check_metal_macbook.sh driver` passed: 32 native Metal driver tests, 0 failures, 0 doctest failures.

## Implementation evidence - 2026-06-07 Metal allocation/copy/readback telemetry slice

Implemented slice:

1. `vyre-driver-metal` now tracks backend-level buffer allocation count and allocated bytes through `metal_buffer_allocation_count` and `metal_buffer_allocation_bytes`.
2. Metal dispatch planning now carries per-buffer allocation and host-to-device copy metadata in `PlannedBuffer`, so borrowed inputs, borrowed read-write resources, output buffers, and backend-owned trap sidecars contribute to allocation/copy telemetry without guessing after dispatch.
3. Resident allocation now records logical Metal buffer allocation telemetry after a handle is installed.
4. Resident uploads and ranged uploads now record host-to-device copy count and byte volume after successful shared-buffer writes.
5. Resident downloads, ranged downloads, and fused ranged batch downloads now record device-to-host copy count and byte volume after successful readback.
6. Dispatch output collection now records device-to-host copy count, device-to-host bytes, and `metal_output_readback_bytes` separately, so zero-copy resident resource-output dispatch can be distinguished from host readback paths.
7. `backend_metric_snapshot` now exposes `metal_host_to_device_copy_count`, `metal_host_to_device_bytes`, `metal_device_to_host_copy_count`, `metal_device_to_host_bytes`, and `metal_output_readback_bytes` alongside cache and resident metrics.
8. The Apple live metric test now proves the new counters through real native Metal resident allocation, resident upload, resident ranged download, cache-miss/cache-hit dispatches, and dispatch output readback.
9. `scripts/check_metal_macbook.sh` now requires the allocation/copy/readback metric fields in both `metal.json` and `metal-resident-queue-closure.json` benchmark reports.
10. The release-contract test pins those script assertions so the operator-facing performance gate cannot silently drop transfer telemetry.

Validation evidence:

1. Local Metal source/non-Apple package gate: `./cargo_full test -p vyre-driver-metal` passed: 7 tests, 0 failures, 0 doctest failures.
2. Local release-contract gate: `./cargo_full test -p vyre-conform-runner --test release_gate_contracts metal_macbook_gate_is_scripted_through_env_and_shared_runner` passed: 1 test, 0 failures.
3. Local shell syntax gate: `bash -n scripts/check_metal_macbook.sh` passed.
4. Scripted complete MacBook gate: `VYRE_MACBOOK_SSH=tt-macbook VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh VYRE_MACBOOK_BENCH_OUTPUT_DIR=/Users/thiruthangarathinam/cargo-target-metal-transfer-metrics-all-reports scripts/check_metal_macbook.sh all` passed.
5. The full MacBook gate included 32 native Metal driver tests, 0 failures, and 0 doctest failures.
6. The full MacBook gate included Metal conformance, catalog-scale parity matrix across all registered ops, release contracts, ULP audit, and benchmark report generation.
7. The benchmark phase wrote validated reports to `/Users/thiruthangarathinam/cargo-target-metal-transfer-metrics-all-reports`, and the script asserted the new allocation/copy/readback metric fields in the Metal JSON artifacts.

## Implementation evidence - 2026-06-07 typed preferred backend selection slice

Implemented slice:

1. `vyre-driver` implicit preferred backend acquisition no longer returns the first initialized backend in `BackendPrecedence` order.
2. Explicit `acquire(id)` behavior remains unchanged for conformance, benchmark overrides, and operator-forced backend selection.
3. `acquire_preferred_dispatch_backend` now acquires every non-reference dispatch-capable GPU candidate that initializes on the host, gathers its `DeviceProfile`, and scores the candidate from typed capability facts.
4. The selector scores storage-buffer limit presence, workgroup capacity, subgroup facts, shared memory, native low-precision support, tensor-core support, indirect-dispatch support, timestamp support, hardware-counter support, timing-quality tier, memory bandwidth, backend precedence, and measured acquisition latency.
5. `BackendPrecedence` remains a deterministic tie-breaker after typed capability/timing facts rather than the whole decision.
6. Measured acquisition latency is used as the measured-cost tie-breaker once typed facts and precedence tie.
7. Reference/oracle backends remain excluded from implicit GPU selection; they are still available only through explicit acquisition.
8. The selected candidate is emitted through a `tracing::trace!` record containing backend id, capability score, timing-quality score, precedence, and acquisition nanoseconds.
9. The failure path still reports concrete backend factory failures, reference-only availability, or lack of dispatch-capable GPU backends with actionable fix guidance.

Validation evidence:

1. Focused selector gate: `./cargo_full test -p vyre-driver preferred_selection` passed: 3 tests, 0 failures.
2. The focused tests prove typed capability outranks raw precedence, precedence remains the tie-breaker after typed capability facts, and measured acquisition cost breaks otherwise-equal ties.
3. Full driver package gate: `./cargo_full test -p vyre-driver` passed: 583 unit tests, integration tests, and 6 doc tests, 0 failures.
4. The full driver gate also passed existing backend capability negotiation, registry closure, backend contract, lifecycle, launch validation, and doc-test surfaces after the selector change.

## Testing plan addendum - 2026-06-07 LegoGate-grounded proof matrix

### Source read

1. `docs/lego-block-rule.md` is the canonical LegoGate source. The rule is not file splitting. Before inventing a sub-op, search `vyre-primitives/src/<domain>/` and `vyre-libs/src/{math,nn,hash,matching,parsing,text,security,logical}` for an existing primitive. Add a new primitive only when nothing maps and at least two callers will reuse it.
2. The `attention` example rejects private `attention_part_a` / `attention_part_b` splits. Correct shape is visible `region::wrap_child` composition over registered `matmul`, `softmax_step`, second `matmul`, and `layer_norm_step` primitives so Gate 1, `print-composition`, and optimizer fusion see the structure.
3. The `visual` example shows the abstraction pressure expected during reviews: proposed `visual::separable_conv` dissolves into `math::conv1d`; `pixel_pack`, `pixel_unpack`, and `color_lerp` dissolve into Tier-1 bit/arithmetic expressions; `sdf_rounded_rect` stays private until a second caller exists.
4. `vyre-libs/tests/SKILL.md` is the testing example for a public composition crate: every public function must validate, round-trip through wire format, dispatch on every linked dispatch-capable backend, and match CPU reference bytes.
5. `vyre-libs/tests/skill_md_examples.rs` is the executable documentation pattern: every decision-table row calls production helpers directly. A table row is not trusted unless a runnable test proves the row still routes to real behavior.

### Required testing shape for this plan

1. Composition truth: every new or moved `fn(...) -> Program` must have a composition assertion proving the top region generator is correct and every reusable child is visible through `region::wrap_child`, not hidden in private helper bodies.
2. LegoGate reuse truth: every proposed primitive must include a source-contract test or inventory-level test proving at least two real non-test consumers, or the code remains private to its single consumer.
3. Wire truth: every public Program producer touched by this plan must pass `validate`, `to_wire` / `from_wire` equality, and content-hash stability checks.
4. Backend parity truth: every public Program producer touched by this plan must execute on `vyre-reference` and every linked dispatch-capable backend, with byte-exact output except where the backend-transcendental-aware ULP contract explicitly applies.
5. Failure evidence truth: every parity, dispatch, acquisition, cache, and persistence failure path must return structured evidence with the op id, backend id, case index when applicable, fix guidance, and replayable witness material when output bytes diverge.
6. Negative twin truth: each positive test needs a nearby invalid-input or invalid-state twin that proves the boundary fails closed with a contextual error instead of panicking, silently passing, or returning an empty result.
7. Adversarial truth: each hot path touched by this plan must include a stress input for zero-sized buffers, maximum dimensions or capped large dimensions, malformed wire bytes, cache corruption, backend unavailability, and repeated acquire/release cycles.
8. Performance truth: each Metal driver optimization must carry a benchmark or telemetry assertion proving allocation count, copy bytes, dispatch count, queue closure behavior, cache-hit behavior, or measured backend-selection facts changed in the intended direction.
9. Persistence truth: resident buffers, pipeline caches, conformance artifacts, and runner state must have round-trip tests proving cold start, warm reuse, corruption handling, and explicit teardown semantics.
10. Documentation truth: every README, SKILL, decision table, CLI help claim, or plan row that describes routing behavior must have an executable test similar to `skill_md_examples.rs`; prose cannot be the only proof.

### Metal driver gates

1. Local unit gate: `./cargo_full test -p vyre-driver` for registry, cache identity, backend selection, pipeline policy, and cross-backend contract tests.
2. Local Metal crate gate: `./cargo_full test -p vyre-driver-metal` for source contracts and non-Apple-safe API invariants.
3. Apple driver gate: `scripts/check_metal_macbook.sh driver` for real MPS/Metal compile, dispatch, resident buffer, cache, and telemetry behavior on the MacBook.
4. Apple full gate: `scripts/check_metal_macbook.sh all` for conformance, parity, release, ULP, and benchmark proof in one artifact directory.
5. Telemetry gate: the Metal benchmark JSON must include cache hit/miss counters, shared cache miss reasons, buffer allocation count/bytes, host-to-device bytes, device-to-host bytes, output readback bytes, resident buffer counts, and queue closure counters.

### Conformance and replay gates

1. Wrong-output gate: a backend/reference mismatch must serialize a replay capsule with op id, backend id, case index, program hash, input hash, reference output hash, backend output hash, first mismatch coordinates, and minimized single-case witness buffers.
2. Panic/error gate: backend panics, dispatch errors, invalid input plans, and acquisition failures must serialize structured `PairResult` failures with fix guidance and no empty message.
3. Release artifact gate: prove-mode JSON must omit replay capsules for passing pairs and include them only for byte-divergence failures.
4. Minimization gate: failing conformance cases must retain the smallest witness surface available to the runner at the time of failure; for per-case dispatch loops this means one failing case, not the whole corpus.

### LegoGate gates

1. Discovery gate: before each new helper or primitive, run the equivalent of name search, op-id search, region-chain search, and Gate 1 composition inspection from `docs/lego-block-rule.md`.
2. Composition gate: `cargo xtask gate1` remains the mechanical floor for composed fraction and loop/node pressure.
3. Reinvention gate: `composition_discipline.rs::no_op_reinvents_another_registered_op` remains the regression net for bodies that fingerprint like existing primitives without routing through them.
4. Decision-table gate: if a table says to use a helper, a test must import that helper and assert exact behavior or exact generator identity, matching the `skill_md_examples.rs` pattern.
5. Promotion gate: a primitive promoted to Tier 2.5 must prove two real consumers and a single-purpose API; otherwise the implementation stays local.

### Test batch policy

1. Batch edits by seam: driver cache, Metal runtime, conformance runner, primitive composition, release gate, docs/source-contracts.
2. Run focused tests after each seam is changed and the full relevant gate after the seam reaches a stable boundary.
3. Do not weaken a failing contract test to match current behavior. A failing contract is a finding; fix the behavior or narrow the claim.
4. Do not add shape-only tests such as `is_ok()` or `!is_empty()` when exact file, line, op id, backend id, generator id, byte output, error code, metric name, or exit code is available.
5. Every test added by this plan must either prove detection truth, performance truth, persistence truth, composition truth, or operator-visible failure evidence.

## Implementation evidence - 2026-06-07 conformance wrong-output replay capsule slice

### What changed

1. `conform/vyre-conform-runner/src/main.rs` now extends `PairResult` with an optional `replay_capsule` field that is skipped for passing pairs and non-byte-divergence failures.
2. Wrong-output failures in both normal dispatch parity and convergence-lens parity now attach a `ReplayCapsule` instead of returning only prose.
3. The capsule records schema version, op id, backend id, case index, replay command, Program content BLAKE3, witness input BLAKE3, reference output BLAKE3, backend output BLAKE3, hex witness buffers, hex reference output buffers, hex backend output buffers, buffer counts, first mismatch coordinates, and a `single_witness_case` minimization record.
4. Non-divergence failures remain explicit `replay_capsule: None`: backend acquisition, input planning, dispatch errors, dispatch panics, and fixpoint execution errors do not pretend to have byte witnesses.
5. `conform/vyre-conform-runner/tests/release_gate_contracts.rs` now contains a source contract proving every production `BufferParity::Mismatch(detail)` branch attaches `build_replay_capsule(...)`.

### Test evidence

1. `./cargo_full test -p vyre-conform-runner replay_capsule` passed.
2. The focused gate ran `tests::replay_capsule_records_hashes_and_first_byte_mismatch` and proved exact BLAKE3 hex length, witness/reference/backend buffer hex, byte mismatch coordinates, and minimization metadata.
3. The same focused gate ran `conformance_runner_wrong_output_pairs_have_replay_capsules_contract` and proved production mismatch branches remain wired to replay capsules.
4. `./cargo_full test -p vyre-conform-runner` was rerun after the CUDA correctness fixes and passed end-to-end, including library tests, binary unit tests, `_compute_pins`, certificate artifacts, gap certificate artifacts, lens parity, all-registered-op parity matrix, release contracts, ULP audit, and doc tests.

### Resolved failing gate found by validation

1. `prove_emits_signed_certificate_on_gpu_build` originally reported that the live CUDA proof refused to emit a certificate while nine `(backend, op)` pairs diverged from `vyre-reference`.
2. `prove_emits_signed_cuda_release_certificate_on_gpu_build` originally failed for the same live CUDA proof refusal.
3. CUDA workgroup-sum pairs originally failed before PTX emission because canonical pre-emit lowering reported `variable local is referenced before binding` for:
   - `vyre-libs::catalog::reduce::workgroup_sum_f32::consumer_a`
   - `vyre-libs::catalog::reduce::workgroup_sum_f32::consumer_b`
   - `vyre-libs::catalog::reduce::workgroup_sum_u32::consumer_a`
   - `vyre-libs::catalog::reduce::workgroup_sum_u32::consumer_b`
   - `vyre-primitives::reduce::workgroup_sum_f32`
   - `vyre-primitives::reduce::workgroup_sum_u32`
4. `vyre-libs::math::dot` originally failed CUDA module loading with `CUDA_ERROR_INVALID_PTX` for `sm_120` and PTX length 4809 bytes.
5. `vyre-libs::math::reduce_mean` originally diverged from the reference beyond the 4-ULP window.
6. `vyre-libs::nn::softmax` originally diverged from the reference beyond the 128-ULP window.
7. The final CUDA/conformance sweep resolved these failures. `./cargo_full test -p vyre-driver-cuda` and `./cargo_full test -p vyre-conform-runner` both passed end-to-end after the workgroup-entry, child-liveness, MAD-liveness, cast-validation, PTX-cache, and CUDA-graph output-clear fixes.

## Implementation evidence - 2026-06-07 final CUDA graph, cast, and validation sweep

This entry extends the LegoGate-grounded testing plan with the concrete fixes and gates from the final validation sweep. The same boundary rule applies: each behavior is owned by one Lego block, every seam has a proving test, and failure evidence is attached to the owning block rather than hidden behind a broader integration label.

### Boundary fixes landed

- `vyre-foundation` cast validation now allows `F32 -> Bool`, matching the already-shipped runtime evaluator and PTX lowering semantics where nonzero and NaN values are truthy.
- `vyre-driver-cuda` CUDA graph capture now records output-only zero fills before sparse stores in both full graph replay and resident-input graph replay.
- `vyre-driver-cuda` keeps this graph-zeroing rule behind one helper, `record_cuda_graph_output_clears`, mirroring the existing single readback helper boundary.
- The CUDA graph source contract now proves full and resident-input captures share the output-clear helper and include the two capture labels for sparse-output safety.

### Bugs closed by the final sweep

- `generated_cast_matrix_matches_reference_on_live_cuda` rejected `cast_f32_to_bool_word` at validation even though PTX and reference execution already had matching semantics.
- `vectorized_dynamic_affine_sparse_scatter_emits_packed_v4_ptx_and_matches_reference_on_live_cuda` exposed stale bytes in compiled CUDA graph replay: output lane 4 retained a prior value where reference/direct CUDA left a sparse output hole as zero.
- The stale sparse-output issue was a CUDA graph capture seam bug, not a vectorizer bug: direct dispatch cleared output-only allocations; graph replay did not capture that clear.

### Testing evidence

- `./cargo_full test -p vyre-foundation cast` passed: 24 focused cast/validation/runtime tests plus filtered integration tests, including `cast_f32_to_bool_is_allowed`, `u32_to_f32_is_valid`, wire cast round trip, and cast rejection for unsupported `U64 -> F32`.
- `./cargo_full test -p vyre-driver-cuda --test generated_cast_fma_cuda_reference_matrix` passed: direct and compiled live CUDA cast/FMA generated matrices, including `F32 -> Bool`.
- `./cargo_full test -p vyre-driver-cuda --test vectorized_memory_live_cuda` passed: all 6 live CUDA vectorized memory tests, including dynamic affine sparse gather and sparse scatter.
- `./cargo_full test -p vyre-driver-cuda cuda_graph_capture_records_output_clears_for_sparse_outputs` passed: the new CUDA graph output-clear source contract.
- `./cargo_full test -p vyre-driver-cuda` passed end-to-end after all fixes: 414 unit tests, every CUDA integration suite, live CUDA graph replay, generated scalar/memory/cast matrices, IFDS exploded parity, vectorized memory, and doc tests.
- `./cargo_full test -p vyre-conform-runner` passed end-to-end after all fixes: replay capsule unit tests, certificate artifact gates, CUDA release certificate gates, gap certificate gates, lens parity, all-registered-op parity matrix, release gate contracts, ULP audit, and doc tests.

### Testing plan additions carried forward as release contracts

- Every validation matrix entry must have a live backend case if a backend already lowers that type pair.
- Every output-only CUDA execution path must prove zero-fill semantics for sparse stores, including direct dispatch, compiled graph replay, resident replay, and materialized-cache replay.
- Every CUDA graph optimization that removes host work must preserve the direct-dispatch memory initialization contract byte-for-byte.
- Every generated matrix failure must produce either a replay capsule, a lane-specific assertion, or a source-contract failure that names the seam.
- Every cache or replay optimization must prove both cold-input and same-input resident replay behavior; stale-output lanes are correctness bugs, not cache artifacts.

## Testing contract and evidence - 2026-06-07 exhaustive closeout sweep

This section is the operative testing record for the Metal driver, cross-backend conformance, LegoGate composition discipline, and recent CUDA/cast repairs. A gate listed here is not aspirational: it was run against the current Vyre tree through the repository `./cargo_full` wrapper unless explicitly marked as the MacBook SSH script.

### Local Linux gates

- `./cargo_full test -p vyre-foundation cast` passed. This proves the cast validation contract, including the corrected `f32 -> bool` validity edge, without relying on backend dispatch.
- `./cargo_full test -p vyre-driver-cuda --test generated_cast_fma_cuda_reference_matrix` passed. This proves the generated CUDA cast/FMA reference matrix against the CUDA driver surface.
- `./cargo_full test -p vyre-driver-cuda --test vectorized_memory_live_cuda` passed. This proves live CUDA vectorized-memory behavior after the PTX/emitter liveness repairs.
- `./cargo_full test -p vyre-driver-cuda cuda_graph_capture_records_output_clears_for_sparse_outputs` passed. This proves CUDA graph capture now records output-only clears for sparse output regions instead of preserving stale bytes.
- `./cargo_full test -p vyre-driver-cuda` passed across the full CUDA crate suite, including unit tests, CUDA integration suites, vectorized memory coverage, and doctests.
- `./cargo_full test -p vyre-conform-runner` passed across runner unit tests, certificate artifact tests, regression pins, dispatch grid contracts, gap certificate artifact tests, lens parity, all-registered-op parity matrix, release gate contracts, ULP audit, and doctests.
- `./cargo_full test -p vyre-driver-metal` passed locally. This proves the non-Apple/source contracts for Metal driver acquisition, compile-native contracts, resident-handle contracts, cache contracts, resource-output contracts, and script integration.
- `./cargo_full test -p vyre-emit-metal` passed locally. This proves the Metal emitter unit surface.
- `./cargo_full test -p vyre-driver` passed locally across 583 unit tests, integration tests, and doctests. This proves the shared backend abstraction remains coherent after the Metal and CUDA changes.
- `./cargo_full test -p xtask` passed. This proves the xtask semantic gates, evidence parsers, benchmark artifact semantics, completion-audit semantics, duplicate-source scanners, Lego quick checks, and release-gate helpers.
- `./cargo_full run -p xtask --bin xtask -- gate1` passed: 607 audited ops, 0 failures. This proves the LegoGate complexity budget is load-bearing for the current op catalog.
- `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-conform-enforce --test composition_discipline` passed: 7 tests, 0 failures. This proves every op has fixtures, the complexity budget is enforced, duplicate-op reinvention is rejected, exemptions do not grow, and tolerance metadata is not dead behavior outside its definition.
- `bash -n scripts/check_metal_macbook.sh` passed. This proves the MacBook orchestration script is syntactically valid before remote execution.

### Native MacBook Metal gates

Executed with:

```bash
VYRE_MACBOOK_SSH=tt-macbook \
VYRE_MACBOOK_VYRE_ROOT=/Users/thiruthangarathinam/Santh/libs/performance/matching/vyre \
VYRE_MACBOOK_CARGO_TARGET_DIR=/Users/thiruthangarathinam/cargo-target-metal-fresh \
VYRE_MACBOOK_BENCH_OUTPUT_DIR=/Users/thiruthangarathinam/cargo-target-metal-final-audit-all-reports \
scripts/check_metal_macbook.sh all
```

Results:

- Native Metal driver gate passed on the MacBook: 32 tests, 0 failures. This proves live Apple Metal dispatch, live device-limit reporting, backend registration, actionable config errors, literal-store dispatch, declared-output-count readback, internal trap sidecar allocation, empty/unaligned output ranges, borrowed output slot reuse, threadgroup memory allocation, enqueue/wait timing metrics, pipeline cache reuse, compile-native persistent resident handles, resource outputs for zero-copy chaining, compiled-state reuse, ranged resident batch downloads, resident transfer error handling, full/ranged/stale resident transfers, shutdown cleanup, real compiled Metal pipeline contracts, shared resident handle tables, host-readback avoidance for resource outputs, subgroup-size builtin dispatch, backend-neutral resident fusion, resident resource errors, shared cache-miss classification, resident binding-order dispatch, metrics snapshots, repeated resident updates, workgroup-policy cache partitioning, ordered resident sequences, cache invalidation on shutdown, and native Metal/WGPU byte parity.
- Metal conformance runner passed on the MacBook through lib/bin tests, compute pins, certificate artifacts, certificate regression pins, dispatch grid contracts, gap certificate artifacts, lens parity, all-registered-op parity matrix, release gate contracts, ULP audit, and doctests.
- The all-registered-op parity matrix passed on the MacBook in 74.74 seconds for the recorded run.
- Native Metal/WGPU/reference benchmark gate completed and wrote reports to `/Users/thiruthangarathinam/cargo-target-metal-final-audit-all-reports`.

### What this testing proves

- Metal is no longer only a documentation or source-contract backend. The native MacBook path proves real Apple GPU dispatch, persistent resident execution, resource-output chaining, conformance, parity, and benchmark report generation.
- The shared backend seam stayed intact: `vyre-driver`, `vyre-driver-metal`, `vyre-emit-metal`, `vyre-driver-cuda`, and `vyre-conform-runner` all passed their targeted gates.
- LegoGate is enforced by both catalog-level complexity auditing and composition discipline tests. The plan's Lego-block rule is tied to executable gates, not prose.
- The CUDA fixes are covered by targeted regression tests and the full CUDA crate suite. Sparse CUDA graph outputs now get captured clears, and the cast contract now accepts the valid `f32 -> bool` edge.
- The conformance harness produces replay capsules on mismatches and keeps certificate, shard, parity, ULP, and release-gate semantics load-bearing.
