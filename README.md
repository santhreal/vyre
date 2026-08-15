# vyre

vyre is a GPU compiler: you build a program out of registered operations as IR, it compiles the whole graph into one immutable artifact, emits that artifact as a target payload in PTX, WGSL, SPIR-V or MSL, and runs it on the device.

Nothing in vyre computes on the CPU. The single exception is `vyre-reference`, a pure-Rust interpreter that exists to be the oracle every backend is proved byte-identical against. A CPU path that produces a user's answer is a product failure, not a fallback.

```rust
use vyre::Program;
use vyre_libs::nn::linear::linear;

// A composition. IR only: no device, no dialect, no kernel yet.
let program: Program = linear("x", "w", "b");

// Compile the graph to one artifact, then to a payload for this device.
let backend = vyre::backend::select()?;
let output = backend.dispatch(&program, inputs)?;
```

## The megakernel model

Every production compile emits a megakernel artifact. The artifact is the compiler's output type: a validated whole-graph plan with its ABI and liveness, device-neutral, identified by digest. It is the module, not the machine code.

Persistence is not the output type. One resident kernel that never returns to the host is a schedule the compiler may select, and it wins only sometimes: when the graph is long enough that launch and materialization dominate, when occupancy survives fusion, when the device can do the required grid synchronization, and when keeping weights and frontiers and scratch resident pays for itself. It loses on a two-op graph, on a fusion that collapses occupancy through register or shared-memory pressure, and on any device with no device-wide barrier, where the correct plan is a sequence of dispatches. A compiler that always fuses loses to a two-launch plan on half the devices.

Static and persistent routes consume the same artifact class and must produce the same bytes for the same inputs. There is no side door that emits a one-off kernel and skips the planner.

Selecting the plan is a bounded search, not a heuristic wearing a budget type. Legality is decided first and is never priced: an illegal fusion is rejected, not made expensive. Hardware enters as a fact vector, never as a backend name: subgroup width, shared memory, tensor-core shapes, cooperative launch, native f16, resident block limits, measured bandwidth. Candidates are legal schedules, and the unfused baseline always stays in the candidate set so no model can win by deleting the control. The analytic cost model ranks them; if the budget allows, the finalists are compiled and timed on the live adapter with device timestamps, and the measured winner is selected. When nothing can be measured, the model's winner ships and is recorded as unmeasured. Unmeasured is never called autoroute.

Artifact identity stays device-neutral; payloads do not. The same semantic plan digests the same everywhere, while a target payload carries the dialect bytes plus the geometry that won on that device profile, and admission fails closed when a payload's profile does not match the live device.

## The two rules that place code

Two rules decide which crate a file belongs in. Every boundary below is an application of one of them.

**Composed, not rewritten.** A composition returns a `Program` built out of IR that already exists. It belongs in `vyre-libs`, and it belongs there whoever calls it: a model author calling `nn::attention` and the optimizer calling its own passes as Programs are the same kind of thing. What does not belong anywhere is rewriting: logic reimplemented in host Rust that IR already expresses, or a second implementation of a composition that exists. Reuse count is not an admission criterion. A composition called by two dialects is still a composition, and promoting it for popularity turns a boundary into a caller census that moves code back and forth as callers come and go.

**Intrinsic means uncomposable.** An operation belongs in `vyre-primitives` only when it cannot be expressed as IR composition: it requires its own backend emitter arm and its own reference-interpreter arm. Physics, not convenience.

## The spine

### vyre-spec

The frozen vocabulary. Operation definitions, data types, categories, algebraic laws, invariants, token tables, known-answer vectors. It has no dependencies and most of the workspace depends on it.

Not here: anything that executes, allocates, or decides. A contract that runs is no longer a contract, and a crate this widely depended on cannot afford behaviour, because a behavioural change here is a change to every crate at once.

### vyre-macros

The proc macros that generate pass and AST-registry boilerplate.

Not here: logic. A macro that decides something moves the decision out of the source a reader can grep and into an expansion they cannot. Macros exist to remove repetition, not to hold policy.

### vyre-foundation

The IR: program and node and expression, the type system, the memory model, the wire format, validation, visiting, the optimizer, the execution plan, program dispatch.

Not here: application semantics. No operation knows what a neural network is. The reason is direct: almost every crate in the workspace depends on this one, so a domain concept admitted here is a domain concept every crate inherits and none can escape.

## Composition

### vyre-libs

Every composition in the workspace. Each public function returns a `Program` built from existing IR. Consumer domains and compiler-internal domains are equal residents: linear algebra, neural network layers, matching, hashing, parsing, decoding, and equally the solvers, encoding, analysis, scheduling and reasoning the compiler composes for itself.

Not here: anything that names a concrete backend, links an emitter crate, or reimplements in host Rust what IR expresses. The first two invert the dependency, because a composition states what it needs and the driver decides who provides it. The third is the failure the crate exists to prevent.

### vyre-primitives

Marker types the interpreter and emitters dispatch on, and hardware intrinsics: operations that need a dedicated emitter arm in every backend and a dedicated arm in the reference interpreter.

Not here: compositions. This is the crate's live defect, and it is written into its own documentation as a third category of "shared builders reused by two or more dialects." That sentence is the boundary failure: it admits code by caller count instead of by whether it can be composed, and every domain admitted under it belongs in `vyre-libs`.

### vyre-reference

The pure-Rust interpreter for vyre IR, and the only crate permitted to compute on the CPU.

Not here: performance work, and any role other than oracle. It is not a backend and not a fallback. It exists so a backend's answer can be proved identical to a definition, so speed here buys nothing and complexity costs the definition's credibility.

## Compilation and emission

### vyre-lower

Substrate-neutral lowering from a `Program` to a `KernelDescriptor`. The last stage where no dialect exists.

Not here: any dialect. The moment a dialect leaks in, every emitter downstream inherits a decision that was supposed to be theirs, and the neutral descriptor stops being a shared contract.

### vyre-megakernel

The whole-program compile seam: a validated typed graph plus external facts plus an explicit search budget in, one versioned immutable artifact and its target payloads out. It owns candidate generation, fusion legality, the cost model, selection, and the target compiler facets.

Not here: device admission, submission, queues, residency, recovery. Those consume the artifact and must not alter its identity, because identity is what makes two routes comparable and a cache sound. Also not here: any claim of a measured winner that no clock produced.

### vyre-emit-naga, vyre-emit-ptx, vyre-emit-spirv, vyre-emit-metal

Descriptor to dialect. Naga produces the WGSL-family module, PTX produces NVIDIA assembly text, and SPIR-V and Metal route through naga rather than carrying their own lowering.

Not here: lowering decisions, and any second copy of a translation another emitter already owns. Routing through naga is why two of these crates are small; a fork here is two implementations of one language that drift apart in the direction of whichever backend was debugged last.

## Execution

### vyre-driver

Substrate-agnostic backend machinery: the backend trait every concrete driver implements, plus registry, routing, pipelines, bindings, residency, caching and eviction, autotune storage, work queues, dispatch policy, diagnostics.

Not here: a driver name, a dialect string, or a backend-specific error message. This crate is what makes five backends interchangeable, and a concrete detail admitted here is a detail every backend must then pretend to have.

### vyre-driver-cuda, vyre-driver-wgpu, vyre-driver-spirv, vyre-driver-metal, vyre-driver-reference

The concrete backends: CUDA through the driver API, wgpu, SPIR-V, Metal, and the adapter that presents the reference interpreter as a backend so parity runs through the same seam as everything else.

Here and nowhere else: driver names, dialect names, vendor error strings, device quirks. The asymmetry in size is real work, not neglect. Resident dispatch, JIT caches and megakernel scheduling live where the device supports them.

Not here: anything another backend would also need. That belongs in `vyre-driver`, or it becomes five copies that drift.

### vyre-registry-link

The sole owner of the inventory link anchors, and of the per-source floor every registry read is judged against.

Deliberately substrate-bound: this is the one crate that names every backend, because linking is the act of naming them. The floor exists because an inventory that silently comes up short reads exactly like a smaller workspace.

Not here: anything but linking and the floor.

### vyre-runtime

Persistent execution: the resident work queue, the persistent executor, artifact admission, resource residency, pipeline cache, tenancy, scheduling, replay, recovery, and io_uring streaming.

Not here: compilation, and any decision about whether to be persistent. Persistence is selected during compile, in the artifact; this crate executes that selection. A runtime that decides to fuse has taken a decision away from the search that could measure it.

### vyre-pass-engine

The optimizer's own passes, executed as vyre Programs through the dispatcher seam. The compiler running on the device it compiles for.

Not here: host reimplementations of passes that exist as compositions. That is the rewriting rule applied to the compiler itself, and this crate is the proof the rule is livable.

### vyre-aot

Ahead-of-time compilation to target bytes plus a self-contained launcher bundle, for embedded and competition targets where no compiler ships alongside.

Not here: a second compile path. It must produce the same artifact class as every other route, or the bundle is a different compiler with the same name. Nothing in the workspace currently depends on this crate, which is a fact its README must state rather than imply an integration that does not exist.

### vyre

The public facade. Re-exports the IR, the driver, the runtime and the artifact compiler behind backend features. The crate a user adds.

Not here: logic. A facade with behaviour is a fourth place to look for a bug.

## Harnesses and tooling

### vyre-test-support

Shared test fixtures and the registry coverage closure gate.

Not here: production code, and a fixture that exists in a crate that already owns it. Fixture duplication is how two suites end up testing two different programs under one name.

### vyre-conform and vyre-conform-spec

The conformance engine and its witness sets: witness plans, certificates, minimizer, convergence lenses, and the composition laws a certificate is checked against.

Not here: a pass that a backend can satisfy without producing the reference's bytes. Conformance is the claim that every route agrees with the oracle, so a soft pass is worse than no suite.

### vyre-bench

The cross-backend benchmark and parity harness.

Not here: a benchmark whose baseline is vyre's own unfused output. Beating your own slow path is not a result. The baseline is the best available native implementation for that class.

### vyre-debug

Inspection: descriptor dumps and diffs, naga traces, source assignments, dangling analysis, carrier inspection.

Not here: anything a user's program depends on. This crate reads what other crates produced.

### vyre-grammar-gen

The host-side generator for the C11 lexer DFA and LR(1) tables that the parsing compositions load as read-only data.

Host-side is the point: table construction happens once at build time, and the device consumes the table. Not here: parsing. The parser is a composition and lives in `vyre-libs`.

### vyre-safetensors

Bounded safetensors metadata and shard identity.

Not here: tensor data handling beyond metadata and identity. Nothing currently depends on this crate.

### vyre-lints

The workspace lints: production CPU fallbacks, silent GPU skip guards, module forks, consumer coupling, raw IR construction in composition crates.

Each lint exists because the defect it catches shipped once. Not here: style. A lint that fires on formatting trains everyone to ignore the lints that catch a CPU fallback.

### structure-gate

The structural contract: the crate roster, one identity per operation, one home per concept. It also owns the correct source-text reader, the one that handles nested block comments, raw strings, and a character literal that is not a lifetime.

Not here: a second source scanner. The reason this crate matters is that a bad masker desynchronises a brace matcher on the first raw string it meets, and a contract built on a bad masker reports confident nonsense.

### xtask, xtask-registry, xtask-evidence

The gate registry and its runner, the gates that need the live operation registry linked, and the gates that read benchmark and release evidence.

Every check in the workspace is a gate in one registry. There are no categories: not evidence, not composite, not tool, not script. A gate states what it found as findings, the baseline pins that count, growth fails, and a fall is reported so the pin can drop. The sweep enumerates the registry at run time, so a gate registered without a baseline row fails and a row naming no gate fails. A gate that owns a generated artifact checks it by default and regenerates only when asked.

Not here: an exemption. A gate that is allowed to be red is not a gate, and three of them stayed red for a fortnight behind exactly such a field.
