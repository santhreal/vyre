# Vyre architecture

Last verified: 2026-08-17

Vyre 0.8.0 is a GPU compiler. You build a `Program` from registered
operations, compile the whole graph into one immutable `Artifact`, emit a
target payload, and run it on the device. There is no host execution path
and no bytecode interpreter. `vyre-reference` is the only crate allowed to
compute on the CPU, and only as the oracle.

Downstream crates do not own shadow operation identities. The live catalog
is `docs/generated/OP_SCHEMA.json`, read through
`vyre-foundation::operation::OperationRegistry`.

## Two placement rules

**Composed, not rewritten.** A composition returns a `Program` built from
IR that already exists. It belongs in `vyre-libs`, whoever calls it.

**Intrinsic means uncomposable.** An operation belongs in `vyre-primitives`
only when it needs its own backend emitter arm and its own reference
interpreter arm.

## Layers

Every workspace member declares one layer in `docs/CRATE_OWNERSHIP.toml`, and
every layer declares one rank plus the closed set of consumer layers admitted
to it. A production dependency is legal only when the consumer's layer outranks
the dependency's layer and the dependency's layer admits the consumer's. Two
layers share a rank when neither depends on the other, and a layer whose
members reach each other admits itself, because rank does not judge an edge
inside one layer.

Cargo owns the edge set, so the manifest holds no second roster of edges.
`xtask crate-ownership` resolves every internal edge under the union of every
feature and rejects a reversal, a cycle, an edge across a layer pair no
admitted set records, and an admitted layer pair no edge crosses. A new member
with no row fails there, and so does a new edge across a pair nothing records,
which is what stops a direct dependency from expanding rebuild fan-out or
pulling a composition into a driver without a recorded decision.
`docs/CRATE_GRAPH.md` renders the ranks, the admitted sets, and the resolved
graph.

- `vyre-spec` is the frozen vocabulary. It does not execute.
- `vyre-foundation` owns validated `ProgramGraph` and schedule-free
  `LogicalProgramGraph` IR, semantic identity, the host optimizer, and the
  registry. No application semantics.
- `vyre-libs` owns every composition: consumer dialects and the compiler's
  own solvers, encoding, analysis, scheduling, and reasoning. Equal
  residents.
- `vyre-primitives` owns marker types and hardware intrinsics. A composition
  belongs in `vyre-libs`.
- `vyre-lower` owns the sole `Program` to validated `PhysicalKernel`
  boundary. Concrete emitters may borrow its verified `KernelDescriptor`.
- `vyre-megakernel` owns Cross-program composition: candidate generation,
  fusion legality, the cost model, validated `SelectedPlan` schedule IR, and
  selection under an explicit `SearchBudget`. It also owns immutable `Artifact`
  identity and authenticated
  `TargetPayload` construction. It does not own admission or claim a measured
  winner that no clock produced.
- `vyre-megakernel` also owns the `SemanticExecutor` seam. A caller submits a
  validated `LogicalProgramGraph` plus device and external facts, an objective
  and a budget; the compiler selects the schedule and the launch. The seam
  accepts no grid, workgroup, persistence or route, so `vyre-libs`,
  `vyre-pass-engine`, `vyre-driver-reference` and `vyre-bench` declare a
  dependency on it, and the layer DAG admits each of those layers into the
  compiler boundary.
- `vyre-driver` is backend-agnostic machinery. Concrete drivers own names,
  dialects, and device quirks.
- `vyre-runtime` executes the artifact's selected persistence. It does not
  decide whether to be persistent.
- `vyre-pass-engine` runs the optimizer's passes as vyre Programs. `vyre-bench`
  measures that pipeline against the host optimizer on a device, so it declares
  an edge to the pass engine as well as to the compiler.
- `vyre-reference` is the oracle, not a backend and not a fallback.
- `vyre-alloc-probe` counts per-thread heap traffic behind a `GlobalAlloc`
  wrapper. It has no workspace dependency, so a harness binary links the
  counter without linking a test tree.

## Publication classes

Every workspace member declares one publication class in `docs/CRATE_OWNERSHIP.toml` and in its manifest under `[package.metadata.vyre.publication_class]`.

- `stable-consumer-sdk`: Public consumer-facing SDKs (`vyre`, `vyre-libs`, `vyre-safetensors`). Exposes the curated compiler, frontend IR, and library operations.
- `extension-sdk`: Stable extension interfaces (`vyre-foundation`, `vyre-spec`, `vyre-macros`, `vyre-primitives`, `vyre-driver`). Allows out-of-tree frontends, backends, and operations.
- `concrete-backend`: Hardware-specific driver packages (`vyre-driver-cuda`, `vyre-driver-wgpu`, `vyre-driver-metal`, `vyre-driver-spirv`, `vyre-driver-reference`).
- `internal-engine`: Compiler engine packages (`vyre-megakernel`, `vyre-registry-link`, `vyre-reference`, `vyre-pass-engine`, `vyre-runtime`, `vyre-aot`, `vyre-lower`, `vyre-emit-naga`, `vyre-emit-ptx`, `vyre-emit-spirv`, `vyre-emit-metal`, `vyre-debug`).
- `conformance-tooling`: Tooling, gates, and benchmarks (`structure-gate`, `vyre-conform-spec`, `vyre-conform`, `xtask`, `xtask-registry`, `xtask-evidence`, `vyre-alloc-probe`, `vyre-bench`, `vyre-lints`).
- `private-test-support`: Private test harnesses (`vyre-test-support`).

## Consumers

`consumers/` holds independently versioned application packages. Each one is
excluded from the workspace, carries its own lockfile, and depends on the
published `vyre` facade plus selected `vyre-libs` features. A consumer package
declares no dependency on an internal engine package, and a seam test fails
when one reaches an internal path.

- `consumers/vyre-graphics-app` compiles and runs an interactive frame
  pipeline.
- `consumers/vyre-model-compiler` compiles a prefill and decode workload.

A composition a consumer needs is domain-neutral and lives in `vyre-libs`. The
domain vocabulary stays in the consumer package.

## Production route

```text
frontend Program(s)
  -> validated ProgramGraph
  -> validated schedule-free LogicalProgramGraph
  -> validated SelectedPlan in an immutable Artifact
  -> validated PhysicalKernel per selected fusion group
  -> authenticated TargetPayload
  -> driver admission and materialization
  -> ArtifactInstance
  -> typed Submission
  -> completion and readback
```

The logical stage records versioned iteration extents, index maps, tensor
layouts, aliases, effects, dependencies and point bounds before schedule
search. Library compositions cross this boundary through their typed graph
value contracts.

Ordinary library compositions contain schedule-free logical domain, tile and
within-tile identities plus logical barriers. Selected-schedule lowering is the
single boundary that introduces physical invocation, workgroup, local and
barrier IR. Descriptor construction rejects unresolved logical markers.

The foundation schedule schema records phase fission and fusion, axis splitting,
tiling, reorder, vectorization and hierarchy mapping, memory placement,
prefetch, bounded producer/consumer pipelines, recomputation, persistent
queues, neutral compute and device partitions, dispatch cuts, synchronization,
and asymmetric joins. Every applied transform contains typed preconditions,
source regions and phases, an inverse identity checkpoint, deterministic replay,
and checked resource bounds. Distinct phases contain independent logical grids,
workgroup shapes, resource ceilings and parallelism axes.

Every operation registration records neutral schedule constraints. The registry
derives the semantic minimum from the canonical program and composes workgroup
and subgroup widths, uniformity, shared scratch, cooperative launch, memory
ordering, and element policy before search. Conflicting constraints reject the
operation contract instead of becoming candidate prices.

Every production compile emits a megakernel artifact. Persistence is a
schedule inside that artifact, not a second output type. Static and
persistent routes consume the same artifact class and must produce the same
bytes. Hardware enters compile as a fact vector, never as a backend name.
Unmeasured selections are recorded as unmeasured and are never called
autoroute. GPU execution is capability-based: a lane runs a device case only
where the probe reports the capability the case declares. Every release crate
enforces a zero panic budget.

## Chapters

- [Crate boundaries](architecture/crates.md): what each crate owns.
- [The artifact is the output type](architecture/artifact.md): identity,
  persistence, payload admission.
- [Whole-program compile search](architecture/compile-search.md): legality,
  budget, cost model.
- [Parsing](architecture/parsing.md): the language-neutral substrate and
  its frontends.
- [The placement rule](lego-block-rule.md): which crate a new operation
  belongs in.

Machine-readable contracts live next to this file:
`CRATE_OWNERSHIP.toml`, `optimization/OWNERSHIP.toml`,
`optimization/OP_MATRIX.toml`, and `testing/TESTING.toml`.
