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
  dependency on it and `docs/CRATE_OWNERSHIP.toml` records those edges.
- `vyre-driver` is backend-agnostic machinery. Concrete drivers own names,
  dialects, and device quirks.
- `vyre-runtime` executes the artifact's selected persistence. It does not
  decide whether to be persistent.
- `vyre-pass-engine` runs the optimizer's passes as vyre Programs. `vyre-bench`
  measures that pipeline against the host optimizer on a device, so it declares
  an edge to the pass engine as well as to the compiler.
- `vyre-reference` is the oracle, not a backend and not a fallback.

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
autoroute. GPU execution is capability-based on the designated execution host
(`axiomexec`). Every release crate enforces a zero panic budget.

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
