# Vyre architecture

Last verified: 2026-08-12

This guide describes Vyre 0.7.2 and the decided target structure it is
migrating to. Workspace membership, dependency edges, and operation counts
are owned by the generated artifacts: [crate graph](CRATE_GRAPH.md),
[crate ownership registry](CRATE_OWNERSHIP.toml), and
[operation schema](generated/OP_SCHEMA.json). This guide does not restate
those numbers; it explains how the pieces fit and where responsibilities
live.

## System shape

A frontend builds typed `Program` values and adapts them into a validated
`ProgramGraph`. The whole-program compiler selects a bounded legal
schedule and produces an immutable `Artifact`. A registered target
compiler attaches an authenticated `TargetPayload`; the matching driver
materializer admits that payload and creates an `ArtifactInstance`. Typed
bindings produce a `Submission`, then completion and readback. The
reference interpreter is an oracle, not a silent fallback for a requested
target.

```mermaid
flowchart LR
    Frontend[Frontend Program values] --> Graph[Validated ProgramGraph]
    Graph --> Compiler[Whole-program compiler]
    Compiler --> Artifact[Immutable Artifact]
    Artifact --> TargetCompiler[Registered target compiler]
    TargetCompiler --> Payload[Authenticated TargetPayload]
    Payload --> Materializer[Driver admission and materialization]
    Materializer --> Instance[ArtifactInstance]
    Instance --> Submission[Typed Submission]
    Submission --> Completion[Completion and readback]
    Graph --> Reference[Reference oracle]
```

The current release evidence selects CUDA as the preferred backend on
NVIDIA systems. WGPU is the portable GPU route. SPIR-V is a registered
dispatch route. Metal is active on supported Apple targets. See
`release/evidence/backends/backend-matrix.json` for the live probe rows.

## Workspace boundaries

| Boundary | Current owner | Responsibility |
| --- | --- | --- |
| Public facade | `vyre` | Canonical graph compilation, artifact sessions, feature-gated target selection. |
| Stable contracts | `vyre-spec` | Frozen cross-engine analysis, soundness, and interchange schemas. |
| IR, registry, optimizer | `vyre-foundation` | `Program`, `ProgramGraph`, validation, serialization, semantic operation identity, diagnostics, backend-neutral optimization passes. |
| Reusable operations | `vyre-primitives` | Category C hardware intrinsic builders under `src/hardware/`, plus the shared composition builders that still await the Category A move; see below. |
| Library compositions | `vyre-libs` | Product-facing Category A compositions. Becoming the single Category A home; see below. |
| Compiler self-use | `vyre-pass-engine` | Executes the optimizer's own passes as Vyre Programs through the dispatcher seam. |
| Whole-program compiler | `vyre-megakernel` | Bounded legal whole-graph schedules, immutable artifacts, target payloads. |
| Backend contracts | `vyre-driver` | Backend-neutral target compiler, materializer, device, binding, submission, completion, capability contracts. |
| Concrete backends | `vyre-driver-cuda`, `vyre-driver-wgpu`, `vyre-driver-spirv`, `vyre-driver-metal`, `vyre-driver-reference` | Target compilers, materializers, devices; admit authenticated payloads; submit typed work. |
| Runtime | `vyre-runtime` | Compilation orchestration, admission, artifact sessions, recovery, persistence, residency, scheduling, readback. |
| Artifact packaging | `vyre-aot` | Package validated artifacts without owning artifact identity or live dispatch. |
| Frontends | `vyre-frontend-rust` | Lower source-language subsets into backend-neutral `Program` or `ProgramGraph` values. |
| Conformance | `vyre-conform`, `vyre-conform-spec` | Execute canonical artifact routes; own frozen conformance schemas. |

Domain logic does not import a CLI, transport, or concrete backend.
Shared crates use neutral target and artifact terms. Concrete backend API
and shader dialect details stay in the concrete driver or emitter that
owns them.

## Operation categories

Every registered operation is one of two categories. There is no third.

- **Category A: composition.** A backend-neutral `fn(...) -> Program`
  built from lower-tier operations over existing `Expr`/`Node` variants.
  It adds no concrete target lowering. Regions preserve composition
  provenance: the outer region keeps the Cat-A generator, each child
  keeps its primitive generator and a `source_region` naming the parent.
- **Category C: hardware intrinsic.** An operation requiring a dedicated
  hardware contract: a dedicated emitter arm AND a dedicated
  reference-interpreter eval arm. Its registration supplies the neutral
  builder and deterministic fixtures; each supported target supplies a
  keyed lowering facet. Missing facets fail closed.
- **Category B is banned.** Vyre does not execute a general operation
  bytecode interpreter on the host or inside a persistent kernel. A
  program remains typed IR until verified lowering. Raw `Program`
  execution exists only in explicitly named reference, parity, and
  conformance oracle seams.

A host-side runtime capability is not an operation and is not registered.
Indirect dispatch, NVMe ingest and write-back, and zero-copy mapping and
unmapping have no `Program` to lower and no fixture to compare, so they
carry no operation id and no tier. They are reached through the backend
capability surface in `vyre-driver` (`VyreBackend`, `DeviceProfile`,
`AdapterCaps`, `RequiredCapabilities`) and through `vyre-runtime`'s
io_uring ingest driver. `vyre-driver` registered five such ids
(`core.indirect_dispatch`, `io.dma_from_nvme`, `io.write_back_to_nvme`,
`mem.zerocopy_map`, `mem.unmap`) as signature-only records with no
builder; that was one capability holding a second identity, and the
registry now refuses an id whose namespace names no owning crate.

`vyre-foundation::operation::OperationRegistry` is the semantic operation
authority. One registration owns the operation ID, version, tier,
signature, neutral builder, fixtures, laws, tolerance, derived effects,
and capability keys. `generated/OP_SCHEMA.json` and the catalog are
projections, not second identities. Libraries, drivers, and products
do not own shadow operation identities. `docs/lego-block-rule.md` owns the
composition policy: discovery before invention, the two-caller
criterion, Gate 1, and the promotion patch contract.

Run these checks after changing an operation:

```text
cargo_full run --bin xtask -- operation-schema --check
cargo_full run --bin xtask -- list-ops --check
cargo_full run --bin xtask -- catalog --check
```

## Target operation crate structure

Decided 2026-08-12. Migration runs after the dedup campaign in
`DEDUP_PLAN.md`; until it lands, the boundary table above describes the
current tree.

Two operation crates, one per category. Nothing else registers
operations.

- `vyre-primitives` owns Category C: strict hardware intrinsics only,
  under `src/hardware/`. The standalone hardware crate was absorbed on
  2026-08-13. An op that does not require both a dedicated emitter arm
  and a dedicated interpreter arm does not belong here.
- `vyre-libs` owns every Category A composition: today's Tier 3 product
  ops, today's Tier 2.5 primitive domains, and the generic compositions
  the pass engine used to park. Public, feature-gated per domain,
  maximally deduplicated. Sharing a helper becomes a `pub` change inside
  one crate, not a cross-crate migration; the two-caller criterion in
  `docs/lego-block-rule.md` gates making a helper public.
- Category B stays banned.

Optimization semantics live only in `vyre-foundation` and run on CPU.
Compile time on CPU is the default; vyre optimizes for the runtime of
user programs, not its own compile time. Executing optimizer passes as
dispatched Programs is an execution strategy for graphs too large for
CPU passes, never a second implementation of a pass: the pass semantics
are foundation's, the pass engine replays them.

`vyre-pass-engine` is narrowed to exactly that pass engine, landed
2026-08-13. It holds `src/lib.rs` and `optimizer/`: the pass pipeline,
the resident pipeline, and the `*_via_encoded` execution paths. Nine
module trees moved out, all of them to `vyre-libs`:

- `scheduling/` solvers to `vyre_libs::scheduling`.
- `analysis/` to `vyre_libs::analysis`.
- `logic/` to `vyre_libs::reasoning`.
- `data/` to `vyre_libs::encoding`.
- `math/` to `vyre_libs::solvers`.
- `graph/` to `vyre_libs::graph::dispatch`, the CPU oracle dispatcher
  with it.
- `hardware/` device-boundary contracts to `vyre_libs::device`.
- `telemetry/` call counters to `vyre_libs::telemetry`.
- The parity-test program-sequence helper to `vyre_libs::test_support`.

The name states the job, not a hardware tier: the crate executes passes,
and which device runs them is the dispatcher's answer, not the crate's.
The dispatch seam itself is `vyre_foundation::program_dispatch::ProgramDispatcher`,
and the CPU dispatcher its parity tests measure against is
`vyre_libs::graph::dispatch::cpu_oracle`.

The dependency DAG: `vyre-foundation` (IR, registry, CPU optimizer) ←
`vyre-primitives` (hardware intrinsics) ← `vyre-libs` (compositions) ←
`vyre-pass-engine` ← compiler and drivers. Foundation cannot consume the
operation crates, so `vyre_foundation::pass_substrate` owns the CPU pass
math outright. It is not a duplicate: the pass engine imports those
functions and adds dispatch around them, so one implementation keeps one
home. `pass_substrate` is now the only `*substrate*` name in the tree,
which `structure-gate` enforces.

Registry impact, landed: `OperationTier::Primitive` merged into
`OperationTier::Intrinsic`, so `vyre-primitives::` classifies as
`Intrinsic` and the operation-matrix spelling is `intrinsic`.
`OperationTier::Runtime` is gone with the five driver registrations, and
`check-tier-deps`' tier table moves with the crate fold.

## Whole-program artifact pipeline

1. Frontends and builders produce typed `Program` values.
2. `ProgramGraph` centralizes graph adaptation, typed value identity,
   constants, lifetimes, effects, and validation.
3. The megakernel compiler performs semantic optimization, explores
   legal whole-graph schedules under a recorded finite budget, and
   returns the best valid explored plan. It does not claim a
   mathematical global optimum.
4. `attach_target` invokes a registered pure target compiler. The
   shared selected-module boundary fuses each selected group and runs
   verified lowering once. Concrete target compilers consume that
   immutable product and return exact module bytes, workgroup and grid
   geometry, dynamic shared bytes, and descriptor-to-artifact binding
   projections for an explicit target profile.
5. The target payload authenticates the profile generation, selected
   node and stage identities, verified descriptor, module format and
   bytes, geometry, binding projection, and neutral artifact digest.
6. A concrete materializer admits authenticated target state without
   fusing, lowering, or re-inferring geometry and creates an
   `ArtifactInstance`.
7. Runtime builds a typed `BindingSet`, rejects zero invocation
   extents, submits work, waits for completion, and reads outputs by
   artifact value identity.
8. Recovery rematerializes authenticated artifact bytes. It does not
   lower or compile a raw `Program` during submission.

## Cross-program composition

`ProgramGraph` is the composition unit. It preserves typed IDs rather
than reconstructing values from names. Compiler requests carry external
facts and bounded search budgets, not caller-selected fusion or schedule
decisions. `vyre-lower` consumes the verified semantic product. Emitters
do not run private optimizers or accept unverified raw programs.

## Megakernel boundary

"Megakernel" names four stages with four owners. Do not collapse them.
The living matrix is [`megakernel-wiring.md`](megakernel-wiring.md).

1. Artifact compiler: `vyre-megakernel`.
2. Persistent runtime protocol: `vyre-runtime/src/megakernel/**`.
3. Backend-neutral wave policy: `vyre-driver/src/megakernel_execution.rs`
   and siblings.
4. IR pre-dispatch fusion oracles: `vyre-foundation/src/optimizer/megakernel/`.

## Optimization placement

Two layers.

- Layer 1 is semantic IR optimization in
  `vyre-foundation/src/optimizer/passes/`. It rewrites typed IR and
  benefits every backend. This is the canonical home of pass semantics.
- Layer 2 is target lowering strategy in the owning concrete driver. It
  changes instruction selection or scheduling without changing program
  semantics.

Pass execution through the dispatcher seam is not a third layer and not
a second implementation: it replays Layer 1 semantics on device for
graphs too large for CPU passes.

Shared launch planning, cache identity, and capability records live in
`vyre-driver`. Persistent queue scheduling lives in `vyre-runtime`. The
complete contract is in [`optimization/README.md`](optimization/README.md)
and [`OPTIMIZATION_ARCHITECTURE.md`](OPTIMIZATION_ARCHITECTURE.md).

## Conformance and release evidence

A backend support claim requires an operation-matrix row and conformance
proof. The release-facing sources are:

- `docs/optimization/OP_MATRIX.toml` for operation and backend support.
- `release/evidence/conformance/conformance-matrix.json` for
  per-operation proof rows.
- `release/evidence/backends/backend-matrix.json` for executable backend
  probes.
- `docs/generated/OP_SCHEMA.json` for the joined operation contract.
- `release/release-train.toml` for the active version and release train.

Generated evidence is refreshed through its owning command. Editing a
digest, count, fingerprint, or support status by hand is not a valid
architecture change.

## Historical rationale

Historical plans and snapshots (including `docs/archive/` and the git
history of deleted pre-0.7 documents) explain why earlier designs were
considered. They do not assign current ownership or support status. When
a historical file conflicts with this guide, the generated crate graph,
ownership registry, operation schema, backend evidence, and optimization
control plane take precedence.
