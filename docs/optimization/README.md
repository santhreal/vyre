# Vyre optimization control plane

This directory is the canonical coordination surface for performance work in
Vyre. Older optimization plans, audits, release notes, and internals are
evidence. They are not authoritative unless this directory links them for a
specific workstream.

The goal is swarm-safe optimization: thousands of agents can improve the
codebase without duplicating work, moving code into the wrong layer, or
shipping superficial patches.

Start with [`START_HERE.md`](START_HERE.md). Use
[`LEGACY_DOCS.md`](LEGACY_DOCS.md) when you find an older plan or audit.
The executable backlog lives in [`ROADMAP.md`](ROADMAP.md), and active work
claims live in [`CLAIMS.toml`](CLAIMS.toml).

## Precedence

For optimization, performance, backend consolidation, and op-placement work:

1. `docs/optimization/README.md` defines the control plane.
2. `docs/optimization/OWNERSHIP.toml` defines write ownership and swarm lanes.
3. `docs/optimization/AGENT_CONTRACT.md` defines what a patch must prove.
4. `docs/optimization/TAXONOMY.md` defines the accepted optimization classes.
5. `docs/optimization/ROADMAP.md` defines the executable backlog.
6. `docs/optimization/OP_MATRIX.toml` defines op/backend coverage tracking.
7. `docs/optimization/BENCH_TARGETS.toml` defines benchmark targets and baseline classes.
8. Other docs and audits are reference material unless one of the files above
   delegates a task to them.

When another document conflicts with this directory, this directory wins.
Update the lower-precedence document with a supersession note instead of
creating a second plan.

## Non-negotiable architecture

Vyre has two optimization layers.

Layer 1 is IR-pure optimization. It changes `Expr`, `Node`, `Program`, or
optimizer facts while preserving semantics for every backend. It lives in
`vyre-foundation/src/optimizer/` and adjacent foundation analysis modules.
Examples: Granlund-Montgomery constant division, Lemire-style constant
remainder, exact-division simplification, shift-add decomposition, FMA
synthesis, loop unroll, vectorization, canonicalization, fusion, shared use
facts, and compile-time O(n^2) removal.

Layer 2 is backend lowering strategy. It does not change the IR contract; it
changes how a concrete backend emits or schedules hardware instructions. It
lives only inside the owning driver crate. Examples: tensor-core lowering,
native multiply-high selection, PTX scheduling, WGSL/naga emission details,
SPIR-V layout details, CUDA stream/event handling, and backend-specific module
caches.

Shared crates may define neutral traits, facts, cache keys, launch plans, and
capability records. Shared crates must not contain concrete backend API names,
shader dialect strings, device object types, or compatibility shims for a
single backend.

## Where work belongs

| Work kind | Canonical home | Notes |
|---|---|---|
| Algebraic rewrite valid for every backend | `vyre-foundation/src/optimizer/passes/` | Backend must never reimplement the same rewrite. |
| Program facts and optimizer cost model | `vyre-foundation/src/optimizer/` | One shared fact graph, invalidated deliberately. |
| Wire/fingerprint canonicalization | `vyre-foundation/src/serial/` and `vyre-foundation/src/ir_inner/` | Cache/security-critical; use canonical bytes. |
| Backend-neutral launch/binding/cache policy | `vyre-driver/src/` | No concrete backend imports or string-specific behavior. |
| Concrete codegen or device API | `vyre-driver-cuda`, `vyre-driver-wgpu`, `vyre-driver-spirv` | Only irreducible substrate glue stays here. |
| Persistent megakernel scheduling/protocol | `vyre-runtime/src/megakernel/` | Primary runtime path; do not duplicate in drivers. |
| Domain ops and libraries | `vyre-libs/src/` | Compose lower tiers; no driver logic. |
| Primitive reusable ops | `vyre-primitives/src/` or `vyre-intrinsics/src/` | Must meet tier rules and matrix entry. |
| Benchmark harness and targets | `vyre-bench/` plus `docs/optimization/BENCH_TARGETS.toml` | Targets must identify baseline class. |

## Swarm lanes

Agents claim one lane from `OWNERSHIP.toml` and stay inside its write set unless
the main integrator explicitly expands scope.

Required lanes:

- `coordination`: canonical roadmap, claims, lane boundaries, and enforcement scripts.
- `foundation_optimizer`: IR rewrites, fact graph, canonicalization, pass timing.
- `foundation_wire`: canonical bytes, fingerprints, serialization allocation behavior.
- `driver_shared`: backend-neutral launch, binding, validation, cache, residency.
- `driver_cuda`: PTX lowering, CUDA residency, streams, events, module cache.
- `driver_wgpu`: naga/WGSL lowering, wgpu buffer/readback/pipeline behavior.
- `driver_spirv`: SPIR-V lowering and experimental parity boundaries.
- `runtime_megakernel`: persistent runtime queue, scheduler, IO, resident protocol.
- `bench_harness`: measurement API, baselines, reporting, regression math.
- `op_matrix`: op coverage, tier placement, parity tracking.

Do not create new lanes casually. If a task does not fit, update
`OWNERSHIP.toml` first.

## Required proof for an optimization patch

Every optimization patch must include all applicable proof:

- Placement proof: state Layer 1 or Layer 2 and why.
- Correctness proof: unit/property/conformance test or exact invariant.
- Performance proof: benchmark, reduced allocation count, asymptotic bound, or
  emitted IR/code shape assertion.
- Integration proof: command output from the relevant crate tests/checks.
- Matrix update: `OP_MATRIX.toml` when op/backend coverage changes.
- Target update: `BENCH_TARGETS.toml` when benchmark targets or baselines change.

Patches that only rename, remove comments, weaken tests, or document a gap are
not optimization patches.

## Op-specific organization

Each op family must have exactly one owner row in `OP_MATRIX.toml`. Backend
support is recorded there, not in scattered prose. If an op is implemented in
one backend but not another, the row must say whether that is experimental,
release-blocking, or intentionally outside the backend's scope.

Op-specific files belong by tier:

- hardware one-instruction intrinsics: `vyre-intrinsics/src/hardware/`
- reusable substrate primitives: `vyre-primitives/src/<family>/`
- domain compositions: `vyre-libs/src/<family>/`
- IR variants and validation: `vyre-foundation`
- backend lowering: owning driver crate only
- runtime scheduling/protocol: `vyre-runtime`

## Benchmark doctrine

Vyre benchmarks measure active backend execution separately from wall time.
Both are recorded. Performance contracts use active device time when the
backend exposes it, and wall time when it does not.

CPU baselines must identify the best known available Rust/native crate or
library class, not a naive loop. GPU competitor baselines are added when a
credible public implementation is available.

The target table lives in `BENCH_TARGETS.toml`; individual benchmark files must
not carry private target logic that disagrees with it.

## Boundary enforcement

Required structural checks:

- No concrete backend names or API types in shared crates except neutral target
  identifiers explicitly owned by `vyre-driver`.
- No duplicated optimizer logic inside drivers.
- No op support claim without matrix coverage and tests.
- No benchmark target without a baseline row.
- No new optimization plan outside this directory unless it has a supersession
  header pointing back here.

When these checks are not yet automated, the patch must include the grep/script
used as proof and a follow-up enforcement issue in this directory.
