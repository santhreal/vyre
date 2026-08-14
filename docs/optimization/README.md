# Vyre optimization control plane

Applies to Vyre 0.7.2.

This directory defines the maintained architecture, placement, proof, and
benchmark contracts for optimization work.

Start with [`START_HERE.md`](START_HERE.md). The generated
[`PASSES.md`](PASSES.md) lists every source-registered semantic pass and
supplemental rule. [`LEGACY_DOCS.md`](LEGACY_DOCS.md) classifies retained
historical material.

## Precedence

Use these authorities in order:

1. `docs/optimization/README.md` defines the two-layer contract and patch proof.
2. `docs/optimization/PASSES.md` projects the live semantic pass registry.
3. `docs/optimization/OWNERSHIP.toml` defines write ownership.
4. `docs/optimization/TAXONOMY.md` defines accepted optimization classes.
5. `docs/optimization/OP_MATRIX.toml` defines operation and backend coverage.
6. `docs/optimization/BENCH_TARGETS.toml` defines benchmark targets and baseline
   classes.

Correct a conflicting document and its generated projections in the same
change.

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
| Primitive reusable ops | `vyre-primitives/src/` | Must meet tier rules and matrix entry. |
| Benchmark harness and targets | `vyre-bench/` plus `docs/optimization/BENCH_TARGETS.toml` | Targets must identify baseline class. |


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

- hardware one-instruction intrinsics: `vyre-primitives/src/hardware/`
- reusable substrate primitives: `vyre-primitives/src/<family>/`
- domain compositions: `vyre-libs/src/<family>/`
- IR variants and validation: `vyre-foundation`
- backend lowering: owning driver crate only
- runtime scheduling/protocol: `vyre-runtime`

## Composition admission checks

Run both structural checks before introducing an operation or extracting a
reusable primitive:

```bash
./cargo_full run --bin xtask -- whats-similar --all
./cargo_full run --bin xtask -- lego-audit --with-repo
```

`whats-similar` compares registered programs by IR shape. Operations routed
through one canonical builder belong to one implementation family, including
the atomic reduction family. Similar members of that family are evidence of
centralization, not duplicate implementations.


`lego-audit` treats registered child regions as composition evidence. A Tier 3
operation with at least 20 nodes must place at least 25% of those nodes under a
registered child, unless the checker classifies it as a reviewed pure-IR leaf.
The two-caller primitive promotion rule is an adoption advisory. It counts real
operation-to-primitive edges only. Synthetic catalog wrappers, generated
aliases, and fixture consumers are hard failures and never satisfy coverage.

The trend check reads `audits/lego-composition.tsv` from the previous tag. If
that tag predates the baseline, it uses the checked-in bootstrap baseline.
Create or refresh the bootstrap after reviewing composition changes:

```bash
./cargo_full run --bin xtask -- lego-audit --write-baseline
```

## Benchmark doctrine

Vyre benchmarks measure active backend execution separately from wall time.
Both are recorded. Performance contracts use active device time when the
backend exposes it, and wall time when it does not.

CPU baselines must identify the best known available Rust/native crate or
library class, not a naive loop. GPU competitor baselines are added when a
credible public implementation is available.

The target table lives in `BENCH_TARGETS.toml`; individual benchmark files must
not carry private target logic that disagrees with it.

## Measuring launch geometry

Read the planned geometry back at runtime. Never cite the workgroup constant
in the source as evidence of what ran:

```text
declared in the builder : [256, 1, 1]
actually launched       : [1024, 1, 1] x 170
```

The planner re-plans geometry, so a declared workgroup size is an input to
that decision and not a record of it. Any occupancy, residency, or width
claim derived from reading a constant is unfounded, including one about a
kernel whose declared width looks safe.

Two consequences follow. A width-specific measurement must read the planned
geometry back from the launch. And the cooperative residency ceiling (1020
blocks at width 256 against 170 at width 1024 on a 170 SM part) binds only
where a GridSync barrier exists, so you must not apply it to an ordinary
dispatch.

On a part with 1536 threads per SM, 1024 is the only common width at or below
1024 that does not divide that figure, so it alone truncates to one block per
SM. Widths 512 and 768 both reach full residency. A 1024 wide cooperative
kernel therefore has a free 33 percent of thread residency available to it,
which is a recovery rather than a tradeoff.

## Workgroup scratch is a launch ceiling

Size workgroup scratch against the static shared memory limit, not against
what the device reports as available. A fixed lane count multiplied by a
runtime extent grows without bound:

```text
vyre_libs::nn::attention::mla::mla_decode, 64 lanes fixed
  q_scratch  = 64 * head_dim  f32
  score_tile = 64 * 64        f32
  o_acc      = 64 * head_dim  f32
  bytes      = (128 * head_dim + 4096) * 4

head_dim =  32 -> 32 KiB, loads and matches the CPU reference
head_dim =  64 -> 48 KiB, exactly the cap, loads and matches
head_dim =  96 -> 64 KiB, refused before load
head_dim = 128 -> 80 KiB, refused before load
```

Crossing the limit refuses rather than corrupting, so you never receive a
wrong answer from this path. The refusal names the measured bytes, the cap,
and the buffers that produced the figure:

```text
CUDA workgroup scratch for this program is 81920 bytes, over the device
per-workgroup static shared memory limit of 49152 bytes. Contributing
buffers: `q_scratch` 32768 bytes, `score_tile` 16384 bytes, `o_acc` 32768
bytes. Fix: reduce the workgroup buffer element counts, narrow the workgroup
width they are sized against, or move the scratch to a storage buffer.
```

That check runs at dispatch in `vyre-driver-cuda`, before PTX emit. It is a
pre-check and not a translation of the load error. A genuine ISA problem
still reports `CUDA_ERROR_INVALID_PTX`, which stays correct and useful for
the cases it actually describes.

The limit is `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK`, the static
allocation ceiling. Do not size against the per-SM figure: a part reporting
128 KiB per SM still refuses a 64 KiB static request from one workgroup.

The boundary is the scratch size and not the module size. head_dim 96 emits a
smaller PTX module than head_dim 128 and is refused just the same, while
head_dim 64 lands exactly on the cap and loads, so there is no off-by-one in
the admission check.

MLA softmax also needs a positive `ulp_budget`. Under `DispatchConfig::default()`
the PTX emitter refuses to lower `Exp` at all, with a message naming the unset
budget and the fix. That refusal is deliberate: choosing an accuracy budget on
your behalf would be a silent numerical downgrade. Set one before you conclude
anything about scratch.

`vyre-driver-cuda/tests/mla_decode_shared_memory_scaling.rs` holds these
measurements.

## Boundary enforcement

Required structural checks:

- No concrete backend names or API types in shared crates except neutral target
  identifiers explicitly owned by `vyre-driver`.
- No duplicated optimizer logic inside drivers.
- No op support claim without matrix coverage and tests.
- No benchmark target without a baseline row.

