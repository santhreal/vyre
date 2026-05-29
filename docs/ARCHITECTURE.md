# vyre Architecture

This document is the implementation complement to [`VISION.md`](VISION.md) and [`THESIS.md`](THESIS.md). Where those describe intent, this describes the code.

## The five-tier rule (read first)

> See [`library-tiers.md`](library-tiers.md),
> [`primitives-tier.md`](primitives-tier.md), and
> [`lego-block-rule.md`](lego-block-rule.md) for the spec,
> admission criteria, and enforcement details.

Every op in vyre lives at exactly one tier. The tier determines the
crate it belongs to, its size cap, its stability contract, and its
audit requirements.

| Tier | Crate(s) | What lives here | Size cap | Stability |
| --- | --- | --- | --- | --- |
| 1 | `vyre-foundation`, `vyre-spec`, `vyre-core` | IR model, wire format, frozen contracts. No ops. |  -  | Frozen at minor versions. |
| 2 | `vyre-intrinsics` | Cat-C hardware intrinsics requiring dedicated Naga emission + dedicated interpreter handling. 9 ops at 0.6. |  -  | Frozen surface; hand-audited. |
| **2.5** | **`vyre-primitives` (feature-gated: `text`, `matching`, `math`, `nn`, `hash`, `parsing`, `graph`)** | **Shared `fn(...)->Program` primitives reused by ≥ 2 Tier-3 dialects. The LEGO substrate. One crate, per-domain feature flags  -  mirrors vyre-libs.** | **Gate 1: ≤ 4 loops AND ≤ 200 nodes** | **Per-domain feature gate; single crate semver.** |
| 3 | `vyre-libs` (monolithic today; modules under `src/`; per-domain splits require a package migration) | Domain-specific compositions over Tier 2.5 primitives. Per the LEGO-block rule, every high-level op composes registered primitives via `region::wrap_child`. | Gate 1 enforces composition (composed_fraction ≥ 60% if over raw budget) | Per-crate / per-dialect semver. |
| 4 | External community packs (extension registry) | Third-party dialect crates that reuse the public OpDef/extension interfaces and may be published independently of core Vyre crates. | Same Gate 1 contract. | Community-governed. |

- Op ID encodes tier: `vyre-intrinsics::…` (T2), `vyre-primitives::<domain>::…` (T2.5), `vyre-libs-<domain>::…` (T3), `<dialect>::…` (T4).
- Dependency direction is one-way (T4 → T3 → T2.5 → T2 → T1). CI gate `cargo xtask check-tier-deps` rejects upward dependencies.
- Gate 1 complexity budget is enforced by `cargo xtask gate1` against every registered op.
- Region chain invariant (below) is mandatory at every tier  -  it's the substrate that makes Gate 1 enforceable.

## The region chain invariant (read second)

Every op at every tier wraps its body in
`Node::Region { generator, source_region, body }`. When an op is built
by composing another registered op, the resulting Region populates
`source_region = Some(parent_ref)`. Helpers: `vyre-libs::region::wrap_anonymous` (and `vyre-intrinsics::region::wrap_anonymous` for Tier-2 ops)
(anonymous body) and `wrap_child(parent_ref, body)` (composed body).

- `transform::optimize::region_inline` has a debug-preserve mode that
  carries the chain through to readback via a side-channel
  `flat_node_index → region_path` map on the Program.
- Backends emit the region path as shader comments: WGSL via
  `// vyre-region: …`, SPIR-V via `OpLine`, photonic via tracing.
- `cargo xtask print-composition <op_id>` walks the chain and prints
  the decomposition tree from public surface down to hardware
  intrinsic leaves.

This is what prevents big Tier-3 compositions (attention blocks,
regex DFAs, whole-grammar parsers) from becoming forensically opaque.

Detailed spec: [`region-chain.md`](region-chain.md).

## GPU-Native Parsing (read third)

vyre is extending toward a fully GPU-native SIMT pipeline for *front ends*:
parsing, packed ASTs, and analysis passes that share the same `Program` IR.
**0.6.x** ships C11-oriented work under `vyre-libs` (`parsing/c/`, `compiler/`,
`c-parser` feature) rather than a separate `vyre-libs-parse-c` crate; see
`parsing-and-frontends.md` for the exact paths. Other languages follow the
same pattern: `parsing/core/` for shared machinery, `parsing/<lang>/` for
language-specific stages.

- **GPU-native compilation** embraces SIMT data-parallel methods. Heavy branching is systematically flattened to minimize warp divergence.
- **Global contention guarantees**: Operations must never rely on scalar atomics spanning all threads. Lock-free subgroup operations (`subgroup_ballot`, `subgroup_add`) must be used for token stream reductions and allocating AST nodes to avoid bottlenecking VRAM throughput.
- **Zero-fallback invariant**: The mandate is complete GPU dominance. Everything from Lexing to Dataflow and Taint analysis stays on the GPU.

Detailed spec + throughput math: [`parsing-and-frontends.md`](parsing-and-frontends.md).

## Workspace layout

```
vyre/
├── vyre-core/             Umbrella crate  -  re-exports the stable public API
├── vyre-foundation/       IR, serialization, validation, transform, optimizer
│   ├── src/ir_inner/      Expr, Node, DataType, Program, visit traits
│   ├── src/serial/        VIR0 wire format encode/decode
│   ├── src/validate/      Program validation (V### error codes)
│   ├── src/transform/     Optimization passes
│   └── src/lower.rs       Lowerable trait definitions
├── vyre-intrinsics/       Tier 2  -  frozen hardware-mapped intrinsics
│   └── hardware/          Cat-C intrinsics (one hardware instruction each:
│                          subgroup, barrier, fma, popcount, bit_reverse,
│                          inverse_sqrt  -  9-op surface)
├── vyre-primitives/       Tier 2.5  -  LEGO compositional primitives
│   ├── graph/ bitset/     CSR traverse, NodeSet / ValueSet ops
│   ├── reduce/ fixpoint/  count/min/max/sum, bitset_fixpoint driver
│   ├── label/ predicate/  Tag-family resolver + 10 canonical predicates
│   └── text/ matching/    char_class, utf8_validate, bracket_match, …
│                          (≤ 200 top-level Nodes per op; CI-enforced)
├── vyre-libs/             Tier-3  -  `src/<domain>/` (C11: `parsing/core`,
│                          `parsing/c`, `compiler` + `c-parser` feature;
│                          per-domain splits require package migration)
├── conform/               Conformance subcrates (`vyre-conform-*`) and shared evidence harness
├── vyre-frontend-c/               C11 driver crate  -  depends on `vyre-libs` w/ `c-parser`
├── vyre-test-harness/     Shared fixtures and proof harness used by conformance and external grading
├── vyre-driver/           Backend traits, registry, routing, diagnostics
│   ├── backend/           VyreBackend, Executable, Streamable
│   ├── registry/          DialectRegistry, OpDefRegistration
│   ├── routing/           Backend auto-picker
│   └── strategy/          Lowering strategy trait + capability-driven selector
├── vyre-driver-wgpu/      wgpu/WGSL backend (primary production path)
├── vyre-driver-spirv/     SPIR-V backend (Vulkan direct)
├── vyre-spec/             Frozen data contracts (AlgebraicLaw, op metadata)
├── vyre-macros/           Proc-macros for op declarations
├── vyre-reference/        Pure-Rust CPU reference interpreter (the oracle)
├── vyre-runtime/          Persistent megakernel, replay, io_uring native ingest
├── xtask/                 Release tooling, benchmarks, quick-check
├── benches/               Criterion benchmarks
├── docs/                  Architecture, memory model, targets, wire format
├── fuzz/                  Fuzzing harnesses and mutators
└── rules/                 Example rule / fixture corpora (optional)
```

## Frozen contracts

The contracts below are frozen. Changing them is a semver-major event and breaks every external frontend and backend. Scripts in `scripts/check_trait_freeze.sh` enforce their continued presence.

| Contract | Location | What it defines |
|----------|----------|-----------------|
| `VyreBackend` | `vyre-driver::backend` | The backend trait. One method per capability. |
| `ExprVisitor` | `vyre-foundation::ir::visit` | Visitor over `Expr` that lets external IR nodes ride through every pass. |
| `Lowerable` | `vyre-foundation::lower` | Bridge between `vyre` IR and a backend-specific lowered form. |
| `AlgebraicLaw` | `vyre-spec::algebraic_law` | Declarative laws (commutative, associative, idempotent, etc.). |
| `EnforceGate` | `vyre-core::dialect::enforce` | Conformance enforcement policy  -  how the conform gate blocks regressions. |
| `MutationClass` | `vyre-core::ir::mutation` | Categorizes IR-rewriting passes (structural, semantic, cosmetic). |
| `PassBoundaryClass` | `vyre-foundation::optimizer` | Declares the legal boundary a pass may cross, preserving extension and backend isolation. |

## Runtime pipeline

```
Program (vyre IR)
   │
   │  1. inline_calls  -  expand Expr::Call via DialectRegistry compose builders
   ▼
Program (inlined)
   │
   │  2. optimize  -  fixpoint scheduler over registered passes (CSE, DCE, fusion, etc.)
   ▼
Program (optimized)
   │
   │  3. verify_program_certificate  -  registry gate, rejects unregistered ops
   ▼
Program (certified)
   │
   │  4. backend.compile  -  backend-specific; wgpu lowers via naga AST
   ▼
Compiled pipeline (backend-owned)
   │
│  5. dispatch  -  GPU execution  (host reference/discovery paths stay in the oracle and diagnostics)
   ▼
Output bytes
```

Every pass after step 1 operates on inlined programs  -  no `Expr::Call` nodes remain. This is the **inlining-first** invariant that keeps backend passes simple: they see primitive Exprs (`BinOp`, `UnOp`, `Load`, `Store`, etc.) only.

## Two-layer optimization architecture

> **Vyre Law Zero:** Runtime performance is sacred. No avoidable runtime overhead, ever.

Optimizations are split into two layers with strict separation of concerns:

### Layer 1  -  IR-level passes (`vyre-foundation/src/optimizer/passes/`)

Pure mathematical rewrites that transform `Expr → Expr` in the IR.
Backend-agnostic  -  every backend benefits equally. Zero runtime cost.

| Pass | Example |
|------|---------|
| Const fold | `3 + 4 → 7` |
| Strength reduce: power-of-2 | `x / 8 → x >> 3` |
| Strength reduce: GM division | `x / 7 → mulhi(x, M) >> s` (Granlund-Montgomery) |
| Strength reduce: reciprocal | `x / 3.0 → x * 0.333` |
| FMA synthesis | `a*b + c → fma(a,b,c)` |
| Complement laws | `x & ~x → 0` |

### Layer 2  -  Backend lowering strategies (`vyre-driver/src/strategy/`)

Target-dependent emission decisions. These don't change WHAT the program
computes  -  they change HOW it's emitted for a specific chip/API.

The `LoweringStrategy` trait lets each backend register strategies that
are selected via priority-based dispatch: the pipeline picks the
highest-priority strategy whose `can_apply()` returns true for the
current backend's `BackendCapabilities`.

| Strategy | Backend | Priority |
|----------|---------|----------|
| SPIR-V `OpUMulExtended` | Vulkan | 100 (native) |
| PTX `mul.hi.u32` | CUDA | 100 (native) |
| 16-bit half-word decomp | WGSL fallback | 10 (software) |
| Dual-issue FP32/INT32 | NVIDIA Ampere+ | 50 (concurrent) |

The boundary rule: **Layer 1 changes WHAT the IR says. Layer 2 changes HOW it's emitted.** Adding a new Layer 1 pass never touches backend code. Adding a new Layer 2 strategy never touches the IR.

## The dialect registry

`DialectRegistry::global()` is a process-wide `OnceLock<FrozenIndex>`. At first access, it walks `inventory::iter::<OpDefRegistration>` to collect every op declared via `inventory::submit!`, leaks each `OpDef` into `'static` storage via `Box::leak`, and inserts the resulting `&'static OpDef` pointers into an `FxHashMap<InternedOpId, &'static OpDef>`. After init the map is immutable. Read lookups (the hot path  -  at least one per dispatch) are a single hash + one table probe, returning `Option<&'static OpDef>`  -  zero locks, zero allocation, sub-nanosecond. An earlier `RwLock<HashMap>` design was rejected because `RwLock::read` still pays an atomic fence plus a branch per call, and `Option<OpDef>` cloned the struct on every lookup. Zero runtime cost on the dispatch path is a hard invariant.

Hot-reload (TOML loader) is deliberately not supported by this struct
today. Any source patch adding it must use `ArcSwap<FrozenIndex>`:
snapshot-publish, reader sees a consistent index with one relaxed load,
never a lock.

`OpDef` carries:
- `id`  -  stable hierarchical ID like `primitive.bitwise.xor`.
- `dialect`  -  the dialect namespace.
- `category`  -  A (composition), B (mixed), C (intrinsic).
- `signature`  -  typed inputs, outputs, attributes.
- `lowerings: LoweringTable`  -  `cpu_ref`, `primary_text`, `primary_binary`, `secondary_text`, `native_module` builders.
- `laws: &'static [AlgebraicLaw]`  -  declared properties.
- `compose: Option<fn() -> Program>`  -  for composition ops, the inlinable IR body.

Historically an `OpDefRegistration` inventory shim supplied op bodies to `lookup_program` for composition ops that still lived in `vyre-core::ops::*`. Task §3 in `RELEASE.md` collapses that shim: composition ops register directly as `OpDefRegistration` with `compose: Option<fn() -> Program>` on `OpDef`. No separate OpDefRegistration registry remains.

## Four CI laws

Every commit to main must keep these scripts green:

| Law | Script | Guarantee |
|-----|--------|-----------|
| A | `scripts/check_no_closed_ir_enums.sh` | IR-family enums carry an `Opaque` escape hatch and are walked through visitor traits, not closed `match`. |
| B | `scripts/check_no_shader_assets.sh`, `check_no_string_wgsl.sh` | No `.wgsl` asset files under op/dialect trees. All shader emission via naga AST builders. |
| C | `scripts/check_capability_negotiation.sh` | Every backend op validates adapter capabilities before dispatch. |
| D | `scripts/check_unsafe_justifications.sh` | Every `unsafe` block has a `// SAFETY:` comment stating the invariant it upholds. |
| H | `scripts/check_architectural_invariants.sh` | Substrate-neutral words only in `vyre-core`; hardware words (`workgroup`, `subgroup`, `warp`, `wgsl`) appear only in backend crates. |

## Validation  -  the V-numbered errors

`validate_program` produces `ValidationError` with a stable `V###` code. Codes are append-only (migrations, not renames). Current high-impact codes:

- `V013`  -  buffer element-type rejects an operation (e.g. load with `DataType::Bytes` into a typed comparison).
- `V016`  -  unknown op ID in `Expr::Call`.
- `V017`  -  mismatched binary comparison operand types.
- `V018`  -  barrier ordering rules violated.
- `V019`  -  shared memory access outside a workgroup region.

The full table lives in `vyre-core::ir::validate::V_CODES`. Every variant includes `Fix:`-prefixed remediation prose in the error message and a structured variant for machine readers.

## Wire format (VIR0)

- Magic: `VIR0` (4 bytes).
- Version: 1 (u8). Bumps require a migration step in `vyre-core::dialect::migration`.
- Encoded data: `Program` metadata header + `Node`/`Expr` tree with one-byte discriminant tags.
- Extension extensibility: `Expr::Opaque`/`Node::Opaque` encode as `(extension_id: u32, bytes: &[u8])`. The decoder uses the extension-id inventory to resolve the payload. Unknown IDs return a typed error whose message names the ID so the consumer can install an extension crate and re-decode.

Full spec: [`wire-format.md`](wire-format.md).

## Memory model

Detailed in [`memory-model.md`](memory-model.md). One-line summary: **invocation-local → shared → global**, with explicit `Node::Barrier` for cross-invocation ordering inside a workgroup and `MemoryOrdering` on atomics for cross-workgroup ordering.

## Targets

Detailed in [`targets.md`](targets.md). Current dispatch-capable path:
`wgpu` (Vulkan / DX12 / Metal / WebGPU via naga). Other backend crates
must state whether they are dispatch-capable, emission-only, or
registry-contract targets.
