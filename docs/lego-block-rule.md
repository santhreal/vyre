# LEGO-block rule: composition is the architecture

**Status: Active.** This is the workspace-wide composition policy every
author and agent follows before adding a new sub-op. `docs/ARCHITECTURE.md`
owns workspace boundaries; this doc owns the reuse rule, the promotion
criteria, and the enforcement loop. The generated operation schema
(`docs/generated/OP_SCHEMA.json`) is the operation inventory this rule
operates over.

## The rule

> **Before inventing a new sub-op, scan the domain folders under
> `vyre-primitives/src/` (one cargo feature per domain: `text`,
> `matching`, `math`, `nn`, `hash`, `parsing`, `graph`, `bitset`,
> `reduce`, `fixpoint`, and the rest) and the domain folders under
> `vyre-libs/src/` for an existing primitive that does the work.
> Only invent a new sub-op when (a) nothing existing maps AND (b) the
> new sub-op will be reused by 2+ callers.**

The Gate 1 complexity budget is enforced by *reuse*, not by bespoke
splitting. When attention has 8 loops, the answer is not 4 new
attention-private sub-ops; it is "composes existing matmul + softmax +
layer_norm primitives." When blake3_compress has 601 nodes, the answer
is not "split into 4 blake3-only chunks"; it is "extract the G mixing
function and the round permutation as `vyre_primitives::hash::blake3_g`
/ `blake3_round`, reused by future `blake3_keyed`, `blake3_xof`, and
`blake3_tree_hash`."

## Operation categories

Every operation is one of two categories. There is no third.

- **Category A: composition.** A backend-neutral `fn(...) -> Program`
  built from lower-tier operations over existing `Expr`/`Node`
  variants. It adds no concrete target lowering. Regions preserve
  composition provenance: the outer region keeps the Cat-A generator,
  each child keeps its primitive generator and a `source_region`
  naming the parent. Tier 2.5 primitives and Tier 3 library ops are
  all Category A.
- **Category C: hardware intrinsic.** An operation that requires a
  dedicated hardware contract: a dedicated emitter arm AND a dedicated
  reference-interpreter eval arm. Its registration supplies the
  neutral builder and deterministic fixture contract; each supported
  target supplies a keyed lowering facet. Missing facets fail closed.
  Category C lives in `vyre-intrinsics` and is intentionally small.
- **Category B is banned.** Category B was runtime interpretation of a
  general operation bytecode, on the host or inside a persistent
  kernel. Vyre does not do this. A program remains typed IR until
  verified lowering; raw `Program` execution exists only in explicitly
  named reference, parity, and conformance oracle seams. An op that
  "needs an interpreter" is an un-decomposed Category A op; decompose
  it.

Placement test: if you can write it as `fn(...) -> Program` using only
existing IR variants, it is Category A and belongs in `vyre-primitives`
(shared) or `vyre-libs` (domain-facing). If it needs a new `Expr`
variant, that is a Tier-1 change in `vyre-foundation`; if the variant
exists to reach hardware, the op is Category C in `vyre-intrinsics`.

## Why

Vyre's product claim is "compose perfect primitives, beat monolithic
kernels." That claim fails the moment a dialect crate reinvents a
primitive locally: the new caller doesn't benefit from the existing
one's hardening, the optimizer can't fuse across the boundary, and
the LEGO substrate fragments. Every primitive that lives in only one
op's source file is wasted leverage.

## Discovery checklist

Before writing a new sub-op:

1. **Search by name.** `rg -i 'fn <verb>' vyre-libs/src vyre-primitives/src`.
   If the work has a name (matmul, scan, hash, dfa_step), someone has
   probably written it.
2. **Search by op id.** `cargo_full run --bin xtask -- print-composition
   vyre-libs::...` walks a registered op's region tree. If a target
   Region's `generator` reads like the work you're about to do, that's
   the primitive.
3. **Search by region chain.** Pick a sibling op (same domain, similar
   shape) and print its composition. The chain shows what primitives
   that sibling already composes; chances are 1+ apply to your op too.
4. **Ask Gate 1.** `cargo_full run --bin xtask -- gate1` reports
   per-op `composed_fraction`. A sibling with a high composed_fraction
   is the playbook to follow.

## Promotion criteria (single-caller → primitive)

A `fn(...) -> Program` graduates from "private to one dialect" to
"public Tier 2.5 primitive" when ALL THREE conditions hold:

1. **Reusability.** ≥ 2 Tier-3 dialects (or one Tier-3 dialect +
   `xtask` / conform harness / an actual community pack) consume it.
2. **Stability.** The primitive's API has settled: argument list is
   small, named, no caller is asking for breaking changes.
3. **No domain glue.** The primitive does ONE concern. `matmul` does
   matmul, not "matmul plus a softmax for transformers." Domain
   compositions glue primitives together; the primitive itself is
   single-purpose (LAW 7).

If only ONE caller has a private helper, leave it alone: premature
promotion creates churn for no gain. `lego-audit` primitive-coverage
reports orphan primitives as adoption advisories; synthetic catalog
consumers registered to fake a second caller are hard failures.

## Cross-crate promotion patch contract

A primitive promotion across crate or tier boundaries changes the public
dependency graph, so one patch must update all of: the crate graph
(`docs/CRATE_OWNERSHIP.toml` + regenerated `docs/CRATE_GRAPH.md`), this
tier rule, and an import-path migration test. The import-path migration
test names the old path, the canonical owner path, or the compatibility
shim that keeps downstream code working.

`check-tier-deps` owns dependency-direction validation and verifies this
contract text is present. `lego-audit` owns composition and cross-dialect
import validation. A promotion that satisfies only one gate is not owned.

## Gate 1 and lego-audit

Gate 1 (`cargo_full run --bin xtask -- gate1`) walks every registered
op's region tree:

1. Count nodes + loops in its expanded body.
2. Under raw budget (loops ≤ 4 AND nodes ≤ 200): pass.
3. Over budget: compute `composed_fraction = nodes inside child
   regions with a registered source_region / total nodes`.
   `composed_fraction ≥ 0.6` means the op composes primitives: pass.
4. Otherwise fail with a diagnostic listing which inline sub-blocks
   would have been composeable as Tier 2.5 primitives.

Wrapping a `Node::Region` around inlined code does not game this; the
body has to call into registered primitive ops.

`cargo_full run --bin xtask -- lego-audit` is the stricter pass: IR
fingerprint no-reinvention (>80% overlap without an invocation edge is
flagged), depth-of-composition, primitive coverage, and cross-dialect
reach-through. `dedup-report` emits the machine-readable duplicate
family schema.

## Before / after example: `attention`

### Before (Gate 1 fails, 8 loops, composed_fraction=0%)

```rust
pub fn attention(q: &str, k: &str, v: &str, out: &str, d: u32, s: u32) -> Program {
    // bespoke 8-loop body that inlines:
    //   - q @ k^T computation         (would be `matmul`)
    //   - per-row max for stable softmax  (would be `reduce_max`)
    //   - per-row exp/sum/divide      (would be `softmax_step`)
    //   - score @ v                   (would be `matmul`)
    //   - residual norm               (would be `layer_norm`)
}
```

### After (Gate 1 passes via composition, 0 inline loops)

```rust
pub fn attention(q: &str, k: &str, v: &str, out: &str, d: u32, s: u32) -> Program {
    let scratch_scores = "scores_scratch";
    let scratch_norm = "norm_scratch";
    Program::wrapped(
        vec![/* ... declarations including scratches ... */],
        [s, 1, 1],
        vec![
            region::wrap_child(
                "vyre-libs::nn::attention",
                /* parent generator-ref */,
                vec![
                    // every step is a wrap_child INTO a registered primitive
                    region::wrap_child("vyre-primitives::math::matmul", /* ref */,
                        vec![/* call into matmul body */]),
                    region::wrap_child("vyre-primitives::nn::softmax_step", /* ref */,
                        vec![/* call into softmax_step body */]),
                    region::wrap_child("vyre-primitives::math::matmul", /* ref */,
                        vec![/* second matmul */]),
                    region::wrap_child("vyre-primitives::nn::layer_norm_step", /* ref */,
                        vec![/* residual norm */]),
                ],
            ),
        ],
    )
}
```

Gate 1 passes: 4 child regions, each a registered Tier 2.5 primitive,
composed_fraction=100%. The optimizer can still inline+fuse across
boundaries; the composition chain stays visible to
`print-composition` for audit.

The win: a future `linear_attention`, `multi_query_attention`, or
`flash_attention_v2` reuses the same matmul + softmax_step +
layer_norm_step primitives. No code duplication. No drift.

## Anti-patterns the rule rejects

- **Inline helpers that should be primitives.** If you're writing a
  `fn(...)->Program` local to one op file, run the discovery checklist
  first. If a primitive exists, use it.
- **Cross-dialect reach-around.** One domain module in `vyre-libs`
  importing private items from a sibling domain. Lift to Tier 2.5
  (`vyre_primitives::matching::dfa_step`) instead.
- **Bespoke split to satisfy Gate 1.** Splitting `attention` into
  `attention_part_a` / `attention_part_b` private helpers passes the
  loop count but fails the LEGO test: there is no reuse, just visual
  surgery. Gate 1 enforcement detects this via composed_fraction.
- **Premature promotion.** Lifting a single-caller helper into
  Tier 2.5 before a second consumer materializes. Wait for the second
  caller; promote when ≥ 2 want it.
- **Category B by another name.** A "dispatch table," "bytecode," or
  "interpreter loop" over op codes is banned interpretation. Lower to
  typed IR instead.

## Workflow when the primitive doesn't exist yet

1. Write the primitive directly in the appropriate
   `vyre-primitives/src/<domain>/<primitive>.rs` (NOT inside the
   consuming dialect), behind that domain's cargo feature.
2. Add an `inventory::submit!(OperationRegistration { .. })`
   registration so the universal harness picks it up.
3. Add `test_inputs` + `expected_output` (use the trace tooling for
   f32 ops).
4. Record the promoting consumers in the commit body (the discovery
   note below covers the review trail).
5. Wire the high-level op to call into it via
   `region::wrap_child(<primitive_op_id>, ...)`.
6. Run `cargo_full run --bin xtask -- gate1`; the op should now pass
   via composed_fraction.

## Enforcement

- `cargo_full run --bin xtask -- gate1` runs in CI and fails on
  budget violations.
- `conform/vyre-conform/tests/contract_cases/composition_discipline__every_op_is_under_complexity_budget.rs::no_op_reinvents_another_registered_op`
  scans for op bodies whose IR fingerprint matches an
  already-registered op but isn't dispatching through it.
- `check-tier-deps` blocks upward path dependencies and unowned tier
  movement; `lego-audit` blocks cross-tier imports that bypass the
  canonical primitive or facade owner.
- Code review: every change touching `vyre-libs/src/<domain>/` or
  `vyre-primitives/src/<domain>/` that adds a new `fn(...)->Program`
  includes a one-line discovery-checklist note in the commit body
  confirming nothing existing applied.

## Before / after example: `visual` domain (Molten)

This is a real decomposition that occurred when adding GPU-accelerated
visual effects (blur, shadow, filter chains) for the Molten web engine.
It illustrates the full discovery process and how domain-level thinking
distills into existing primitives.

### Attempt 1: new Tier 2.5 domain `visual/` with 6 primitives

Initial plan created an entire `vyre-primitives/src/visual/` domain:

```
vyre-primitives/src/visual/
├── blur.rs            # two-pass Gaussian blur
├── shadow.rs          # box shadow with SDF falloff
├── filter_chain.rs    # brightness/contrast/saturate/invert
├── composite.rs       # Porter-Duff alpha over
├── gradient.rs        # CSS gradient rasterization
└── downsample.rs      # 2× box-filter downsample
```

Each file was a monolithic `fn(...) -> Program`: ~200 lines of inlined
IR with custom pixel unpacking, color math, and kernel loops. Gate 1
would have rejected every one of them on composed_fraction = 0%.

### Attempt 2: decompose into real primitives

Applied the discovery checklist. Proposed Tier 2.5 primitives:

- `visual::separable_conv`: 1D convolution along an axis
- `visual::pixel_pack`: RGBA u32 ↔ separate channels
- `visual::color_lerp`: interpolate between two colors
- `visual::sdf_rounded_rect`: signed distance to a rounded rect

### Attempt 3: dissolve into existing domains (correct answer)

Applied the checklist *again*. Each "visual primitive" collapsed into
something that already exists or belongs in `math/`:

| Proposed visual primitive | Actual home | Why |
|---|---|---|
| `separable_conv` | **`math::conv1d`** | 1D convolution is domain-neutral: used by signal processing, audio, NLP, and image processing. Not visual-specific. |
| `pixel_pack/unpack` | **Already Tier 1 IR** | `Expr::bitand`, `Expr::shr`, `Expr::shl`, `Expr::bitor` do this directly. No primitive needed. |
| `color_lerp` | **Already Tier 1 IR** | `lerp(a, b, t) = a + (b - a) * t`: three Expr ops. A color is just a value. |
| `sdf_rounded_rect` | **Private in Tier 3** | Only one consumer (`box_shadow`). By the promotion rule, stays inline until a second caller appears. |

**Result:** the `visual/` Tier 2.5 domain was deleted entirely. The only
new Tier 2.5 primitive is `math::conv1d`, a 1D separable convolution
that blur, signal processing, and future audio ops all compose from.

The domain-specific compositions (`blur`, `box_shadow`, `filter_chain`,
`glass`) live in `vyre-libs/src/visual/` as Tier 3 compositions over:

- `math::conv1d` (blur kernel)
- Existing IR expressions (pixel bit manipulation, lerp, clamp)
- Private helpers where only one caller exists (SDF)

### The lesson

**Domain thinking creates domain primitives. LEGO thinking dissolves
them into math.** When the discovery checklist says "nothing existing
maps," run it again: you're probably looking at the wrong abstraction
level. A "color interpolation" is marketing language for a multiply-add.
A "pixel unpack" is marketing language for a bit shift. The LEGO rule
forces you to see through the domain framing to the underlying
operation, which is almost always already in `math/`, `text/`, or the
IR itself.
