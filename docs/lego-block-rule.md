# LEGO-block rule: composition is the architecture

The workspace composition policy: the discovery step, the Gate 1 budget, the
promotion criteria, and Category A/C placement. Read it before adding or
restructuring an operation. `docs/ARCHITECTURE.md` owns workspace boundaries
and the crate structure. `docs/generated/OP_SCHEMA.json` is the operation
inventory this rule operates over.

## The rule

Before inventing a new sub-op, scan the domain folders under `vyre-libs/src/`
for a primitive that already does the work. Invent a new sub-op only when
nothing existing maps and the new sub-op will be reused by two or more callers.

The Gate 1 budget is met by reuse, not by bespoke splitting. When
attention has eight loops, the answer is not four attention-private
sub-ops; it is composing the existing matmul, softmax and layer-norm
primitives. When a hash compression body reaches 601 nodes, the answer is
not four hash-only chunks; it is extracting the mixing function and the
round permutation as primitives that a keyed, extendable-output or tree
variant composes next.

## Operation categories

Every operation is one of two categories. There is no third.

- **Category A: composition.** A backend-neutral `fn(...) -> Program`
  built from lower-level operations over existing `Expr` and `Node`
  variants. It adds no concrete target lowering. Regions preserve
  composition provenance: the outer region keeps the Category A
  generator, and each child keeps its primitive generator plus a
  `source_region` naming the parent. Every Category A op lives in
  `vyre-libs`.
- **Category C: hardware intrinsic.** An operation that requires a
  dedicated hardware contract: its own emitter arm in every backend and
  its own arm in the reference interpreter. Its registration supplies the
  neutral builder and the deterministic fixture contract; each supported
  target supplies a keyed lowering facet, and a missing facet fails
  closed. Every Category C op lives in `vyre-primitives`, which stays
  small for that reason.
- **Category B is banned.** Category B is runtime interpretation of a
  general operation bytecode, on the host or inside a persistent kernel.
  A program stays typed IR until verified lowering; raw `Program`
  execution exists only in the named reference, parity and conformance
  oracle seams. An op that needs an interpreter is an un-decomposed
  Category A op. Decompose it.

Placement test: if it can be written as `fn(...) -> Program` over
existing IR variants, it is Category A and belongs in `vyre-libs`. If it
needs a new `Expr` variant, that variant is a change in
`vyre-foundation`; if the variant exists to reach hardware, the op is
Category C in `vyre-primitives`.

## Why

The product claim is that composed primitives beat monolithic kernels.
The claim fails the moment a dialect reinvents a primitive locally: the
new caller does not inherit the existing one's hardening, the optimizer
cannot fuse across the boundary, and the substrate fragments. A primitive
that lives in one op's source file is reuse nobody else gets.

## Discovery

Before writing a new sub-op:

1. **Search by name.** `git grep -n 'fn <verb>' -- vyre-libs/src`. If the work
   has a name, someone has probably written it.
2. **Search by op id.** `./cargo_full run --bin xtask -- print-composition
   --op-id <id>` walks a registered op's region tree. If a region's
   generator reads like the work you are about to do, that is the
   primitive.
3. **Search by region chain.** Print the composition of a sibling op in
   the same domain. The chain names the primitives that sibling already
   composes, and usually one of them applies.
4. **Ask Gate 1.** `./cargo_full run --bin xtask -- gate1` reports each
   op's composed fraction. A sibling with a high composed fraction is the
   playbook to follow.

## Promotion criteria

A `fn(...) -> Program` graduates from private to one op file to published
`vyre-libs` op when all three hold:

1. **Reuse.** Two or more callers consume it: dialect ops, xtask or
   conform tooling, or a community pack.
2. **Stability.** The argument list is small and named, and no caller is
   asking for a breaking change.
3. **One concern.** `matmul` does matmul, not matmul plus a softmax for
   transformers. Domain compositions glue primitives together; the
   primitive itself is single-purpose.

Promotion inside `vyre-libs` is a visibility change plus a registration,
not a move. With one caller, leave the helper private: publishing early
buys churn and no reuse. `lego-audit` reports an orphan published op as
an adoption advisory, and a catalog consumer registered to fake a second
caller is a hard failure.

## Gate 1 and lego-audit

`./cargo_full run --bin xtask -- gate1` walks every registered op's region
tree and passes an op when either half holds:

1. Under raw budget: four loops or fewer and 200 nodes or fewer.
2. Composed: nodes inside a region whose generator is a registered op id
   are at least 60 percent of total nodes.

Wrapping a region around inlined code does not satisfy the second half. A
region that names no operation carries one of the two prefixes in
`vyre_foundation::composition::ANONYMOUS_GENERATOR_PREFIXES`: `inline::`,
minted when the composer reparents a body onto its caller, and
`anonymous::`, written by a builder that needs a named boundary inside one
operation. Such a region still carries a `source_region` naming its own op,
so a `source_region` alone says nothing about composition; the generator
has to name another registered op. On failure the diagnostic lists the
inline sub-blocks that should have been primitive calls.

`gate1` owns the budget. `abstraction-gate` reads the same walk for the
boundary question: whether every child region names a building block that
is registered, and whether every cited parent is an op that exists.

`./cargo_full run --bin xtask -- lego-audit` is the stricter pass: IR
fingerprint no-reinvention, depth of composition, primitive coverage, and
cross-dialect reach-through. `./cargo_full run --bin xtask -- dup-scan`
reports duplicated line counts per crate against their pins.

## Cross-crate promotion patch contract

Promoting a primitive across a crate boundary changes the published
dependency graph, so one patch updates all of: `docs/CRATE_OWNERSHIP.toml`
with the regenerated `docs/CRATE_GRAPH.md`, this document, and an
import-path migration test. The import-path migration test names the old
path and the owner path, and it migrates every caller in the same change:
this workspace ships no compatibility shim for a moved item.

`check-tier-deps` owns dependency-direction validation and checks that
this contract is present here. `lego-audit` owns composition and
cross-dialect import validation. A promotion that satisfies one gate and
not the other is not owned.

## What the rule rejects

- **An inline helper that should be a primitive.** A local
  `fn(...) -> Program` inside one op file, written without running
  discovery first.
- **A cross-domain reach-around.** One `vyre-libs` domain importing
  private items from a sibling domain. Publish the shared helper under
  the promotion criteria instead.
- **A bespoke split to satisfy Gate 1.** Splitting an op into `part_a`
  and `part_b` private helpers lowers the loop count and adds no reuse.
  The composed fraction detects it.
- **Premature publication.** Publishing a single-caller helper before the
  second caller exists.
- **Category B under another name.** A dispatch table, a bytecode or an
  interpreter loop over op codes. Lower to typed IR instead.

## When the primitive does not exist yet

1. Write the primitive in its own file under the owning domain
   (`vyre-libs/src/<domain>/<primitive>.rs`, not inside the consuming
   op's file), behind that domain's cargo feature.
2. Submit one `OperationRegistration` so the universal harness discovers
   it.
3. Give it `test_inputs` and `expected_output`.
4. Wrap the consuming op's call in a child region naming the primitive
   op id, so the composition stays visible to `print-composition`.
5. Run `gate1`. The op passes on its composed fraction.

## Enforcement

- `gate1` runs in CI and fails on a budget violation.
- `conform/vyre-conform/tests/contract_cases/composition_discipline__every_op_is_under_complexity_budget.rs`
  holds `no_op_reinvents_another_registered_op`, which scans for an op
  body whose IR fingerprint matches a registered op it does not dispatch
  through.
- `check-tier-deps` blocks an upward dependency and unowned layer
  movement; `lego-audit` blocks a cross-tier import that bypasses the
  canonical owner.

## Worked example: the visual domain

Adding GPU visual effects (blur, shadow, filter chains) started as a
`visual/` primitive domain with six primitives: two-pass Gaussian blur,
box shadow with signed-distance falloff, a brightness and contrast filter
chain, Porter-Duff alpha compositing, gradient rasterization, and box
downsampling. Each was one monolithic `fn(...) -> Program` of inlined IR
with its own pixel unpacking, color math and kernel loops. Gate 1 would
have rejected every one of them at a composed fraction of zero.

The second pass proposed smaller primitives: a 1D convolution along an
axis, RGBA word packing, color interpolation, and a signed distance to a
rounded rectangle.

The third pass dissolved them:

| Proposed primitive | Where it went | Why |
|---|---|---|
| Separable convolution | `vyre-libs` `math::conv1d` | A 1D convolution is domain-neutral: signal processing, audio, natural language and image work all use it. |
| Pixel pack and unpack | Existing IR | `Expr::bitand`, `Expr::shr`, `Expr::shl` and `Expr::bitor` do it directly. |
| Color interpolation | Existing IR | `a + (b - a) * t` is three expressions. A color is a value. |
| Signed distance to a rounded rectangle | Private to its one consumer | Only the shadow op needs it, so it stays private until a second caller appears. |

The `visual/` primitive domain was deleted. The one shared primitive that
came out of it is `math::conv1d`, which blur composes and future signal
and audio ops compose too. The domain compositions live in
`vyre-libs/src/visual/` over `math::conv1d`, existing IR expressions, and
private helpers where one caller exists.

Domain thinking creates domain primitives, and composition thinking dissolves
them into math. When discovery says nothing existing maps, run it again at a
lower level. A color interpolation is a multiply-add. A pixel unpack is a bit
shift. The operation is almost always already in `math`, in `text`, or in the
IR itself.
