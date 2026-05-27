# Vision-Alignment Audit  -  vyre + surgec north-star fidelity

**Date:** 2026-04-23  
**Scope:** vyre-foundation, vyre-core, vyre-intrinsics, vyre-libs, vyre-driver-wgpu, vyre-runtime, surgec  
**Source of truth:** `VISION.md` (vyre), `docs/library-tiers.md`, `docs/primitives-tier.md`, `docs/region-chain.md`, `libs/tools/surgec/README.md`  
**Authored by:** Claude, hands-on read.

This file tracks where the codebase drifts from the north star. The vision is:

> **vyre** is the missing GPU compute stack layer  -  a **cognitive-offload
> stratum** that lets authors express "I need a state machine / fixed-point
> solver / stack / hashmap" without carrying warp topology or shared-memory
> bank conflicts in working memory. **Core is frozen. Frontends are
> infinite. Backends are substrate-specific.**
>
> **surgec** is the *first* infinite-frontend consumer: a SURGE-lang
> compiler + orchestrator that translates domain rules into arbitrary
> computation authored against `vyre::ir::*` and dispatched through
> `vyre_driver::VyreBackend`. The moat is: every rule is a Program; every
> Program is portable across Vulkan, Metal, DX12, SPIR-V, photonic…; every
> community-contributed rule becomes permanent leverage.

The 4-tier rule: Tier 1 = IR model only (no ops). Tier 2 = hand-audited
hardware intrinsics (9 ops). Tier 2.5 = shared `fn(...) -> Program`
primitives across Tier 3. Tier 3 = domain-specific compositions. Tier 4
= community packs.

Every finding below is a vision drift. Rank by how far it pulls the
codebase from the north star, not by code LOC or CVE severity.

---

## V1  -  Tier 1 leakage: domain types live in `vyre-foundation` that should be Tier 3

**Severity:** HIGH  
**File:** `vyre-foundation/src/match_result.rs:17-26`  
**Re-exports:** `vyre-core/src/lib.rs:144,170` (`pub use match_result::Match`)  
**Call sites:** `vyre-driver-wgpu/src/.../scan_shared` returns `Vec<vyre::Match>`, multiple surgec sites consume it.

**Drift:** The `Match { pattern_id: u32, start: u32, end: u32 }` struct lives in
`vyre-foundation`. That crate is Tier 1: "IR model, wire format, frozen
contracts. **No ops**." A "pattern match" is a matching-domain concept  - 
*pattern_id* only makes sense once a dialect has decided that byte-range
scans exist and enumerate patterns. Foundation shouldn't know "pattern".

The core vision says: "Frontends translate domain concepts into generic
math *before* hitting core. Core never knows what a 'rule', a 'file', or
a 'borrow checker' is." `Match.pattern_id` is exactly that kind of
domain concept.

**Why it matters:** every new dialect that wants to return byte-range
evidence (crypto decoder positions, AST spans, taint-source locations,
regex capture groups) must either (a) reuse this matching-flavored type,
cementing the "vyre = scanner" misread, or (b) ship its own parallel
byte-range type, fragmenting the universal return shape vyre promises.

**Fix:** rename `Match` to a neutral `ByteRange { tag: u32, start: u32,
end: u32 }` and move it into `vyre-primitives::range` (Tier 2.5). Keep
`Match` as a `#[deprecated] pub type Match = ByteRange;` re-export for
one cycle so downstream code breaks with a clear migration arrow, not a
silent trait-search storm. `pattern_id` becomes `tag` because the field
is just "whatever index the producer wants", and the producer  -  a
dialect, a scanner, a decoder  -  gets to choose what the tag means.

**Test hint:** `use vyre::ByteRange; use vyre_primitives::matching::DfaMatch;`
 -  both work, the second is a thin type alias. Ossify via a public-API
lockfile so the tier boundary can't silently regress.

---

## V2  -  `vyre-foundation` exports scan-engine data structures through a Tier-1 door

**Severity:** MEDIUM (follows from V1)  
**File:** `vyre-foundation/src/match_result.rs` + `vyre-core/src/lib.rs:144`

**Drift:** `pub use match_result;` at the core level also violates the
"redundant re-exports" finding already logged in `audits/V7_api.toml:64`.
Re-export hygiene is a symptom, but the root cause is V1: the type
should never have been in foundation. Fixing V1 fixes V2 automatically.

**Fix:** bundled with V1.

---

## V3  -  surgec's SURGE lowering has string-domain predicates that don't yet map to arbitrary compute

**Severity:** HIGH  
**File:** `libs/tools/surgec/src/compile/ir_emit.rs:386-520` (`emit_predicate`)

**Drift:** surgec's `emit_predicate` has 20+ hard-coded `Predicate::*`
variants (`Any`, `All`, `Count`, `FileSize`, `Before`, `After`, `Near`,
`Between`, `SameScope`, `SameFile`, `CrossFile`, `CrossFileChain`,
`NotInScope`, `Contains`, `Chain`, …). Each arm is a specialised
pattern-matcher / co-ordination predicate. This is exactly the
"infinite frontend"  -  GOOD.

BUT: the vision is **arbitrary compute**, not just predicates. The
`Predicate` enum is closed. A SURGE author who wants a brand-new
coordinate predicate (e.g., "rule-fires-on-ast-subtree") today has to
edit `surge` the language crate, then `surgec` the compiler, then
potentially `vyre-libs::security` to add a helper. That is three crates
edited for one user-facing op.

**Goal under the vision:** a SURGE author registers a new predicate
through a `PredicateDef` extension point, which lowers to `fn(args) ->
Program` via `Predicate::Opaque(DialectId)`. No edits to surgec core.
Mirrors vyre's `Expr::Opaque` / `Node::Opaque` extension mechanism.

**Status:** surgec has a `PredicateDef` registry file
(`libs/tools/surgec/src/compile/predicates/`). Confirm every built-in
predicate is registered through the same door (not hard-coded in
`emit_predicate`). Lowering should be
`registry.get(pred_id).lower(args)?`, never a match arm per built-in.

**Fix (structural):** reshape `emit_predicate` into a registry-first
dispatch:
```rust
match pred.kind() {
    PredicateKind::Any | PredicateKind::All => /* short-circuit helpers stay */,
    PredicateKind::Registered(id) => registry.lower(id, pred.args())?,
}
```
Every other arm in today's `emit_predicate` becomes a registered
built-in, each a file under
`libs/tools/surgec/src/compile/predicates/` that ships its
`fn(args) -> Program`.

---

## V4  -  Promise vs reality: "arbitrary compute" is still rule-shaped surface

**Severity:** HIGH (this is the biggest drift)  
**Files:** `libs/tools/surgec/src/cli/mod.rs` (subcommands), `src/scan/*`

**Drift:** surgec's CLI exposes `scan`, `compile`, `bundle`,
`diff-replay`, `watch`, `distribute`. Every verb is scan-shaped. The
tool does not yet expose `run <program.surge>` that dispatches an
*arbitrary* compiled SURGE program  -  a SURGE author who wants to
express "here is a dataflow solver" or "here is a bitset fixpoint on
my CSR graph" cannot invoke surgec to run it end-to-end without
wrapping it as a "rule" first.

The user's directive is clear: **surgec ready to scan AND perform
arbitrary compute**. A `run` verb that takes a compiled `.surge` +
binary inputs and dispatches through `vyre_driver` (and streams outputs
back) is the missing piece. The existing `scan` path can be implemented
as `run` with a standard matching-output schema.

**Fix:** design + ship a `surgec run` subcommand. Binding contract:
```
surgec run <program.surge> \
    --input a=path/to/a.bin \
    --input b=path/to/b.bin \
    --output c=path/to/c.bin \
    --workgroup-size 256 \
    --max-dispatch-bytes 1GiB
```
`scan` then becomes a frontend-convenience wrapper that locks the
output schema to `[Vec<ByteRange>]` and handles the walker. The
arbitrary-compute path goes through `run` and emits raw bytes.

---

## V5  -  `vyre-libs::security` is Tier-3, but `surgec::compile::ir_emit` calls it directly

**Severity:** MEDIUM  
**File:** `libs/tools/surgec/src/compile/ir_emit.rs:7,410,421` (`use vyre_libs::security::topology::match_order`)

**Drift:** surgec reaches into `vyre_libs::security::topology`  -  a
Tier-3 dialect  -  to get `match_order`. That's correct IF surgec is a
security-domain consumer. But surgec is pitched as a **general-purpose
compiler**: anything a SURGE author writes should be able to run. If
surgec itself hard-codes a dependency on the `security` dialect inside
the Generic compile path, every non-security user (e.g. data-science,
robotics, search) pulls `vyre-libs::security` through no choice of
theirs.

**Fix:** either (a) move `match_order` into `vyre-primitives::coord`
if it's generic ordering of slot firings (it is), or (b) gate the
surgec call behind the `security` feature so a `surgec --no-default-features`
compile works for non-security SURGE programs.

**Test hint:** `cargo tree -e no-dev -p surgec --no-default-features`
must not list `vyre-libs-security` (or whatever the split name
becomes).

---

## V6  -  vyre-intrinsics doc says 9 ops; hardware/ has 9 subdirs  -  confirm

**Severity:** INFO (verified, matches)  
**File:** `docs/library-tiers.md:35`

**Status:** verified. `hardware/` contains exactly
`{bit_reverse_u32, fma_f32, inverse_sqrt_f32, popcount_u32,
 storage_barrier, subgroup_add, subgroup_ballot, subgroup_shuffle,
 workgroup_barrier}` = 9. Every op requires dedicated Naga emission
and a CPU reference. OK, not a drift.

---

## V7  -  Region-chain invariant: is it provable today?

**Severity:** HIGH (needs answering before vyre-libs-extern ships)  
**Files:** `docs/region-chain.md`, `vyre-intrinsics/src/region.rs`,
`vyre-libs/src/region.rs`

**Drift:** the vision promises "`cargo xtask print-composition
<op_id>` walks the chain from any public op down through every
intermediate composition to the hardware intrinsic leaves." That is
the *only* mechanism that keeps Tier-3 black boxes auditable.

**Question to close:** does `print-composition` exist? Is it CI-gated?
If an attention block (Tier 3) composes a softmax (Tier 3) that
composes an `exp` (Tier 2.5) that calls `fma_f32` (Tier 2), can I `grep`
that chain from a shipped binary today?

**Test hint:** write a CI test that constructs an attention program,
runs `print-composition`, and asserts the output contains every Tier-2
leaf on the chain. Fail if any `Region::Opaque { source_region: None }`
appears on a non-Tier-2 op (because that means a black-box ingested
without provenance).

---

## V8  -  "The frontend owns domain concepts" is violated where surgec imports vyre-libs::security::topology inside generic lowering

See V5. Folded here for the map, but the concrete fix is V5.

---

## V9  -  Vision mentions PTX, Metal, photonic backends; driver is WGSL/SPIR-V only today

**Severity:** INFO (roadmap, not a bug)  
**File:** vyre-driver-wgpu, vyre-driver-spirv

**Status:** OK that only two backends ship in 0.6. But the *architecture
doc* should name which backend targets are on the 0.7 / 0.8 roadmap so
the "backends are substrate-specific, swappable" promise is concrete,
not aspirational. Add to `docs/targets.md` if not already present.

---

## V10  -  surgec lacks a `run-arbitrary-program` end-to-end test

**Severity:** HIGH  
**File:** none (the test does not exist).

**Drift:** every surgec E2E test is scan-shaped. There is no test that
does: compile a `.surge` expressing matrix multiply / convolution /
generic dataflow → dispatch through wgpu → assert raw output bytes.
Until that test exists, "arbitrary compute" is a claim not a capability.

**Fix:** add `libs/tools/surgec/tests/run_arbitrary.rs` with one
hand-rolled program (e.g., `gemv` 32x32) that compiles, dispatches,
and asserts bytewise-correct output against a CPU reference.

---

## Summary

| ID | Severity | Area | Status |
|----|----------|------|--------|
| V1 | HIGH | Tier 1 leakage of `Match` | Open  -  propose rename to `ByteRange` |
| V2 | MEDIUM | Core re-export hygiene | Bundled with V1 |
| V3 | HIGH | SURGE predicates hardcoded vs registered | Partial  -  confirm registry-first dispatch |
| V4 | HIGH | `surgec run <program>` verb missing | Open  -  new subcommand required |
| V5 | MEDIUM | surgec unconditional `vyre-libs::security` import | Open  -  gate or hoist to primitives |
| V6 | INFO | vyre-intrinsics 9-op lock | Verified matches doc |
| V7 | HIGH | Region-chain `print-composition` CI-gate | Open  -  audit existing tooling |
| V8 |  -  | Folded into V5 |  -  |
| V9 | INFO | Backend roadmap visibility | Minor doc task |
| V10 | HIGH | Arbitrary-compute E2E test missing | Open  -  ship one fixture |

**Next hands-on:** V1 (mechanical rename to `ByteRange` + deprecation
alias), V10 (write the gemv E2E test as a soak-proof). V3/V4/V5/V7 are
cross-crate structural changes best dispatched as a deep wave.

---

*Vision audits must run every session from now on. A codebase that
silently drifts from its north star is a codebase that never ships the
product its investors, users, and maintainers believed they were
buying. Reading the vision against the code each cycle is the only
way to catch slow drift.*
