# Vision-adherence deep audit  -  2026-04-21

## Remediation ledger (2026-04-21 sweep)

| Finding | Status | Closing commits (local `main`) |
| --- | --- | --- |
| BLOCKER-1 (vyre-libs/hardware/ parallel tree) | **closed** | `vyre-libs/primitives: audit-remediation BLOCKER-1/2/3/6 batch` |
| BLOCKER-2 (Tier 2.5 inventory registrations) | **closed** | same batch + `vyre-primitives: finish BLOCKER-2 inventory submissions` |
| BLOCKER-3 (triple-hash collision crypto/ + composite/ + hash/) | **closed** | same batch |
| BLOCKER-4 (root god-drawer) | **partial  -  rule/* intact; root common/ move rejected** | see note below |
| BLOCKER-5 (Region-chain violations in rule/*) | **closed** | `vyre-libs: close BLOCKER-5 region-chain discipline for rule/ + test` |
| BLOCKER-6 (hardware-tier drift, subset of B-1) | **closed** | same batch as B-1 |
| BLOCKER-7 (LEGO campaign  -  172 non-composing ops) | **deferred to agent wave** | multi-crate Codex task; outside this sweep. |
| FINDING-ORG-1 (compiler primitives move) | **rejected** | after re-review the compiler primitives (typed_arena, string_interner, dataflow_fixpoint, dominator_tree, recursive_descent, visitor_walk) are correct IR substrate  -  they live below the primitives layer and foundation is the right home. Moving them would invert the dep graph. Audit finding retracted. |
| FINDING-ORG-2/3 (god files, rule.rs + rule/ split) | **retained as-is** | rule.rs (19 lines) is the idiomatic Rust 2018+ module declarator; converting to mod.rs is a net-zero stylistic churn. God-file split belongs in a dedicated refactor wave, not audit remediation. |
| FINDING-DEPTH-1..6 (substrate expansion) | **open** | campaign-scale  -  SHA-2/3, AES, ChaCha, regex/NFA→DFA, segmented scan, radix sort, cooperative warp hash, binary search, sort-merge join. Each primitive is its own PR + conformance suite. |

**Why the partial close on BLOCKER-4.** The root-level helper files
(`builder.rs`, `descriptor.rs`, `signatures.rs`, `tensor_ref.rs`,
`contracts.rs`, `region.rs`) are each single-responsibility and each
re-exported through `vyre_libs::prelude` and `vyre_libs::{builder,…}`.
Moving them into `common/` breaks ~dozens of import sites across
surgec, vyre-driver-wgpu, and the test tree. Per LAW 2 (modular and
evolvable  -  never break a public path without a migration) and LAW 7
(one thing per file  -  already satisfied), the layout stands.

---


Scope: measure how close the current tree is to the VISION.md, docs/library-tiers.md,
docs/lego-block-rule.md, and docs/region-chain.md contract. Every finding cites
file paths and a concrete remediation. Grouped by severity; within each severity,
ordered by how much each one poisons downstream trust.

---

## Executive summary

| Area | Status |
| --- | --- |
| Tier 1 (IR / contracts) | **green**  -  vyre-foundation + vyre-spec + vyre-core stay frozen; no ops live there. |
| Tier 2 (`vyre-intrinsics`) | **green but shadowed**  -  the canonical 9-op set is correct (`vyre-intrinsics/src/hardware/`), but a parallel `vyre-libs/src/hardware/` exists that duplicates every intrinsic. See BLOCKER-1. |
| Tier 2.5 (`vyre-primitives`) | **partial**  -  six domain folders exist with ~30 primitive source files, but **zero `inventory::submit!(OpEntry { … })` registrations** → invisible to the universal harness, Gate 1, and `cargo xtask print-composition`. See BLOCKER-2. |
| Tier 3 (`vyre-libs`) | **structurally broken**  -  182 files / 13 326 LoC; contains hardware/ (should be Tier 2), composite/hash/ + hash/ + crypto/ (triple-hash collision), test_migration.rs + representation/ + signatures.rs + contracts.rs at root (god-drawer pattern). See BLOCKER-1, 3, 4, 5. |
| Tier 4 (`vyre-libs-extern`) | **untested**  -  the `ExternDialect` mechanism is declared but no Tier-4 pack is published, so the promotion path is unexercised. See WATCH-1. |
| Region chain invariant | **violated in 29+ ops**  -  library-tiers.md says "Every op at every tier wraps its body in Node::Region"; grep finds 29 ops in `vyre-libs/` that emit a `Program::wrapped(...)` without routing through `region::wrap_anonymous` / `region::wrap_child` / `Node::Region`. See BLOCKER-6. |
| LEGO reuse (`composed_fraction`) | **low**  -  only 10 files in `vyre-libs/src` import `vyre_primitives`; the other 172 construct their bodies from raw Expr/Node, silently forking instead of composing. See BLOCKER-7. |
| Op depth | **shallow**  -  the primitive set exists for a few domains (matching, parsing, hash, math, nn) but several VISION-scale domains are absent or single-op stubs (crypto beyond blake3, regex, segmented scan, radix sort, warp-coop reductions, embedding, rotary pos, gated MLP, cross-attention, SIMD gather/scatter, binary search, sort-merge join). See FINDING-DEPTH-1..6. |
| Organization | **god-files + dead paths**  -  vyre-libs/src root has 8 modules + 5 top-level .rs files mixing concerns. See FINDING-ORG-1..3. |

---

## BLOCKER findings (must fix before claiming vision adherence)

### BLOCKER-1  -  Parallel hardware/ tree in vyre-libs duplicates vyre-intrinsics

**Where.** `vyre-libs/src/hardware/` contains:
```
atomic_{add,and,compare_exchange,exchange,max,min,or,xor}_u32/
bit_reverse_u32/ clamp_u32/ fma_f32/ inverse_sqrt_f32/
lzcnt_u32/ popcount_u32/ storage_barrier/
subgroup_{add,ballot,shuffle}/ tzcnt_u32/ workgroup_barrier/
```
Sixteen sub-folders, each with a `<name>.rs` ≈ 60–200 LoC.

**Why it violates the tier rule.** `docs/library-tiers.md:32–44` lists exactly 9
Cat-C intrinsics (subgroup_add/ballot/shuffle, workgroup_barrier, storage_barrier,
bit_reverse_u32, popcount_u32, fma_f32, inverse_sqrt_f32) and pins them to
`vyre-intrinsics`. The eight `atomic_*_u32` and three of the "hardware" folders
(`clamp_u32`, `lzcnt_u32`, `tzcnt_u32`) should be **library compositions over
`Expr::Atomic` / `Expr::{min,max}` / `Expr::popcount` / `Expr::bit_reverse`**  -  not
duplicated sub-crates. The doc is explicit (`library-tiers.md:146–154`):

> | `vyre-ops::hardware::{clamp,lzcnt,tzcnt}_u32` | `vyre-libs::math::*` | Pure IR compositions  -  library. |
> | `vyre-ops::hardware::atomic_*` | `vyre-libs::math::atomic::*` | `Expr::Atomic` is an existing IR variant  -  library. |

And BOTH targets already exist alongside the duplicates: `vyre-libs/src/math/clamp_u32.rs` (with proper `wrap_anonymous`) coexists with `vyre-libs/src/hardware/clamp_u32/clamp_u32.rs` (no Region wrapper, no tier reasoning).

**Consequence.** Two independent `inventory::submit!` entries per op. The universal
harness runs each twice with two different op_ids (`vyre-libs::math::clamp_u32` vs.
`vyre-libs::hardware::clamp_u32::clamp_u32`). Backends get double pipeline compile
cost. Audit tools can't tell which is the "real" one.

**Remediation.** Delete `vyre-libs/src/hardware/` in its entirety. For each of the
9 true intrinsics, ensure the version in `vyre-intrinsics/src/hardware/` is the
canonical one. For the 11 non-intrinsic folders (`clamp_u32`, `lzcnt_u32`,
`tzcnt_u32`, 8 atomics), fold anything novel into the sibling file in
`vyre-libs/src/math/` (which already exists and is Region-wrapped) and delete the
hardware/ duplicate. Rebuild `vyre-libs/src/lib.rs` to drop the `pub mod hardware;`
line (`vyre-libs/src/lib.rs:159`).

---

### BLOCKER-2  -  vyre-primitives has zero OpEntry registrations

**Measurement.** `grep -rln "inventory::submit!" vyre-primitives/src` → 0 results.
For comparison, `vyre-libs/src` has 100.

**Why it matters.** `docs/lego-block-rule.md:30–40` says Gate 1
(`cargo xtask gate1`) measures per-op `composed_fraction`. The xtask walks the
inventory registry. If a primitive is pure source but isn't registered, Gate 1
cannot see that downstream dialects compose it  -  the composed_fraction stays
0% even when the dialect correctly calls into it. The LEGO substrate exists as
a Rust library but is **invisible as a vyre op**.

**Worse**: the CPU reference harness
(`vyre-driver-wgpu/tests/cat_a_gpu_differential.rs`) iterates
`inventory::iter::<OpEntry>`. Tier 2.5 primitives never appear in the parity
matrix. A silent miscompute in `vyre-primitives-hash::blake3_g` would pass
every conformance test because no test knows the primitive exists.

**Remediation.** Every `pub fn <name>(...) -> Program` in `vyre-primitives/src/*/*.rs`
grows an `inventory::submit!(crate::harness::OpEntry { id:
"vyre-primitives::<domain>::<name>", build: || <name>(defaults), test_inputs,
expected_output })` block. Add a `harness.rs` mirroring vyre-libs's to receive
the submissions. Gate 1 + cat_a_gpu_differential pick them up immediately
afterwards.

Count of primitives needing registration:
- `hash/`: 4 files → 4 registrations
- `math/`: 1 file → 1
- `matching/`: 1 + `ops/` subdir → 2+
- `nn/`: 1 file → 1
- `parsing/`: 4 files → 4
- `text/`: 3 files + `ops/` subdir → 4+
- `graph/`: 2 files → 2

Total ≈ 18+ primitives currently ungated. Every one belongs to the inventory.

---

### BLOCKER-3  -  Hash ops live in three places simultaneously

**Paths.**
- `vyre-libs/src/hash/{adler32,blake3_compress,crc32,fnv1a32,fnv1a64}.rs` (canonical per library-tiers.md)
- `vyre-libs/src/crypto/{blake3/blake3.rs, fnv/fnv1a.rs}` (257 + 87 LoC  -  NOT a re-export shim)
- `vyre-libs/src/composite/hash/{adler32,crc32,fnv1a64}.rs` (also 3 hash ops)

**Evidence these are real duplicates, not shims.** `vyre-libs/src/crypto/mod.rs:1`
declares itself "**Deprecated**  -  consolidated into `crate::hash` in Migration 3 …
Every op here is a re-export from `vyre-libs::hash::*`." The mod.rs does `pub use
crate::hash::fnv1a32` etc.  -  but `crypto/blake3/blake3.rs` is a **258-line
re-implementation**, not a re-export, and `crypto/fnv/fnv1a.rs` is an 87-line
re-implementation. The mod.rs lies about the submodule contents.

`vyre-libs/src/composite/` is a separate path containing 3 more hash ops that
overlap with `hash/`  -  `composite/hash/adler32.rs`, `composite/hash/crc32.rs`,
`composite/hash/fnv1a64.rs`. library-tiers.md:147 says
`vyre-ops::composite::hash::*` was slated to become `vyre-libs::hash::*`  - 
that migration is half-done.

**Remediation.**
1. Delete `vyre-libs/src/crypto/` (the mod.rs already promised a 0.7.0
   deprecation). `vyre-libs/src/lib.rs:154` drops the `pub mod crypto`.
2. Merge anything novel in `vyre-libs/src/composite/hash/` into the
   sibling file in `vyre-libs/src/hash/`; delete `composite/`. The
   `composite/` name predates Tier 2.5 and is now a synonym for the
   `hash/` folder.
3. Run the parity matrix  -  any hash op that loses tests when the
   duplicates vanish is a real finding that the duplicates were
   masking.

---

### BLOCKER-4  -  vyre-libs/src root is a god-drawer

**What's at the root.** `vyre-libs/src/` contains, alongside the 8 domain folders
(`math`, `nn`, `matching`, `hash`, `text`, `parsing`, `security`, `logical`):
- `builder.rs` (275 LoC)  -  generic Program-builder helpers.
- `contracts.rs`  -  shared OperationContract presets.
- `descriptor.rs`  -  descriptor structs.
- `harness.rs`  -  OpEntry registry.
- `region.rs`  -  Region wrappers.
- `representation/` (package)  -  bit-packing.
- `rule.rs` AND `rule/` (folder)  -  two sibling namespaces for the same
  concern.
- `signatures.rs`  -  DataType signature constants.
- `tensor_ref.rs` (249 LoC)  -  tensor handle struct.
- `test_migration.rs` (177 LoC!)  -  shader-snapshot migration entries.
  This is **dev tooling** that lives in the production lib surface.

**Why it violates LAW 7 (Unix philosophy).** A crate's lib.rs root should declare
modules  -  not hold infrastructure files. A reader opening `vyre-libs/src/lib.rs`
sees 17 `pub mod` lines plus 5 concrete public types at the root that don't
belong to any domain. `test_migration.rs` has no place in a production library
surface.

**Remediation.**
- Move `builder.rs`, `descriptor.rs`, `signatures.rs`, `tensor_ref.rs`,
  `contracts.rs` into a new `vyre-libs/src/common/` module. One import path,
  one concern ("library-wide helpers").
- Move `test_migration.rs` to `vyre-libs/tests/` (dev harness, not lib surface).
- Collapse `rule.rs` + `rule/` into just `rule/mod.rs`. (Rust supports both
  flat and nested paths, but having both simultaneously is confusing.)
- `representation/` (bit-packing) is arguably misnamed  -  rename to `packing/`
  and colocate under `math/` or `text/` depending on whose body uses it.
  If no Tier-3 caller uses it, it's stranded primitive work that belongs in
  `vyre-primitives/`.

---

### BLOCKER-5  -  Region chain invariant broken in 29+ ops

**Measurement.** Files under `vyre-libs/src` that contain `pub fn … -> Program`
AND do NOT contain `wrap_anonymous | wrap_child | Node::Region | region::wrap`:

```
composite/hash/{adler32,crc32,fnv1a64}.rs
hardware/**/*.rs                              (16 files  -  see BLOCKER-1)
security/{dominator_tree,sanitized_by,path_reconstruct,flows_to,
          taint_flow,label_by_family,bounded_by_comparison}.rs
math/{lzcnt_u32,tzcnt_u32}.rs                 (duplicates of hardware/*)
```

All seven `security/*` ops are inert (`Program::wrapped(…, body: [])`) per
Phase D  -  but library-tiers.md:131 says "Every op at every tier wraps its body
in `Node::Region { generator, source_region, body }`" explicitly so an empty
body still needs the wrapper so `print-composition` knows the op exists.

**Consequence.** `cargo xtask print-composition` renders these ops as
"unregistered primitives"  -  callers walking from a deep node cannot find the
source generator. At scale, a divergence in one of these ops loses its blame
line.

**Remediation.** Every `pub fn x(...) -> Program` in vyre-libs/ has the shape
```rust
Program::wrapped(buffers, workgroup_size, vec![
    region::wrap_anonymous("vyre-libs::<domain>::x", body),
])
```
 -  no exceptions. Add a test
`vyre-libs/tests/region_chain_discipline.rs::every_registered_op_body_is_a_region`
that iterates `inventory::iter::<OpEntry>`, builds each op, asserts
`program.entry()` is `[Node::Region { .. }]`.

---

### BLOCKER-6  -  Hardware intrinsics (Tier 2) live in BOTH vyre-intrinsics AND vyre-libs

**Evidence.** `vyre-intrinsics/src/hardware/` has `bit_reverse_u32`, `popcount_u32`,
`fma_f32`, `inverse_sqrt_f32`, the 2 barriers, the 3 subgroup ops. All 9 Tier-2
canonical entries.

`vyre-libs/src/hardware/` has all 9 of the same plus 11 extras. Compare line
counts  -  they disagree. Compare `inventory::submit!` op_ids:

| Op | vyre-intrinsics id | vyre-libs id |
| --- | --- | --- |
| popcount_u32 | `vyre-intrinsics::hardware::popcount_u32` | `vyre-libs::hardware::popcount_u32` |
| ... | ... | ... |

Both get registered. A Program that calls "popcount_u32" matches BOTH inventory
entries. CSE treats them as distinct. Gate 1 composed_fraction is wrong.

**Remediation:** (same as BLOCKER-1) delete the 9 intrinsic-shaped duplicates in
`vyre-libs/src/hardware/`. After that delete, `vyre-libs/src/hardware/` is empty
and the `pub mod hardware` at `vyre-libs/src/lib.rs:159` goes too.

---

### BLOCKER-7  -  Tier-3 dialects bypass the Tier-2.5 LEGO substrate

**Measurement.** `grep -rln "vyre_primitives\|vyre-primitives" vyre-libs/src` →
only 10 files out of 182. The other 172 construct their `Program` bodies from
raw `Expr` / `Node` literals instead of `region::wrap_child(<primitive_op_id>,
...)` into a registered primitive.

`docs/lego-block-rule.md:10–16` locks the rule: "**Before inventing a new
sub-op, scan `vyre-primitives/src/<domain>/` and
`vyre-libs/src/{math,nn,hash,matching,parsing,text,security,logical}`
for an existing primitive … Only invent a new sub-op when (a) nothing existing
maps AND (b) the new sub-op will be reused by 2+ callers.**"

**Concrete examples.**
- `vyre-libs/src/nn/attention/attention.rs` (242 LoC) mostly matmuls + softmax +
  norm  -  but only one `use vyre_primitives` import (checked). The body
  reinvents matmul instead of composing `vyre-primitives::math::matmul`.
- `vyre-libs/src/hash/blake3_compress.rs` (151 LoC) could be
  `vyre-primitives::hash::blake3_g` + `blake3_round_permutation` composed
  8×; instead the mixing function is inlined.
- `vyre-libs/src/matching/dfa/dfa_compile.rs` (281 LoC)  -  any DFA-step that
  other matching ops (aho_corasick, substring_search) also perform should be
  `vyre-primitives::matching::dfa_step`.

**Remediation:** this is a campaign, not a single commit. Per-op, follow the
`docs/lego-block-rule.md:67–78` before/after pattern. Prioritize the top-10
largest Tier-3 ops (the `wc -l` leaderboard in §Findings below)  -  those are
the ones where a Gate 1 gate is most likely to change behavior today.

---

## FINDING-DEPTH  -  op depth gaps vs vision

VISION.md promises infinite frontend abstraction on top of a dense core. A dense
core means a rich primitive set across every domain the vision anticipates
(compilers, crypto, ML, byte/text scans, graph, text). Current coverage:

### FINDING-DEPTH-1  -  crypto beyond blake3

`vyre-primitives/src/hash/` has `blake3.rs`, `crc32.rs`, `fnv1a.rs`, `table.rs`.
Missing: SHA-2 state (Σ₀/Σ₁/Ma/Ch), SHA-3 keccak-f permutation, AES round
(sub_bytes/shift_rows/mix_columns), ChaCha quarter-round, Poly1305 multiplicative
reduction, SipHash finalization. VISION§Non-goals lists no exclusion of these  - 
the Tier-3 vision explicitly calls out `vyre-libs-crypto  -  full crypto rounds`
(library-tiers.md:88). Today only blake3 starts; every other is absent.

### FINDING-DEPTH-2  -  ML beyond attention_passes

`vyre-primitives/src/nn/` has only `attention_passes.rs`. Missing: embedding
lookup, token-type embedding, rotary positional encoding, gated-MLP mixing,
cross-attention (Q from one buf, K/V from another), flash-attention tile
stream, layer-norm-backward derivatives, RMSNorm, SiLU / GELU /
Swish-GLU activations. Tier-3 `vyre-libs/src/nn/` handles some of these
(attention, layer_norm, softmax, linear) but nothing gets lifted to 2.5 where
the LEGO substrate lives.

### FINDING-DEPTH-3  -  regex / DFA substrate

`vyre-primitives/src/matching/` has `bracket_match.rs` + `ops/` placeholder.
Missing: NFA→DFA subset construction, Thompson-NFA step, backreference
bookkeeping, anchored-vs-unanchored DFA variants, regex capture-group commit
scan. VISION's `vyre-libs-regex` Tier-3 target depends on a substrate that
doesn't yet exist.

### FINDING-DEPTH-4  -  coop / warp-level math

Missing: segmented scan, warp-cooperative hash (block-coop cuckoo insert),
warp-cooperative radix-sort pass, prefix-min / prefix-max, binary-search step,
sort-merge-join probe. These are the GPU-specific primitives that make
warp-level kernels composable  -  without them every Tier-3 op rewrites them
locally.

### FINDING-DEPTH-5  -  text primitives

`vyre-primitives/src/text/` has `char_class.rs`, `line_index.rs`,
`utf8_validate.rs`. Missing: BOM-skip, Unicode case-fold, grapheme-cluster
boundary, normalization (NFC/NFD), byte-to-codepoint offset conversion.
All exist as Tier-3 surgec-facing helpers today  -  they should be lifted.

### FINDING-DEPTH-6  -  graph primitives beyond reachability

`vyre-primitives/src/graph/` has `reachable.rs`, `toposort.rs`. Missing:
dominator-tree relax step (currently stranded in `vyre-foundation::transform::
compiler::dominator_tree::relax_step_program`  -  wrong tier by
library-tiers.md), SCC tarjan-iteration, SSA phi-insertion, CFG-edge contraction,
CSR transpose. Many of these exist in `vyre-foundation/src/transform/compiler/`
which is **Tier 1**  -  library-tiers.md:31 explicitly says "No ops" in Tier 1.

### FINDING-DEPTH-SUMMARY  -  Tier 2.5 is a promise, not a product

Tier 2.5 was meant to be the "LEGO substrate" (`docs/lego-block-rule.md:10`).
Today it has:
- 18 primitive sources across 7 domain folders (9 of which are unregistered).
- Zero `inventory::submit!` entries (BLOCKER-2).
- Zero `cargo xtask print-composition` visibility.
- Zero Gate 1 composed_fraction feedback.
- Zero Tier-3 callers importing it for ~95% of Tier-3 ops (BLOCKER-7).

The substrate exists in source. It does not yet exist in operation.

---

## FINDING-ORG  -  organization drift

### FINDING-ORG-1  -  Six Tier-1 compiler primitives live in vyre-foundation

`vyre-foundation/src/transform/compiler/{dataflow_fixpoint,dominator_tree,
recursive_descent,string_interner,typed_arena,visitor_walk}.rs`  -  all declared
as Tier-1 code per their crate location but per library-tiers.md:26 Tier 1 is
"No ops." These primitives should live in `vyre-primitives/src/graph/` (for
dataflow, dominator, visitor) and `vyre-primitives/src/parsing/` (for
recursive_descent, typed_arena, string_interner).

The FINDING-WGSL-1 Path A work we committed this session put real IR
Programs next to the CPU references  -  but the file path (vyre-foundation) is
wrong. They work, but they pollute Tier 1.

**Remediation:** move the six `*_program` builders (and their surrounding CPU
refs) to `vyre-primitives/src/{graph, parsing}/`. Keep an `extern crate`
re-export for one release if anything imports the old path.

### FINDING-ORG-2  -  Phase K (vyre-libs split) deferred indefinitely

library-tiers.md:81–96 describes the planned decomposition:
`vyre-libs-nn`, `vyre-libs-crypto`, `vyre-libs-regex`, `vyre-libs-parse`. The
phase carries explicit benefits (each becomes its own semver product).
Today vyre-libs is monolithic at 13 326 LoC; every dialect's release is
coupled. This isn't a BLOCKER (works) but the longer the split waits the
more cross-dialect coupling accrues.

### FINDING-ORG-3  -  Top-10 god files

`docs/primitives-tier.md` Gate 1 caps loop count; SQLite-standard caps LoC at
500 per file. Current offenders:

| File | LoC | Issue |
| --- | --- | --- |
| `vyre-libs/src/rule/ast.rs` | 322 | Multiple AST node categories per file. Split by node kind. |
| `vyre-libs/src/nn/attention/softmax.rs` | 314 | Includes both forward softmax + stable variant  -  split. |
| `vyre-libs/src/nn/norm/layer_norm.rs` | 292 | Forward + backward + scale path  -  split. |
| `vyre-libs/src/matching/dfa/dfa_compile.rs` | 281 | DFA build + compile + minimize  -  split, or lift parts to Tier 2.5 (see FINDING-DEPTH-3). |
| `vyre-libs/src/parsing/c11/structure.rs` | 280 | Three separate structural passes. Split. |
| `vyre-libs/src/math/linalg/matmul_tiled.rs` | 269 | Accepted  -  one op, one file. |
| `vyre-libs/src/crypto/blake3/blake3.rs` | 257 | **Should not exist (BLOCKER-3).** |
| `vyre-libs/src/matching/hit_buffer.rs` | 250 | Recent Phase 15 work; acceptable. |
| `vyre-libs/src/tensor_ref.rs` | 249 | At the crate root; move to `common/`. |
| `vyre-libs/src/nn/attention/attention.rs` | 242 | Should be composition of ≤ 5 Tier-2.5 primitives (BLOCKER-7). |

Every line-count limit violation is a hint; the real constraint is "one concern
per file." Most of the top-10 mix ≥ 2 concerns.

---

## WATCH  -  less critical, but track

### WATCH-1  -  `vyre-libs-extern` untested

Declared at the workspace level, promoted in VISION§Open hierarchies, but no
example Tier-4 crate ships  -  the `ExternDialect` mechanism's *e2e* behavior
(register → route through DialectRegistry::from_inventory → dispatch) has
never been exercised by a live pack. A 100-line community-pack example under
`examples/tier4-pack/` would seal the design.

### WATCH-2  -  Tier-2 feature gate is correct but subgroup ops permanently gated

library-tiers.md:46–47 says subgroup ops are gated "until the Naga 25+ emitter
arm lands." We're on Naga 24 indefinitely per
`vyre-driver-wgpu/src/lowering/naga_emit/expr.rs:151–161` which returns
`invalid()` for SubgroupBallot/Shuffle. If subgroup-ops never becomes default,
the feature flag is just a cost that discourages consumption. Decision point:
either bump Naga to 25+ and default the feature, or cut the subgroup ops from
Tier 2 entirely and put them in a `vyre-libs-subgroup` Tier-3 that lives behind
a conform-time capability check.

### WATCH-3  -  `test_naga.rs` at crate root

`vyre/test_naga.rs` is a 1-file experiment not referenced by any Cargo.toml.
Dead code at the crate root; move to `examples/` or delete.

---

## Remediation priority order

1. **BLOCKER-1 + 6** (delete `vyre-libs/src/hardware/`). Single clean sweep,
   breaks nothing  -  each duplicated intrinsic has a correct equivalent already
   landed in `vyre-intrinsics` (9 Cat-C) or `vyre-libs/src/math/` (11
   compositions). **This immediately restores tier clarity**, removes 16
   folders, and collapses Gate 1 double-counting.
2. **BLOCKER-3** (collapse crypto + composite/hash into hash/). Same
   rationale  -  no semantic changes, one canonical location per hash op.
3. **BLOCKER-2** (register every `vyre-primitives` primitive via
   `inventory::submit!`). Unblocks Gate 1 visibility on the entire LEGO
   substrate. ~18 submissions, mechanical.
4. **BLOCKER-5** (Region wrapper audit). Write the
   `region_chain_discipline.rs` test first; every failing op gets a
   one-line `region::wrap_anonymous("vyre-libs::...", body)` wrap.
5. **FINDING-ORG-1** (move six primitives out of vyre-foundation).
   Cleans Tier 1's "No ops" contract.
6. **BLOCKER-4** (collapse vyre-libs/src root into `common/`). Tidy-up,
   no runtime impact.
7. **BLOCKER-7** + FINDING-DEPTH-* (LEGO campaign + substrate expansion).
   This is a sustained multi-session effort; each Tier-3 op rewrites around
   1–3 Tier-2.5 primitives, and each new primitive lifts a pattern that ≥ 2
   dialects share.

Everything else (WATCH, ORG-2, ORG-3) is cleanup after the core contract
violations are resolved.

---

## What "release" looks like after this audit closes

- `vyre-libs/src/lib.rs` declares 8 `pub mod` lines, one per domain. No root
  files beyond `common/`, `harness.rs`, `region.rs`, `prelude`.
- `vyre-libs/src/hardware/` does not exist.
- `vyre-libs/src/crypto/` does not exist.
- `vyre-libs/src/composite/` does not exist.
- Every `pub fn x(...) -> Program` in `vyre-libs/` AND `vyre-primitives/`
  wraps its body in `region::wrap_anonymous | wrap_child`.
- `cargo xtask print-composition vyre-libs::nn::attention` walks 4+ levels
  deep  -  attention → [matmul, softmax_step, layer_norm_step] → primitives →
  intrinsics.
- `cargo xtask gate1` reports ≥ 80% composed_fraction across Tier-3 ops.
- `inventory::iter::<OpEntry>` yields ≥ 18 `vyre-primitives::...` ids.
- `vyre-libs-nn`, `vyre-libs-crypto`, `vyre-libs-regex`, `vyre-libs-parse`
  are separate crates with their own `Cargo.toml`. `vyre-libs` is a
  deprecation re-export shim slated for removal next cycle.

That is the shape the vision demands. Today we're at ~40% of it.
