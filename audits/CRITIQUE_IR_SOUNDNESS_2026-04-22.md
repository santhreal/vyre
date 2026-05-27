# CRITIQUE_IR_SOUNDNESS_2026-04-22

Date: 2026-04-22
Scope: `libs/performance/matching/vyre/vyre-foundation/src/`  -  the vyre IR (Expr, Node, Program, BufferDecl, DataType), its wire format (`serial/wire/{encode,decode,framing,tags}`), validator (`validate/*`), visitor traits (`visit/{expr,mod,traits}.rs`), and the `Program::from_entry` / `Program::new` call-site inventory across the workspace.

Method: static read-only audit of the source tree as of commit `f3b0f6ef0b`. Every finding names a concrete path, describes what breaks, and proposes a fix. No source files modified.

Every finding is a soundness bug  -  a shape of Program where one of
(a) `decode(encode(p)) ≠ p`, (b) a malformed Program passes validation then crashes a backend, (c) two nominally-equal Programs produce different dispatch results, or (d) a visitor silently drops information. Soundness bugs are worse than performance bugs: they make every rule depending on the IR wrong, not slow.

---

## 1. Wire format  -  encode/decode round-trip

**F-IR-01 HIGH [fixed 2026-04-23]** | `vyre-foundation/tests/wire_roundtrip_proptest.rs` | Proptest strategy now enumerates every `Expr` + `Node` variant (`AtomicOp`, `BinOp`, `UnOp`, `Var`, `Call`, `Region`, `Loop`, `Let`, `AsyncLoad/Store/Wait`, `Trap`/`Resume`, Opaque expr + node with real `ExprNode` / `NodeExtension` resolvers registered via `OpaqueExprResolver` / `OpaqueNodeResolver`) plus adversarial inputs (empty names, UTF-8 edge cases like `"snow_雪"`, NUL-embedded strings). Each generated Program is encoded, decoded, and asserted structurally equal with zero residual bytes; the regression file lives alongside the fuzz target that also hammers the same path.

**F-IR-02 CRITICAL [fixed 2026-04-23]** | `vyre-foundation/src/opaque_payload.rs::canonical_regex_flags` | No in-tree `Literal::Regex` variant exists  -  regex literals flow through `Expr::Opaque` extensions owned by SURGE / future dialects. The endian-fixed opaque payload contract now includes `canonical_regex_flags(flags) -> String` which sorts + dedupes inline flag characters, so a SURGE-lang regex literal encoded as `(?mi)pat` and `(?im)pat` reaches the encoder as the same bytes and hashes equal under `Program::hash`. The docstring + regression test pin the contract, and the `ExprNode::wire_payload` docstring points extension authors at the helper explicitly.

**F-IR-03 HIGH [fixed 2026-04-23]** | `vyre-foundation/tests/wire_roundtrip_proptest.rs:604+` | `signaling_nan_payload_roundtrips_bit_exactly` pins the NaN payload contract: `f32::from_bits(0x7f80_0001)` (signalling) plus quiet-NaN and every sign combination encode + decode to the same `to_bits()`. The general proptest excludes NaN so the targeted case owns the payload-preservation guarantee rather than silently masking NaN equality through `==`.

**F-IR-04 HIGH [fixed 2026-04-23]** | `vyre-foundation/tests/wire_roundtrip_proptest.rs:565-586` | `subnormal_f32_roundtrips_bit_exactly` walks the 2^23 − 1 subnormal mantissas in both signs, encodes the Program, decodes, and asserts `to_bits()` equality  -  so a flush-to-zero regression in the encoder / decoder is caught immediately.

**F-IR-05 HIGH [fixed 2026-04-23]** | `vyre-foundation/src/ir_inner/model/program.rs:~330` | `BufferDecl::with_count(0)` now panics at construction with the Fix: hint distinguishing the runtime-sized convention ("don't call `with_count`") from the positive-count convention ("pass a strict `> 0` value"). Zero-size static buffers are a validation failure on every shipped backend (WebGPU `ZERO_SIZE_BUFFER_USAGE`, Vulkan zero-size allocation, reference interpreter), and the panic catches the author at the call site instead of at dispatch time. Workgroup buffers have an independent `V0xx` rejection in `validate/validate.rs:90` for count==0 that pre-dates this fix.

**F-IR-06 HIGH [fixed 2026-04-23]** | `vyre-foundation/src/validate/validate.rs:65-68` | Validator rejects any `workgroup_size[axis] == 0` with `"workgroup_size[{axis}] is 0. Fix: all workgroup dimensions must be >= 1."`. Rejection is unconditional across all three axes and fires before any other per-buffer checks, so a malformed wire-decoded program can't slip past.

**F-IR-07 HIGH [fixed 2026-04-23]** | `vyre-foundation/src/ir_inner/model/program.rs:744-747` | `Program::empty()` now produces the canonical no-op shape (one empty root `Node::Region`), and `is_explicit_noop()` recognises that shape explicitly so backends dispatch-through the shader without rejecting the program. Non-empty programs must satisfy the region-wrap invariant via `is_top_level_region_wrapped()`  -  a raw `entry: Vec::new()` is no longer reachable from a blessed constructor.

**F-IR-08 HIGH [fixed 2026-04-23]** | `vyre-foundation/src/serial/wire/tags.rs` + `encode/to_wire.rs` + `decode/from_wire.rs:82-90` | `FLAG_OPAQUE_ENDIAN_FIXED = 1<<2` is now a mandatory framing flag. Every encoded program sets the flag; every decoder rejects a payload missing it with `"InvalidDiscriminant: wire header is missing OPAQUE_ENDIAN_FIXED. Fix: reserialize with a producer that writes opaque payload numerics using little-endian bytes."`. The docstring contract plus the new `opaque_payload::push_*` / `read_*` helpers (F-IR-32) give extension authors an actionable path to honour the contract.

**F-IR-09 HIGH [fixed 2026-04-23]** | `vyre-foundation/src/serial/wire/decode.rs:10-12` | Opaque ids are now widened to u32 and the decoder rejects any id whose top bit is clear  -  core IR lives in `0x0000_0000..=0x7fff_ffff`, dialect extensions must claim `0x8000_0000..=0xffff_ffff`. Mismatches surface as `InvalidDiscriminant: {surface} opaque id 0x{raw:08x} collides with core IR. Fix: dialect extensions must use ids in 0x8000_0000..=0xffff_ffff.`, catching every reserved-range collision at decode time.

**F-IR-10 HIGH [fixed 2026-04-23]** | `vyre-foundation/src/serial/wire/framing/magic.rs` + `serial/wire/decode/from_wire.rs:76-80` | `WIRE_FORMAT_VERSION = 3` is a strict equality check (`if version != WIRE_FORMAT_VERSION { ... UnknownSchemaVersion ... Fix: upgrade the consumer or re-serialize with this Vyre version. }`). Newer-version programs are rejected rather than silently partial-decoded.

**F-IR-11 MEDIUM [fixed 2026-04-23]** | `vyre-foundation/src/validate/validate.rs:90-94` + `ir_inner/model/program.rs:~330` | Workgroup-memory buffers with `count == 0` are rejected up front by the validator (`"workgroup buffer `{name}` has count 0. Fix: declare a positive element count."`), and `BufferDecl::with_count(0)` panics at construction across all access kinds (see F-IR-05). The semantics are fully defined now instead of undefined.

---

## 2. Validator coverage  -  every variant must have a rule

**F-IR-12 CRITICAL [fixed 2026-04-23]** | `vyre-foundation/src/validate/expr_rules.rs` | `grep '_ =>' expr_rules.rs` returns zero matches  -  every `Expr` variant is listed explicitly, including `Opaque`, so a new `#[non_exhaustive]` variant forces a compile error instead of silently PASSing validation.

**F-IR-13 HIGH [fixed 2026-04-23]** | `vyre-foundation/src/validate/options.rs` + `expr_rules.rs:113` | The validator is now parameterised by a `BackendCapability` trait with `supports_cast_target(&DataType) -> bool`. When options carry a concrete backend, `expr_rules.rs` rejects any `Expr::Cast` whose target the backend cannot emit. Callers that skip the backend parameter get the "best-effort universal" fallback (`supports_cast_target` defaults to true), which the module docstring names explicitly.

**F-IR-14 CRITICAL [fixed 2026-04-23]** | `vyre-foundation/src/validate/atomic_rules.rs` | `validate_atomic` walks every atomic-op site and rejects targets whose access is not `ReadWrite`: three distinct V009 rejection arms name the bad access (ReadOnly, Uniform, or other) and carry a targeted Fix: hint. `expr_rules.rs:143` wires the check into the standard validator pipeline so every `Expr::Atomic` variant is covered before the program can reach the backend.

**F-IR-15 HIGH [fixed 2026-04-23]** | `vyre-foundation/src/validate/binding.rs:39` | `V032` rejects sibling-duplicate Let bindings within the same region: `"V032: duplicate sibling let binding `{name}` in the same region. Fix: rename one binding or move one declaration into an inner Block/Region/Loop if a new scope is intended."`. Nested scopes opt in via `Program`-level `allow_shadowing` so inner regions can legitimately rebind without the validator flagging them.

**F-IR-16 HIGH [fixed 2026-04-23]** | `vyre-foundation/src/validate/nodes.rs:343-378` | `check_constant_store_index` evaluates every `Node::Store` whose index is a `LitU32` or `LitI32` literal and rejects values `>= buffer.count` (plus negative LitI32) with `V036: store index {value} overflows buffer `{buffer_name}` with count {count}. Fix: keep constant store indices below the declared element count.`. Non-constant indices fall through to runtime bounds checking  -  the validator cannot evaluate arbitrary expressions but catches every static overflow.

**F-IR-17 HIGH [fixed 2026-04-23]** | `vyre-foundation/src/validate/depth.rs` | `DEFAULT_MAX_EXPR_DEPTH = 1024` is now the hard cap. `check_expr_depth` rejects deeper trees with `"V033: expression nesting depth {depth} exceeds max {DEFAULT_MAX_EXPR_DEPTH}. Fix: split the expression into intermediate let-bindings before lowering."`. Validator call sites (`expr_rules.rs`) bail out on the first depth-exceeded error so pathological nested Arrow/And/Or trees cannot DoS the validator stack.

**F-IR-18 MEDIUM [fixed 2026-04-23]** | `vyre-foundation/src/serial/wire.rs:56` + `decode/from_wire.rs:56-60` | `MAX_PROGRAM_BYTES = 64 * 1024 * 1024` is the framing-layer budget. `from_wire` rejects any payload exceeding the cap before allocation, so a hostile blob cannot force unbounded allocation. `MAX_BUFFERS` / `MAX_NODES` add per-list caps on top of the overall byte budget.

**F-IR-19 HIGH [fixed 2026-04-23]** | `vyre-foundation/src/validate/shadowing.rs:11-13` | `V008` rejects nested-scope shadowing by default: `"V008: duplicate local binding `{name}` shadows an outer scope. Fix: choose a unique local name, or opt into nested shadowing with ValidationOptions::with_shadowing(true)."`. Callers who intentionally want shadowing flip `ValidationOptions::allow_shadowing = true`, so silent scope confusion between the reference interpreter and naga emitter is ruled out.

**F-IR-20 MEDIUM [fixed 2026-04-23]** | `vyre-foundation/src/validate/cast.rs` + `expr_rules.rs:129-131` | `cast_is_narrowing(source, target)` flags every truncating conversion and the validator emits `"V035: narrowing cast from `{src}` to `{target}` may truncate high bits. Fix: prove the source value fits before casting, or rewrite through an explicit checked conversion path."`. The diagnostic names both source and target types so the author can pick the right fix immediately.

---

## 3. Buffer aliasing  -  Programs shipped under `vyre-libs/src/`

**F-IR-21 CRITICAL [fixed 2026-04-23]** | `vyre-libs/src/decode/inflate.rs`, `base64.rs`, `hex.rs`, `vyre-libs/src/hash/*.rs` | Every shipped Cat-A hash and decode builder now rewrites generic placeholder names (`input`, `output`, `out`, `decoded`, `cv_in`, `msg`, `params`, `cv_out`, `chaining_in`, `chaining_out`, `message`) into family-scoped names via `buffer_names::scoped_generic_name(FAMILY_PREFIX, role, requested, aliases)`. `blake3_compress`  -  the only remaining family without scoping  -  was wired in this pass. Regression: `vyre-libs/tests/buffer_name_cross_family.rs` builds every hash + decode family with the canonical generic names and asserts the flattened buffer-name set has zero collisions.

**F-IR-22 HIGH [fixed 2026-04-23]** | `vyre-libs/src/math/atomic/mod.rs`, `vyre-foundation/src/optimizer/fuse_batch.rs` | Atomic ops declare the target buffer as ReadWrite. If the SAME buffer is passed as an INPUT (ReadOnly) to another node and as ReadWrite to the atomic, the program is valid per single-node validation but the fused dispatch has a write-after-read without a barrier. **Fix**: `fuse_programs` walks buffer declarations across all arms, builds `name -> Vec<(arm_index, BufferAccess)>`, and inserts `Node::Barrier` between a read arm and a later write/atomic arm.  Barrier insertion preserves the fused-dispatch semantics instead of forcing a split.  Regression: `vyre-foundation/tests/fusion_atomic_aliasing.rs` pins the barrier-insertion shape.

**F-IR-23 HIGH [fixed 2026-04-23]** | `vyre-libs/src/parsing/core/delimiter.rs` and every parser Program in `vyre-libs/src/parsing/` | The parser state machine stores intermediate state in a workgroup buffer. If two rule invocations of the same parser run in the same fused kernel, they share the workgroup buffer → state corruption. **Fix**: `Program::non_composable_with_self` field (default false) plus `Program::with_non_composable_with_self(true)` on every parser builder.  `fuse_programs` rejects batches where two programs share the same `entry_op_id` and either is non-composable-with-self, returning `FusionSelfAliasingError { op_id, fix }`.  Regression: `vyre-foundation/tests/fusion_atomic_aliasing.rs` proves two `delimiter_parser` programs fail fusion with the expected error.

**F-IR-24 HIGH [fixed 2026-04-23]** | `vyre-libs/src/hash/fnv1a64.rs` + every sibling hash family | Subsumed by F-IR-21's family-scoped naming: each hash family defines `FAMILY_PREFIX = "hash_<name>"` and rewrites its generic `"input"` / `"out"` aliases to `__vyre_hash_<name>_input` / `__vyre_hash_<name>_out` via `buffer_names::scoped_generic_name`. The cross-family regression `buffer_name_cross_family.rs` builds adler32, crc32, fnv1a32, fnv1a64, blake3_compress simultaneously with the same generic placeholders and asserts zero collisions.

---

## 4. Visitor completeness

**F-IR-25 CRITICAL [fixed 2026-04-23]** | `vyre-foundation/src/visit/expr.rs` | Every `ExprVisitor` variant method is now abstract (no default body); rustc rejects any implementor that drops a variant. `ExprVisitor::walk_children_default(expr, order)` is the opt-in pass-through helper so visitors that want default recursion name it explicitly.

**F-IR-26 HIGH [fixed 2026-04-23]** | `vyre-foundation/src/visit/mod.rs` | Traversal order is now explicit  -  `VisitOrder::{Preorder, Postorder}` plus the `visit_preorder` / `visit_postorder` entry points are documented on the trait and exercised by unit tests (`visit_preorder`/`visit_postorder` assertions in `visit/mod.rs`).

**F-IR-27 HIGH [fixed 2026-04-23]** | `vyre-foundation/src/visit/traits.rs` | `NodeVisitor` now mirrors the abstract-by-default discipline: every variant method takes `&mut self` + node ref and returns `ControlFlow<Self::Break>`, with `walk_children_default(node, order)` as the opt-in recursion helper.

**F-IR-28 MEDIUM [fixed 2026-04-23]** | `vyre-foundation/src/visit/expr.rs` | Every visitor method returns `ControlFlow<Self::Break>`; callers can short-circuit traversal. Tests cover a first-occurrence visitor that breaks early.

---

## 5. Region chain invariant  -  every composition must wrap

**F-IR-29 CRITICAL [fixed 2026-04-23]** | `vyre-foundation/src/ir_inner/model/program.rs` + every `vyre-libs` Cat-A builder | `Program::wrapped(buffers, workgroup_size, entry)` is the blessed constructor  -  it wraps `entry` in `Node::Region` and is what every shipped builder calls. `Program::new` is `#[deprecated]` with a note directing authors to `Program::wrapped`, and the validator rejects a runnable program whose entry node is not `Node::Region` with an actionable Fix: hint. The historical call sites named in the finding all now route through `Program::wrapped`.

**F-IR-30 HIGH [fixed 2026-04-23]** | `vyre-reference/src/interp.rs:13-20` | `ensure_top_level_region(program)` now guards both public entry points (`reference_eval` and `run_arena_reference`). Programs whose top-level entry is not a `Node::Region` are rejected with `"reference interpreter requires a top-level Region-wrapped Program: {message}"` where `message` is `Program::top_level_region_violation()`. Conformance divergence between the reference and GPU backend caused by an unwrapped Program is now a loud interp error, not a silent scoping mismatch.

**F-IR-31 HIGH [fixed 2026-04-23]** | `vyre-runtime/src/megakernel/dispatcher.rs` + upstream | `MegakernelDispatcher::dispatch` currently takes `BatchRuleProgram` (a DFA-shape bundle) rather than raw `Program`, and every upstream compiler emits Programs through `Program::wrapped`, which is the only blessed constructor. Region-wrap is enforced at the `Program::wrapped` / `Program::new_raw` boundary via `is_top_level_region_wrapped`; the reference interpreter rejects unwrapped Programs (F-IR-30); the validator reports `"program entry node {index} is \`{}\` instead of \`Node::Region\`"` for non-compliant programs. Let-hoisting in the fusion layer can only operate on arms that already carry their original Region, so a non-wrapped arm cannot reach dispatch without tripping one of those gates first.

---

## 6. Opaque payload round-trip across endianness

**F-IR-32 HIGH [fixed 2026-04-23]** | `vyre-foundation/src/ir_inner/model/expr.rs` (Opaque definition) | The endian-fixed contract is now enforced at the SDK level. `ExprNode::wire_payload` and `NodeExtension::wire_payload` both carry the little-endian convention in their docstrings, and the new `vyre_foundation::opaque_payload` module gives extension authors `push_u{16,32,64}` / `push_i{16,32,64}` / `push_f{32,64}` (plus matching `read_*`) built on `to_le_bytes` / `from_le_bytes`. The readers return an actionable `OpaquePayloadTruncated` error with field name, expected width, actual size and a Fix: hint when input is short. Regression: `vyre-foundation/tests/opaque_payload_endian.rs` pins the canonical byte layout (`0x01020304` → `[04,03,02,01]`), round-trips every width, and proves the truncation diagnostic surfaces.

**F-IR-33 MEDIUM [fixed 2026-04-23]** | `vyre-foundation/src/dialect_lookup.rs` | `install_dialect_lookup` now requires every `DialectLookup` impl to expose a stable `provider_id()`. A same-id reinstall is a silent no-op (harness-friendly). A different-id reinstall is a loud panic at install time that names both providers and carries an actionable Fix hint, closing the LAW 4 "<60s to root cause" gap. Regression: `vyre-foundation/tests/dialect_lookup_install.rs` proves the idempotent path stays quiet and the conflicting-id path panics with the prescribed message.

---

## 7. Category A/B/C classification drift

**F-IR-34 HIGH [fixed 2026-04-23]** | `docs/migration-vyre-ops-to-intrinsics.md` | The classification rule: A = pure composition over existing IR, B = needs a dedicated Naga arm, C = needs hardware-specific emit. Confirmation that shipped ops are classified correctly is not machine-checked. An op that claims Category A but actually requires a Naga arm passes through `vyre-libs` but breaks on non-wgpu backends. **Fix**: add a static-assertion `build.rs` in `vyre-intrinsics` that walks the inventory and asserts every entry's category matches the presence/absence of a Naga emitter arm in `vyre-driver-wgpu`.  -  *Closed by `vyre-intrinsics/build.rs` (static source scan) + `vyre-intrinsics/src/category_check.rs` (runtime inventory walk + adversarial fixtures).*

**F-IR-35 HIGH [fixed 2026-04-23]** | `vyre-libs/src/math/atomic/mod.rs` | Atomic ops are classified as Category A (composition) in some parts of the codebase but require naga atomic emission (which F-NAGA FINDING-34 shows is handled via `atomicCompareExchangeWeak`). If they're composition-only, a backend that lacks atomics silently produces incorrect output. **Fix**: reclassify atomics as Category B; add a build-time assertion that the backend must provide atomic arms.  -  *Closed by adding `OpDefRegistration` with `Category::Intrinsic` to every `vyre-libs/src/math/atomic/*.rs` op, updating crate-level docs to acknowledge the Category-B exception, and wiring the build.rs scanner to catch any Composite op with a Naga arm.*

**F-IR-36 MEDIUM [fixed 2026-04-23]** | `vyre-intrinsics/src/hardware/popcount_u32.rs` (and sibling bit intrinsics) | Classified as Category C (hardware-bound). Most GPUs have a `countOneBits` instruction, so the naga arm is correct. But f32 FMA is also C  -  and some older GPU drivers legitimately don't support `fma` as a distinct instruction and the driver falls back to `a * b + c` without the fused rounding. **Fix**: document the FMA round-mode guarantee in `vyre-intrinsics/src/hardware/fma_f32.rs` + fall-back strategy if the backend reports the capability as absent.  -  *Closed by adding function-level docstrings to `fma_f32` (explicit `FMA` capability check, no silent `a*b+c` fallback) and `popcount_u32` (emit `Unsupported` when `countOneBits` is absent).*

---

## 8. Literal equivalence and hashing

**F-IR-37 CRITICAL [fixed 2026-04-23]** | `vyre-foundation/src/opaque_payload.rs::canonical_regex_flags` + `canonical_f32_zero` | The opaque-payload SDK now ships both canonicalisation helpers that extension authors opt into before framing: `canonical_regex_flags` (sort + dedupe inline flag characters) and `canonical_f32_zero` (normalise `-0.0 → +0.0` while preserving every non-zero f32 bit pattern including NaN payloads). The docstrings explain WHY the core IR does not canonicalise automatically (IEEE-754 `-0.0` has real semantic meaning under division, and NaN payload preservation is load-bearing  -  see F-IR-03), so authors that need hash equality opt in explicitly. Regression in the same module covers flag sorting, dedup, and sign-of-zero normalisation; non-zero f32 and NaN payloads round-trip unchanged.

**F-IR-38 HIGH [fixed 2026-04-23]** | `vyre-foundation/src/serial/wire/tags/data_type_tag.rs` + `decode/impl_reader.rs` | `DataType::Bool` is pinned to a single-byte tag `0x06` on the wire, and `Expr::LitBool` encodes/decodes as one `u8` (`self.u8()? != 0`). Separate bitset packing stays in the `Vec<u32>` primitive path  -  no part of the IR wire format packs bools into bits. `put_data_type` has a matching `expect("Fix: DataType::Bool must encode as one u8 tag")` that makes accidental divergence a loud panic at encode time.

---

## 9. Program identity / structural equality

**F-IR-39 HIGH [fixed 2026-04-23]** | `vyre-foundation/src/ir_inner/model/program.rs` (PartialEq derive) | `Program` now documents and enforces order-insensitive buffer equality, so semantically-identical programs do not compare unequal just because buffer declarations arrived in a different order.

**F-IR-40 HIGH [fixed 2026-04-23]** | `vyre-foundation/src/ir_inner/model/arena.rs` | `Program::structural_eq(&self, &Self)` is now the explicit O(N) public contract for structural comparison, keeping arena-local allocation identity out of public equality semantics.

---

## 10. Dispatch / runtime soundness

**F-IR-41 HIGH [fixed 2026-04-23]** | `vyre-driver/src/shadow.rs` | The sampled shadow path has been replaced by an exhaustive conformance matrix. Module docstring: `"The old sampled shadow path compared a runtime sample of live dispatches. That could never prove soundness: a backend bug whose divergence rate stayed below the sample rate would slip through. This module replaces sampling with an explicit conformance matrix."`. Callers build a deterministic `ConformanceCase` set from `vyre-conform-spec`'s witness inventory, and every op × witness tuple must agree byte-for-byte between the backend and the reference adapter  -  no runtime sampling, no tolerance for divergence below a "sample rate".

**F-IR-42 HIGH [fixed 2026-04-23]** | `vyre-runtime/src/pipeline_cache.rs` | `PipelineFingerprint::of(&Program)` is pinned at compile time to `fn(&Program) -> PipelineFingerprint`, so no dispatch-time parameter can accidentally enter the key without the public signature changing. The hash input is `canonicalize::run(program).to_wire()`, and `PIPELINE_FINGERPRINT_ALLOWED_FIELDS` names the four program-intrinsic fields (`canonical_ir_graph`, `buffer_layout`, `declared_workgroup_size`, `canonical_wire_framing`)  -  input buffer count/contents, DispatchConfig labels, timeout, ULP budget, and runtime workgroup overrides are all excluded, as the docstring explicitly calls out.

**F-IR-43 MEDIUM [fixed 2026-04-23]** | `vyre-runtime/src/megakernel/dispatcher.rs` | `accepted_rule_fingerprints` now walks every rule, runs `validate_rule_shape`, and attaches failures to a per-rule `BatchRuleRejection { rule_idx, reason }` without aborting the batch. Duplicate rule-ids, out-of-bounds indices, and empty slots each surface with their own actionable Fix: hint, and the dispatch carries the rejection list through to the caller so remaining rules still dispatch.

---

## 11. Layered / compositional soundness

**F-IR-44 HIGH [fixed 2026-04-23]** | `vyre-foundation/src/optimizer/passes/fusion.rs` | Fusion now carries an explicit happens-before contract plus a regression that pins the load-before-write ordering shape which would flip output if reordered.

**F-IR-45 HIGH [fixed 2026-04-23]** | `vyre-foundation/src/optimizer/passes/dead_buffer_elim.rs` | Dead-buffer elimination now respects pipeline-wide liveness through wrapped and nested `Region` bodies so caller-readback / `pipeline_live_out` buffers are preserved even when their visible writes live inside the root region chain.

**F-IR-46 HIGH [fixed 2026-04-23]** | `vyre-foundation/src/optimizer/passes/autotune.rs` | `Autotune::transform` now injects `Node::if_then(Expr::lt(gid_x, inferred_bound), body)` whenever it rewrites the workgroup size and the program does not already carry a gid.x bounds check. The bound is derived from the largest declared output / pipeline-live-out storage buffer's `BufLen`. When the workgroup size is unchanged, a debug `assert_even_divisible_without_guard` panics on any latent unguarded OOB. Regression: `injects_gid_x_bounds_check_when_rewriting_workgroup_size` in the pass's unit tests.

---

## 12. Dialect / extension soundness

**F-IR-47 HIGH [fixed 2026-04-23]** | `examples/external_ir_extension/` | The external extension docs now spell out the opaque contract: payload bytes are passthrough when the owning resolver is linked, and consumers that cannot handle the tag fail loudly with an actionable missing-resolver error. Regression coverage now pins both behaviours.

**F-IR-48 MEDIUM [fixed 2026-04-23]** | `vyre-foundation/src/dialect_lookup.rs`, `vyre-driver/src/registry/registry.rs` | Dialect registry duplicate-id rejection now names the stable id plus both registrants at init time, so diamond-linked duplicate registrations fail loudly and traceably instead of panicking with a lossy message.

---

## 13. Test surface gaps

**F-IR-49 HIGH [fixed 2026-04-23]** | `vyre-foundation/tests/wire_roundtrip_proptest.rs` | Round-trip is the fundamental soundness claim for any wire format. The suite now exercises wire round-trip over the Program surface and includes an explicit fixture that covers every current `Expr` variant.

**F-IR-50 HIGH [fixed 2026-04-23]** | `vyre-foundation/tests/adversarial_program_canonical_laws.rs` | Program canonical-law coverage now exists alongside the graph suite: hash stability, equality symmetry, wire serialisation idempotence, and structural equality modulo buffer declaration ordering.

**F-IR-51 MEDIUM [fixed 2026-04-23]** | `conform/vyre-conform-runner/tests/parity_matrix.rs` | The parity matrix now checks coverage against a `vyre-spec` expression inventory so every declared `Expr` variant must appear in at least one matrix row.

**F-IR-52 HIGH [fixed 2026-04-23]** | `vyre-foundation/fuzz/fuzz_targets/decoder.rs` | Added a `cargo-fuzz` decoder target so attacker-supplied Program binaries can be exercised continuously against `from_wire` without panicking the harness.

---

## 14. Runtime-only soundness

**F-IR-53 MEDIUM [fixed 2026-04-23]** | `vyre-runtime/src/uring/stream.rs:36-80,194` | `GpuMappedBuffer<'a>` carries `PhantomData<&'a mut [u8]>` and its constructors take a `&'a mut` slice (or `&'a owner` anchor) so the borrow checker prevents the stream from outliving the mapped region. `AsyncUringStream<'a>` inherits the same lifetime bound. Two `unsafe impl Send + Sync` arms document the safety contract (kernel-side DMA only, no Rust-side reads through the pointer).

**F-IR-54 MEDIUM [fixed 2026-04-23]** | `vyre-runtime/src/megakernel/io.rs` | Publishing more than `IO_SLOT_COUNT = 64` completions now returns a structured `QueueFull` error with Fix: "`MegakernelIoQueue exceeds the compiled IO poll window of 64 slots; enlarge IO_SLOT_COUNT and rebuild the megakernel before publishing more than 64 completions`". Regression in the same file exercises the overflow path and asserts the Fix: hint stays actionable.

---

## 15. Summary

**54 findings**: 8 CRITICAL, 30 HIGH, 11 MEDIUM, 5 implicit gaps.

The three biggest surface areas with soundness risk:

1. **Wire format round-trip is not proptest-covered.** The first 11 findings boil down to this. One day of proptest work would collapse most of them.
2. **Visitor defaults are silent on new variants.** Because `Expr` is `#[non_exhaustive]`, every default-method visitor becomes a silent bug carrier the first time a dialect adds a variant.
3. **Region-chain invariant is not enforced.** 30+ call sites to `Program::from_entry` skip the wrap, and the validator does not reject non-wrapped Programs.

Fixing the first two is a 1-2 day investment. Fixing the third is a sweep of ~30 call sites plus a validator rule  -  1 more day.

The ≥1000× gate requires that every rule produces byte-identical output on every backend. Every finding in this document is a possible silent divergence. Close them before publishing performance claims.
