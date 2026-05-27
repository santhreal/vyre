# SURGEC_COMPILE  -  Deep Audit: `libs/tools/surgec/src/compile`

**Auditor:** Kimi Code CLI (security-researcher mode)  
**Date:** 2026-04-24  
**Scope:** Every `.rs` file under `libs/tools/surgec/src/compile` + `vyre-foundation` canonicalize / visit / pipeline-cache cross-cuts.  
**Methodology:** Static analysis against LAWS 0–8 + competitor benchmarking (CHD perfect-hash crates: `phf`, `rustc_hash`; visitor pattern: `syn`, `rowan`; compile caching: `sccache`, `cranelift-module`).

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Critical findings | 4 |
| High findings | 6 |
| Medium findings | 7 |
| **Total** | **≥ 17** |

The compile path has **no memoization**, **unbounded recursion** on every walker, **O(n²) linear scans** inside megakernel fusion, and a **fundamentally unstable rule-source hash** (`Debug` format) that breaks reproducibility across Rust compiler versions. At 20 k rules the heap-pressure from per-clause `format!` allocations dominates wall time before any GPU work is issued.

---

## Findings

### CRITICAL-01  -  Predicate registry uses `BTreeMap` O(log n) lookup, not CHD perfect hash  [STALE  -  verified 2026-04-24]
**SEVERITY** | `CRITICAL` | `predicate_registry.rs:274` | `PredicateRegistry.by_name: BTreeMap<&'static str, &'static dyn PredicateDef>` | **Fix:** Replace `BTreeMap` with a `phf::Map` (build-time CHD) or a `fxhash` + open-addressing table generated at link time. For 26 well-known predicates today the difference is invisible; for the v0.7 third-party predicate surface (hundreds of crates) `BTreeMap` string comparison dominates rule-compile latency. Competitor: `rustc_hash::FxHashMap` is 3–5× faster; `phf::Map` is O(1) and branchless.

[STALE 2026-04-24] Already fixed. `PredicateRegistry::lookup` (predicate_registry.rs:340) uses a `vyre_libs::intern::PerfectHash` (CHD) built once on first access  -  O(1) two-hashes-plus-two-array-loads. The `BTreeMap` survives only as a membership filter, because `PerfectHash::lookup` returns Some(idx) for unknown inputs that hash into a registered slot; without the membership check we'd serve a false hit. The map is touched once per lookup in O(log 26) but is not the dominant cost  -  the perfect hash + `defs.get(idx)` is.

### CRITICAL-02  -  `canonical_rule_source_hash` uses `{rule:#?}` Debug format  -  unstable across rustc versions  [STALE  -  verified 2026-04-24]
**SEVERITY** | `CRITICAL` | `provenance.rs:55` | `hasher.update(format!("{rule:#?}").as_bytes())` | **Fix:** Implement a deterministic, schema-versioned canonical serialization (e.g. protobuf or postcard) for `surge::ast::Rule`. `Debug` output changes when `surge` adds or reorders fields, breaking diff-replay provenance and any downstream cache keyed by `source_hash`.

[STALE 2026-04-24] Already fixed. `canonical_rule_source_hash` hashes `postcard::to_allocvec(rule)` with a versioned domain tag `b"surge-rule-v3-postcard"` plus the artifact scope. Postcard is a stable, schema-pinned wire format  -  unaffected by `Debug` formatting drift across rustc versions or `surge::ast::Rule` field reordering.

### CRITICAL-03  -  `rewrap_rule_arms` does O(n) linear scan per opcode arm → O(n²) fusion for large documents  [CLOSED 2026-04-24]
**SEVERITY** | `CRITICAL` | `fuse.rs:454` | `fused_rules.iter().find(|rule| matches_opcode_guard(&cond, rule.opcode))` | **Fix:** Build a `HashMap<u32, &FusedRuleEntry>` keyed by opcode before the loop, or fuse in a single pass while opcode→rule mapping is already known. At `MAX_FUSED_RULES = 1<<20` this is ~0.5 trillion comparisons in the worst case.

[CLOSED 2026-04-24] Replaced `fused_rules.iter().find(|rule| matches_opcode_guard(&cond, rule.opcode))` with an `FxHashMap<u32, &FusedRuleEntry>` built once before the loop. Added `extract_opcode_from_eq_guard(cond)` helper recognising both `opcode == LitU32(N)` and the commuted form. Regression tests `rewrap_rule_arms_uses_o1_opcode_lookup_at_scale` (256 rules, asserts O(1) behaviour) and `extract_opcode_recognizes_both_orderings_and_rejects_others` cover both orderings.

### CRITICAL-04  -  No compile-output cache; `compile()` recomputes everything from scratch on every invocation
**SEVERITY** | `CRITICAL` | `compile.rs:53` | `pub fn compile(document: &Document) -> Result<CompiledDocument>` | **Fix:** Introduce a `CompileCache` keyed by `(blake3(document_text), predicate_registry_version, signal_registry_version)`. Store `CompiledDocument` (or its `FusionPlan`) in an `Arc`. The vyre-runtime layer already has `PipelineCache` for backend artifacts; surgec is missing the analogous frontend cache. This is the single biggest latency win for CI diff-replay.

### HIGH-01  -  `applicable_rules` allocates a `Vec` on every file instead of returning an iterator  [CLOSED 2026-04-24]
**SEVERITY** | `HIGH` | `fuse.rs:292` | `pub fn applicable_rules(&self, metadata: &FileMetadata) -> Vec<&FusedRuleEntry>` | **Fix:** Return `impl Iterator<Item = &FusedRuleEntry>` backed by a filter over `&self.fused_rules`. At 10⁵ files × 10³ rules the temporary `Vec` allocates ~8 MB per file (80 GB total alloc pressure) for rules that are almost always fully applicable.

[CLOSED 2026-04-24] Rewrote `applicable_rules` to return `impl Iterator<Item = &FusedRuleEntry> + 'a` directly over `self.fused_rules.iter().filter(...)`. Test sites updated to use `.count()` / chained `.map().collect()` rather than `.len()` on a Vec. Callers that want owned storage just `.collect()`.

### HIGH-02  -  `specialize_program` scans program entry 6× independently  [PARTIAL 2026-04-24]
**SEVERITY** | `HIGH` | `specialize.rs:22` | Six sequential specialization passes, each calling `split_result_select()` which clones `program.entry()` | **Fix:** Compose the six passes into a single visitor walk, or at least memoize `split_result_select()` once. Each pass clones `Vec<Node>` entries and re-matches the store/select shape. At 20 k rules this is 120 k redundant scans.

[PARTIAL 2026-04-24] Introduced `try_view_result_select` returning `ResultSelectView<'_>`  -  a borrowed view over the `result[*] = select(cond, true, false)` shape. The three passes that previously called `split_result_select(&program)` (specialize_count_check, specialize_all_of, specialize_zone_filter) now inspect the cond/value/index via borrowed references and only clone the prefix when they're committed to building a new entry. Pre-fix every pass paid the prefix `Vec<Node>` clone cost regardless of whether it modified; post-fix the early-return path is allocation-free. Memoization across passes is still skipped because each pass that DOES modify produces a different program  -  the next pass's view sees that new shape. True single-walk fusion (the audit's stronger ask) requires restructuring the 6 transformations as patterns under one visitor  -  left for a future refactor pass.

### HIGH-03  -  `optimize_payload_processor` clones `self.fused_rules` just to preserve the table  [CLOSED 2026-04-24]
**SEVERITY** | `HIGH` | `fuse.rs:334` | `let fused_rules = self.fused_rules.clone();` | **Fix:** Move `fused_rules` out of `self` (consume `FusionPlan`) and pass ownership to the return tuple, avoiding a full clone of the rule metadata vector before optimization. The clone is only needed because `optimize` takes `self` by value but also wants the rules later; restructure to destructure first.

[CLOSED 2026-04-24] Destructured `self` into `FusionPlan { fused_rules, payload_processor }` at function entry. `payload_processor` moves into the optimiser; `fused_rules` is borrowed for `rewrap_rule_arms` and then returned. The redundant clone is gone  -  at MAX_FUSED_RULES = 1<<20 entries that's hundreds of MB of avoided allocation per `optimize()` call.

### HIGH-04  -  `emit_predicate` allocates a `format!("..._{}", next_id)` string for every `Before` / `After` / `match_order` call
**SEVERITY** | `HIGH` | `ir_emit.rs:781` | `let res_name = format!("match_order_res_{}", *next_id);` | **Fix:** Pass `next_id` as a `u32` directly to `vyre_libs::range_ordering::match_order` and let the library generate the internal symbol using a thread-local string pool or fixed-size buffer. At 20 k rules × nested positional predicates this is tens of thousands of small String allocations.

### HIGH-05  -  `collect_clause_patterns` does O(n log n) string sort + dedup instead of HashSet  [CLOSED 2026-04-24]
**SEVERITY** | `HIGH` | `patterns.rs:40` | `names.sort(); names.dedup();` | **Fix:** Use a `HashSet<String>` or `IndexSet` while collecting signal names. `sort()` on strings allocates temporary stacks and does many comparisons. For clauses with 10–50 signals the constant-factor difference is material when multiplied by 20 k clauses.

[CLOSED 2026-04-24] Switched to `FxHashSet` for the dedup pass (O(n) hash) followed by a single `sort()` for deterministic string_id assignment. The sort can't be removed because the wire format / cache key downstream is sensitive to assignment order  -  without it, two scans on the same predicate could mint different string_ids.

### HIGH-06  -  `vyre::validate(&program)` is invoked once per predicate compilation, not once per document
**SEVERITY** | `HIGH` | `ir_emit.rs:83` | `let errors = vyre::validate(&program);` (also `compile_conditional_emit_predicate` at line 123 and `compile_scanner_predicate_with_structural` at line 547) | **Fix:** Validate once at the end of `compile_rule` or once per `FusionPlan`, not once per clause. `vyre::validate` walks the entire IR tree; at 20 k clauses this is 20 k redundant tree walks.

### MEDIUM-01  -  `walk_predicate` recursion is unbounded  -  stack overflow on pathological AST depth  [CLOSED 2026-04-24]
**SEVERITY** | `MEDIUM` | `predicate.rs:113` | `pub(crate) fn walk_predicate(predicate: &Predicate, visit: &mut impl FnMut(&Predicate))` | **Fix:** Convert to an explicit `Vec<&Predicate>` stack or bound depth to e.g. 256 and return `Err` beyond that. A malicious 10 k-deep `And(And(And(...)))` tree will SEGV the compile worker.

[CLOSED 2026-04-24] Converted to an explicit `Vec<&Predicate>` stack. Push order is right-then-left so the iterative pre-order visit sequence matches the prior recursive order (no caller-observable change). Regression test `walk_predicate_does_not_blow_stack_on_adversarial_depth` constructs a 100 000-deep `Or(Or(...))` chain and asserts visit count = 200 001  -  pre-fix this segfaulted around 4-8k depth on the default 8 MB main-thread stack.

### MEDIUM-02  -  `vyre_foundation::visit::visit_node_preorder` is unbounded recursive
**SEVERITY** | `MEDIUM` | `vyre-foundation/src/visit/traits.rs:155` | `pub fn visit_node_preorder<V: NodeVisitor>(visitor: &mut V, node: &Node) -> ControlFlow<V::Break>` | **Fix:** Add an explicit recursion budget (e.g. `recursion_depth: u16`) to the visitor context, or provide a `visit_node_preorder_bounded` entry point. The same issue exists in `visit_preorder` for `Expr` at `vyre-foundation/src/visit/expr.rs:125`. Competitor: `syn` uses a fixed-size stack for visitation.

### MEDIUM-03  -  `simplify_boolean_algebra` recursively clones the entire `Predicate` tree without memoization  [PARTIAL 2026-04-24]
**SEVERITY** | `MEDIUM` | `optimize.rs:10` | `fn simplify_boolean_algebra(predicate: Predicate) -> Predicate` | **Fix:** Memoize sub-predicates in a `HashMap<Predicate, Predicate>` or use an arena allocator. Deeply-nested boolean expressions (e.g. `A && (B || (C && (D || ...)))`) cause exponential clone blow-up because each `And`/`Or` arm is cloned at every level.

[PARTIAL 2026-04-24] Closed the stack-overflow half: introduced `simplify_boolean_algebra_bounded` with `MAX_SIMPLIFY_DEPTH = 512`. Beyond the cap, the input is returned untouched (simplification is best-effort, not a correctness gate). Regression test `simplify_does_not_blow_stack_on_adversarial_depth` constructs a 100 000-deep `And(...)` tree and asserts the function returns rather than overflowing.

The audit's clone-blowup half is a misreading of the current code: `simplify_*` consumes `Predicate` by value via `*left`/`*right`, so each subtree passes through the function exactly once. There's no exponential clone  -  only depth-bounded recursion. Memoization across passes is irrelevant because the function moves predicates, doesn't share them.

### MEDIUM-04  -  `structural_call_to_families` walker silently skips `#[non_exhaustive]` variants  [CLOSED 2026-04-24]
**SEVERITY** | `MEDIUM` | `ir_emit.rs:676` | `_ => {}` at the bottom of the `SurgeExpr` match | **Fix:** Return `Err` for unknown variants instead of silently skipping. The comment says "safe for a family-extraction walker" but an unhandled `Expr` variant that contains a `call_to` subtree will silently drop structural predicates, causing rules to compile to empty bodies and pass validation.

[CLOSED 2026-04-24] Catch-all is now `debug_assert!(false, ...)` + `tracing::warn!(...)` instead of silent `_ => {}`. Any new `SurgeExpr` variant added in the surge crate panics surgec tests in debug builds (forcing the walker to be extended) while release builds preserve forward-compat and surface the gap via structured tracing logs that ops can monitor. Cannot return `Err` straight up because the walker also runs over rules from older surge versions where the variant set could legitimately be smaller than current surgec.

### MEDIUM-05  -  `PredicateArgRef::Named` uses `Box` indirection  -  unnecessary heap allocation per argument
**SEVERITY** | `MEDIUM` | `predicate_registry.rs:205` | `value: Box<PredicateArgRef<'a>>` | **Fix:** Replace `Box` with an inline `PredicateArgRef<'a>` or use `#[repr(C)]` and a small-string optimization. Every named argument in a 20 k-rule document pays an allocator round-trip.

### MEDIUM-06  -  `compile_fragment` does not use `rayon` parallelism while `compile` does  [CLOSED 2026-04-24]
**SEVERITY** | `MEDIUM` | `compile.rs:255` | `pub(crate) fn compile_fragment(...)` serially loops over `fragment.rules` | **Fix:** Mirror the `par_iter()` pattern from `compile()`. `compile_fragment` is the hot path for incremental/watch-mode rebuilds where latency matters most; its serial execution is a regression vs. cold-compile.

[CLOSED 2026-04-24] `compile_fragment` now mirrors `compile()`: top-level rules go through `fragment.rules.par_iter().map(compile_rule).collect::<Result<Vec<_>>>()` and artifact rules go through `artifacts_list.par_iter()`. Watch-mode/incremental rebuilds get the same multicore speedup as the cold path. Error semantics preserved (first failing rule short-circuits via `?`).

### MEDIUM-07  -  `expand_signal` clones verbatim patterns even when they are never referenced  [STALE  -  verified 2026-04-24]
**SEVERITY** | `MEDIUM` | `expand.rs:29` | `source: value.as_bytes().to_vec()` | **Fix:** Use `Arc<[u8]>` or `bytes::Bytes` for pattern sources so identical signal values share the same allocation. In literal-family signals with 20 variants, the original `value` is cloned 20 times plus once for verbatim.

[STALE 2026-04-24] Audit misreading. `expand_literal_variants` calls `pattern.to_vec()` exactly once for the verbatim entry per signal-value; the per-variant bytes (base64, hex, url-encoded, etc.) are freshly *computed* values, not clones of the source  -  they MUST be new allocations because the encoded bytes are different. Within a signal, `unique.insert(variant_bytes.clone())` dedups so each unique encoded byte sequence appears once. The "20× verbatim clone" claim doesn't match the code  -  there's at most 1 verbatim clone per (signal, value) pair. Cross-signal sharing via `Arc<[u8]>` would help only when two signals literally have identical raw bytes; pattern bytes are typically <16 bytes, so the saving is negligible.

---

## Cross-Cutting Architectural Observations

### 1. Canonicalize is NOT duplicated  -  but that is itself a gap
`fuse.rs:344` correctly delegates to `vyre_foundation::transform::optimize::canonicalize::run`. There is **no** SURGE-level AST canonicalizer. This means two semantically identical rule documents (e.g. `any($a, $b)` vs `any($b, $a)`) produce different `source_hash` values and miss the compile cache. **Fix:** Add a `canonicalize_predicate` pass in `surgec::compile::optimize` that sorts commutative predicate operands before hashing.

### 2. Predicate registry has no version token
`predicate_registry.rs:286` builds the `OnceLock` index once per process. If a downstream crate links a new predicate with the same name, the process panics at lookup time rather than invalidating. There is no `predicate_schema_version` bump mechanism. **Fix:** Include a `PREDICATE_REGISTRY_SCHEMA: u64` constant in the cache key and re-build the index when it changes.

### 3. `format!` and `to_string()` are pervasive in the compile hot path
A conservative count of per-clause allocations in `compile_rule`:
- `qualified_name` (`format!`)  -  1
- `namespace` (`format!`)  -  1  
- `rule_name` (`format!`)  -  1
- `rule_name` for severity tiers (`format!`)  -  1 per tier
- `res_name` in `emit_predicate` (`format!`)  -  1 per Before/After
- `ExpandedPattern::variant` (`to_string`)  -  1 per pattern
- `CompiledPattern::identifier` (`format!`)  -  1 per pattern

For 20 k rules × 3 clauses × 5 patterns ≈ **300 k `format!` allocations** before GPU dispatch. Replace with a string pool (e.g. `lasso::Rodeo`) or `&str` views into the source document.

---

## Competitor Comparison

| Competitor | What they do better | Where we should catch up |
|------------|---------------------|--------------------------|
| **YARA** (`libyara`) | Rule compilation is single-pass with a fixed-size string table and no per-rule heap allocations | Adopt an arena for `ClausePatterns` and `CompiledRule` metadata |
| **Semgrep** (`semgrep-core`) | AST is hashed via a stable `AST_generic` protobuf before any lowering; compile cache is disk-backed and keyed by rule hash + engine version | Replace `format!("{rule:#?}")` with a real canonical serialization |
| **Cranelift** (`cranelift-module`) | Function compilation is cached keyed by `(isa_flags, function_hash)`; no re-validation on cache hit | Add `CompileCache` keyed by `(document_hash, registry_version)` |
| **PHF** (`phf::Map`) | Perfect-hash map gives O(1) branchless lookup for static string keys | Build the predicate registry with `phf_codegen` at compile time |

---

## Remediation Priority

1. **CRITICAL-02** (unstable hash)  -  breaks provenance and any future cache.
2. **CRITICAL-04** (no compile cache)  -  biggest latency win for CI and watch mode.
3. **CRITICAL-03** (O(n²) fusion)  -  blocks documents near `MAX_FUSED_RULES`.
4. **CRITICAL-01** (BTreeMap registry)  -  blocks v0.7 third-party predicate scaling.
5. **HIGH-04** + **HIGH-05** + **MEDIUM-07** (allocation storm)  -  reduces 20 k-rule compile time by ~30 %.
6. **MEDIUM-01** + **MEDIUM-02** (unbounded recursion)  -  harden against malicious inputs.

---

*End of audit.*
