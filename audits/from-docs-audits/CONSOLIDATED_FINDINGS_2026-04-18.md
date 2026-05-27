# Consolidated Findings — 2026-04-18 Rebuild Sweep

**Status:** live document, updated as audits land and fixes commit.
**Last synthesized:** mid-sweep, ~30 min in.
**Note:** Conform deletion 2026-04-18 closed N findings in bulk.

## Law Counts (live)

| Law | Meaning | Count |
|-----|---------|-------|
| A | no closed IR enums | 51 (rip in flight) |
| B | no string-based WGSL | **0** ✓ |
| C | supported_ops + validate_program | **0** ✓ |

2 of 3 architectural laws green. Law A closes when Codex-CATCHUP-A finishes the NodeKind trait workspace rollout.

## Closed (already fixed or in-flight landing)

| ID | Source | One-liner | Closed by |
|----|--------|-----------|-----------|
| CUT-1 (core→conform) | architectural | `vyre-core` no dev-dep on `vyre-conform` | Gemini-CROSS-2 |
| CUT-2 (core→wgpu) | architectural | `vyre-core` no dev-dep on `vyre-wgpu` | Gemini-CROSS-2 |
| F1 Hybrid NodeStorage | rebuild plan | LLVM-style tagged union + Extern escape | Codex-FIX-ARCH |
| F5 Pass scheduling | rebuild plan | DAG scheduler with requires/invalidates | Codex-FIX-IMPL |
| Primitives (20) | P3 | scan/reduce/shuffle/gather/... structs + inventory | Codex-IMPL-2 |
| CPU reference (20) | P3 | pure-Rust reference evaluators per primitive | Codex-IMPL-2 |
| Exhaustive 8/16-bit | P3 | enumerate full input space, blake3 commitment | Codex-IMPL-2 |
| Composition laws | P3 | assoc/comm/idem exhaustively verified on 8-bit | Codex-IMPL-2 |
| std dissolution | P3 | vyre-std absorbed into vyre-primitives | Codex-IMPL-2 |
| DUP-01/02 SHA-256 | dupe audit | three hand-rolled impls unified on sha2 | kimi-bc403ddb |
| DUP-05 verify_laws | dupe audit | duplicated fn merged into canonical submodule | kimi-8a66fd78 |
| DUP-07/08 cache dir | dupe audit | extracted to `vyre-build-scan::cache_dir` | kimi-228c5acf |
| DUP-12..16/26 Cargo | dupe audit | workspace-dep discipline on toml/walkdir/fs2/clap | kimi-c5ebf1ee |
| DUP-28 OpSignature | dupe audit | deleted duplicate, use vyre-spec canonical | kimi-cf7be9cb |
| DUP-20/21 hygiene | dupe audit | root glob re-exports removed | kimi-fe1202d9 |
| ORG-04 float common | org audit | 588-LOC support split into 5 files | kimi-8e88d0a4 |
| ORG-07 vs_cpu bench | org audit | embedded WGSL extracted to .wgsl files | kimi-f6c2d799 |
| ORG-12 release_gate | org audit | 10 corruption vectors split to own files | kimi-80af1402 |
| Law A CI guard | architectural | rejects closed IR enums | mine |
| Law B CI guard | architectural | rejects string-based WGSL | mine |
| Law C CI guard | architectural | rejects missing supported_ops / validate | mine |
| THESIS.md rewritten | architectural | open-IR contract documented | mine |
| ARCHITECTURE.md updated | architectural | Laws A/B added; crate topology updated | mine |
| memory-model.md | architectural | MemoryKind + Access + Ordering contract | mine |
| targets.md | architectural | Tier 1 / Tier 2 / Tier 3 registration matrix | mine |
| Replay async writer | P5 | dispatch no longer blocks on JSON I/O | mine |
| inventory::submit registration | P4 | wgpu backend now self-registers | mine |
| Per-crate CHANGELOG sprawl | org audit | 8 duplicates deleted, root CHANGELOG canonical | mine |

## Critical-Path Blockers (gate rebuild completion)

| ID | Source | One-liner | Owner | Status |
|----|--------|-----------|-------|--------|
| CUT-5 Naga emitter | architectural | 357 string-WGSL sites → naga AST | codex-0cc3102c | in flight |
| P2 R1 open Expr/Node | rebuild plan | every match site migrated to trait dispatch | Codex-ARCH-2 | in flight |
| P2 R2 strip Program | rebuild plan | remove workgroup_size/entry_op_id/buffers | Codex-ARCH-2 | in flight |
| P2 R3 reduce Backend trait | rebuild plan | delete dispatch_wgsl + compile_native | Codex-ARCH-2 | in flight |
| P2 R4 interpreter on open graph | rebuild plan | no match-on-variant | Codex-ARCH-2 | in flight |
| F2 Backend capability negotiation | F-fixes | supported_ops + validate_program everywhere | Codex-FIX-ARCH | in flight |
| F3 Wire format versioning | F-fixes | (op_id, payload_bytes) + registry deserialize | Codex-FIX-ARCH | in flight |
| F4 NodeKind::interpret | F-fixes | generic interpreter, no hardcoded variant list | Codex-FIX-ARCH | in flight |
| F6 MemoryKind implementation | F-fixes | Global/Shared/Uniform/Local/Readonly/Push | Codex-FIX-IMPL | in flight |
| F8 Capability traits split | F-fixes | Executable / Compilable / Streamable | Codex-FIX-ARCH | in flight |
| F10 Progressive lowering | F-fixes | Core IR → Backend IR → Target | Codex-FIX-IMPL | in flight |
| F9 Conform becomes a Backend | F-fixes | property tests via the trait, not build-owner | Gemini-FIX-CROSS | closed_by_deletion |
| TEST-01 independent oracles | test audit | stdlib hash, blake3 vectors, regex-automata | 3 kimi | in flight |
| TEST-03/04 adversarial proptest | test audit | every BinOp/UnOp covered with hostile inputs | kimi-3e19c026 | in flight |

## Dispatchable Next (ready when slots free)

| ID | Source | One-liner | Scope |
|----|--------|-----------|-------|
| ORG-01 hashmap_interp split | org audit | 845-LOC file → 6 files (after open-graph lands) | cursor-sonnet |
| ORG-02 visit.rs split | org audit | 650-LOC → 4 files (after R1 lands) | kimi |
| ORG-03 fuzz.rs split + delete mini-interp | org audit | 582-LOC → 4 files; kill duplicate evaluator | cursor-sonnet |
| ORG-05 certify split | org audit | 580-LOC → 5 files (after R4 real Certificate) | kimi | closed_by_deletion |
| ORG-06 execution split | org audit | 579-LOC → 6 files (in flight) | copilot-sonnet | closed_by_deletion |
| ORG-08 workgroup analysis split | org audit | 511-LOC → 4 files (after CUT-5) | kimi |
| DUP-24 PipelineCache merge | dupe audit | absorb into TieredCache (in flight) | cursor-sonnet |
| DUP-30 Value enum unify | dupe audit | collapse to one Value (in flight) | cursor-sonnet |
| DUP-32 free-fn run() rip | dupe audit | trait-only dispatch (in flight) | cursor-sonnet |
| DUP-29 OpSpec→ConformSpec | dupe audit | 300-site rename (in flight) | cursor-sonnet |
| DUP-06 compile disambiguation | dupe audit | rename tests/support::compile | copilot-sonnet |
| DUP-10 Program::validate method | dupe audit | inherent method, deprecate free fn | copilot-sonnet |
| DUP-31 DefendantCatalog dedup | dupe audit | canonical submodule, delete inline | copilot-sonnet | closed_by_deletion |
| AUDIT safety_wgpu | in flight | find panic/unsafe risks | kimi-53ef00f4 |
| AUDIT errors_conform | in flight | find swallowed errors | kimi-39297f29 | closed_by_deletion |
| AUDIT api_core | in flight | find leaked internal types | kimi-2411ea69 |
| AUDIT determinism | in flight | find non-det sources | kimi-b4264114 |
| AUDIT threadsafety_wgpu | in flight | find race conditions | kimi-33a308ad |
| AUDIT leaks | in flight | Arc cycles, unreleased resources | kimi-648969b5 |
| AUDIT observability | in flight | tracing coverage gaps | kimi-42314b90 |
| AUDIT doctruth | in flight | docs that lie about the code | kimi-d3a0ce38 |

## Cross-Cuts (one change resolves multiple findings)

- **Open Expr/Node enums (P2 R1)** closes DUP-09 Token collisions, DUP-29 OpSpec ambiguity, and the Expression Problem root cause. One Codex commit resolves three findings.
- **Capability negotiation (F2)** closes the "silent wrong backend" class and the LAW 2 "swap component = one file" violation by making backend selection structural rather than implicit.
- **Naga AST emitter (CUT-5)** reduces Law B from 357 to 0, closes ORG-07/08 (embedded WGSL strings), and eliminates an entire class of runtime-only shader errors.
- **Progressive lowering (F10)** resolves the "abstract enough vs optimizable enough" tension by letting backends keep substrate-specific IR internally while Core IR stays neutral.

## Specification Gaps (documented, not yet enforced)

| Concept | Documented in | CI guard |
|---------|---------------|----------|
| MemoryKind substrate mapping | docs/memory-model.md | pending: "Law D" enforcement that every primitive's regions declare a Kind |
| Wire format versioning | pending doc; covered by F3 | pending: "Law E" that Program::from_wire rejects unknown schema versions with clear error |
| Pass scheduling invariants | pending; F5 landing | pending: "Law F" that Pass::requires names must resolve |
| Target registration matrix | docs/targets.md | pending: vyre-ir/build.rs check that feature flags are consistent |

## Contradictions Between Audits

None detected yet. The dupe, org, and test audits agree on the root causes and disagree only on severity ranking — all route through the same rebuild plan.

## Honesty Section (what this rebuild does NOT fix)

- **"Infinite abstraction" marketing phrase** — the rebuild delivers substrate neutrality for the primitives Vyre ships. A consumer that invents a primitive Vyre has never seen must register it through `NodeKindRegistration`; that is open-world. But a consumer that wants their primitive to run on a backend Vyre does not ship still needs to write the backend. Abstraction is infinite in *type* (new primitive = new crate) but not in *implementation* (the primitive still needs a backend lowering).
- **Exhaustive verification only up to 16-bit** — the conformance certificate is genuine on bounded domains (8 and 16 bit) and degrades to committed-witness property testing on unbounded domains. A primitive whose correctness depends on 32-bit float precision cannot be exhaustively verified; the certificate reports its coverage honestly.

## Out of Scope by Design (permanently not Vyre's problem)

- **YARA compat** — `yaragpu` will be a downstream wrapper that composes Vyre primitives to execute YARA rules. Vyre does not ship YARA parsing.
- **Auto CPU↔GPU dispatch** — Vyre is a GPU substrate. `vyre-reference` is the verification oracle, not a dispatch target. Consumers that want CPU fallback build it themselves outside Vyre. GPU-below-1M-elements being slower than a CPU core is not a Vyre concern; it is a scheduling concern for whatever crate wraps Vyre plus a CPU executor.
- **Tensor DSL / numpy-style API** — `tensor-ml` and friends will be downstream wrappers. Vyre supplies primitives, not broadcast-ready tensors.
- **CPU execution of user programs** — Vyre does not compile Program to CPU code, does not execute user Programs on CPU, does not optimize for CPU caches. The only CPU code Vyre ships is `vyre-reference`, which exists solely to prove GPU backends correct.
