# RELEASE_GATE — the only remaining release gate

**Status.** 0.6 is not a release. It is a bar. 0.5 gets yanked if we can't clear it.

**Authority.** This is the active release gate and execution backlog.
Documentation precedence is defined in
[`docs/DOCUMENTATION_GOVERNANCE.md`](../docs/DOCUMENTATION_GOVERNANCE.md).
Older `MASTER_PLAN*` and `V7_*_PLAN` files are historical unless this
file explicitly delegates work to them.

**Gate criterion.** Surgec compiled on vyre executes arbitrary compute as rules and beats every competitor by **≥1000×** on real corpora, **zero stubs**, **zero known limitations**, **Torvalds-tier organization**, **20 impeccable rules** shipped at launch, **10 load-bearing innovations** that do things the competition cannot do at all.

**Law.** Nothing in this document is "out of scope." Every item lands or the plan extends. No deferrals, no "for next release." Every deletion, every fix, every innovation is part of the same ship — not a sequence of ships.

This file is the single source of truth for the current release gate. Fronts A–E describe *what release means*. Phases P0–P7 describe *the order the work lands*. Section F (line-level findings) pins every flaw to a file + line. Section I (innovations) enumerates the ten load-bearing capabilities that are categorically new. Section M is the master execution order.

---

## Contents

- [FRONT A — Vyre: incapable of being questioned](#front-a--vyre-incapable-of-being-questioned)
- [FRONT B — Surgec: arbitrary compute as rules, ≥1000×](#front-b--surgec-arbitrary-compute-as-rules-1000×)
- [FRONT C — SURGE the language itself](#front-c--surge-the-language-itself)
- [FRONT D — Flaw hunt (micro, macro, missing innovations)](#front-d--flaw-hunt-micro-macro-missing-innovations)
- [FRONT E — Gate closure mechanics](#front-e--gate-closure-mechanics)
- [PHASES P0–P7 — implementation order](#phases-p0p7--implementation-order)
- [SECTION F — line-level findings (every flaw, every file)](#section-f--line-level-findings-every-flaw-every-file)
- [SECTION I — the ten load-bearing innovations](#section-i--the-ten-load-bearing-innovations)
- [SECTION M — master execution order](#section-m--master-execution-order)
- [Working posture](#working-posture)

---

## FRONT A — Vyre: incapable of being questioned

### A.1 Zero-stub sweep (LAW 1 / LAW 9)
- Grep every `.rs` in vyre for `todo!`, `unimplemented!`, `panic!("not implemented")`, `panic!("TODO")`, `Ok(vec![])`, `Ok(())` where the body does nothing, empty match arms, silently returned defaults, no-op loops, `return Default::default()` where the function name promises work. Each: implement or prove obsolete and delete.
- Every `expect("Fix: …")` audited — real invariant or shortcut? If shortcut, replace with proper `Result`.
- Every `#[cfg(feature = "…")]` audited — does disabling weaken a guarantee? Traps get removed.
- Every `// known limitation` / `// FIXME` / `// TODO` / `// XXX` comment — implement or delete. No documented surrender (LAW 9).

### A.2 Organization sweep (LAW 7 — Torvalds standard)
- Every `.rs` file >500 lines split by responsibility. Start: `vyre-driver-wgpu/src/lib.rs` (745), `pipeline.rs` (679), then every other oversized file.
- Every directory answers exactly one question. No `utils/`, no `common/`, no `helpers/`.
- Every `pub` item deliberate. `#[doc(hidden)]` or private for everything not load-bearing.
- Per-crate `README.md` reads like a Linux subsystem doc: invariants, boundaries, contract, three worked examples, extension guide.
- Naming consistency sweep repo-wide — one name per concept (`dispatch` vs `execute`, `Program` vs `IR`, `*_ref` vs `*Reference`).
- Module dependency DAG — no cycles, no upward imports. `docs/crate-graph.dot` committed.
- Public surface snapshot test per crate (`cargo public-api` or equivalent). Accidental API changes fail CI.

### A.3 Hot-path O(1) proofs
Criterion benches under `benches/hot_path_*.rs` proving:
- Buffer pool: per-acquire work is O(1) in pool size. Flat scaling from 1 → 1M entries.
- Pipeline cache: hot-path is one atomic load.
- Validation cache: short-circuits before any traversal.
- DialectRegistry: frozen-index lookup sub-ns.
- Readback: persistent-mapped, page-locked staging — hidden under compute time.

### A.4 Per-op surface complete (every registered `OpEntry`/`OpDefRegistration`)
- Witness (`test_inputs` + `expected_output`) or load-bearing `UniversalDiffExemption`.
- CPU reference in `vyre-reference` as canonical spec.
- GPU dispatch via WGSL (wgpu).
- SPIR-V emission path.
- Photonic stub returning `HardwareUnavailable` with actionable message until hardware lands.
- Structural certificate entry.
- Parity-matrix coverage (byte-exact for non-F32, ULP-bounded for F32).
- Documented ULP/byte contract.
- Hostile-input fuzz (NaN, Inf, -0, subnormal, empty, max-size, malformed wire).
- Proptest invariant (associativity, commutativity where applicable, identity, idempotence, fixpoint).

### A.5 Naga lowering — zero holes
- Match-completeness audit on every `Expr` / `Node` / `UnOp` / `BinOp`.
- `Expr::Atomic` covers every `AtomicFunction`: Add, And, ExclusiveOr, InclusiveOr, Min, Max, Exchange{compare}, Subtract.
- `Expr::Subgroup*` covers the full subgroup API: Add, Mul, Min, Max, And, Or, Xor, Ballot, Shuffle, ShuffleXor, ShuffleUp, ShuffleDown, Broadcast, First, Any, All.
- Workgroup-shared (`BufferAccess::Workgroup` + `DataType::Shared`) lowers end-to-end — tiled matmul, FlashAttention-v2, cooperative scan, cooperative reduce, prefix-sum all compile and run.
- Region inline complete — no orphan regions, no dropped metadata.
- Constant fold exhaustive (no `_ => expr` fallthroughs where a concrete fold exists).

### A.6 Runtime pipeline innovations — verify each delivers its factor
Each item is both an existing claim and a bench target:
- GPUDirect Storage → VRAM truly zero-copy (tie to Innovation I.3).
- Megakernel fusion (N rules → 1 dispatch).
- Persistent-threads work-stealing scheduler.
- Multi-GPU work-stealing with straggler rebalance.
- Adaptive workgroup sizing with occupancy feedback (tie to Innovation I.6).
- Pipeline disk cache keyed on `(program wire hash, adapter fingerprint, wgpu version)`.
- Shadow execution: sampled CPU-reference diff fires a finding on divergence.
- Differential megakernel replay log.
- Multi-tenant megakernel with rule-applicability mask gating.
- Streaming hash-map hit compaction.
- Multi-op packed slots.
- `Program → Pipeline → Dispatch` end-to-end latency bench.
- `dispatch_borrowed` zero-copy vs owned-vec bench.

### A.7 SQLite / NASA / Linux / Chromium-grade testing
- Every op: unit + adversarial + proptest + bench + gap (LAW 5).
- Every subsystem: ≥24 h fuzzer before the gate is claimed clear.
- Every boundary: loom for concurrency, miri for UB on non-backend code.
- Every error path: test asserts the `Fix:` remediation is actionable and non-empty.
- Every public fn: doctest that compiles and runs.
- Every emitted WGSL: parsed by naga, validated, round-tripped through pipeline creation.
- Property: wire round-trip is identity for every `Program`.
- Property: canonicalize is idempotent and a fixpoint.
- Property: optimize preserves observable semantics (CPU-reference diff after == before).
- Chaos: random optimize passes composed in random order all produce semantically equivalent programs.
- OOM injection at every arena checkpoint surfaces an error, never panics.
- Poisoned-mutex recovery / actionable-error surface.
- Stress: 1M compile + dispatch cycle, zero leaked wgpu resources.

### A.8 Error surface
- Every error: `vyre-E####` code + `Fix:` remediation + doc page.
- `gap_error_code_catalog` test enforces catalog covers every emitted error.
- CLI output: no raw Debug — every message is an actionable sentence.

### A.9 Docs
- `ARCHITECTURE.md` is architecture, not a changelog.
- `CONTRIBUTING.md` — exact N-step add-an-op template.
- Per-op doc page generated from `OpDef` metadata.
- `VISION.md`: three paragraphs, zero weasel words.
- `THESIS.md`: engineering thesis (LLVM-for-GPU, conformance ratchet, composable primitives, CPU parity reference).
- `rustdoc --deny broken_intra_doc_links` green.

### A.10 Release engineering
- CI matrix green: Linux/macOS/Windows × stable/MSRV × default/all-features.
- `cargo-deny` passes (licenses, advisories, bans, sources).
- `cargo-semver-checks` passes vs previous tag.
- `cargo-public-api` snapshot reviewed per PR.
- Signed conformance cert artifact produced + published.
- Crate publish dry-run succeeds for every publishable crate.

---

## FRONT B — Surgec: arbitrary compute as rules, ≥1000×

### B.1 The compilation-target surface — every capability is a vyre intrinsic
Every capability below callable from a SURGE rule, compiles to vyre IR, fuses into one megakernel:
- **Decode/decompress**: gzip, deflate, brotli, zstd, lz4, snappy, base64, base32, hex, URL, HTML-entity, JSON-escape, UTF-8/UTF-16, punycode.
- **Hash**: BLAKE3, SHA-{1,256,512}, MD5 (compat), FNV-1a, xxHash.
- **Regex / n-gram / multi-pattern**: finite-automaton-on-GPU, hyperscan-class throughput (ties to Innovation I.9 cooperative DFA).
- **Tokenize / lex**: character-class DFAs generated from SURGE grammar declarations.
- **AST walk**: C, C++, Python, JS, TS, Rust. GPU-parsed, GPU-resident, GPU-analyzed (tie to Innovation I.7).
- **Dataflow / taint**: source → sink reachability, sanitizer barriers, `flows_to`/`sanitized_by`/`bounded_by_comparison` as rule predicates.
- **Dominator tree / CFG**: `dominator_tree`, `path_reconstruct` callable from rules.
- **Fixpoint drivers**: user-defined fixpoint loops dispatch on GPU until convergence (tie to Innovation I.8 incremental cache).
- **State machines**: SURGE states/transitions → GPU SIMT state update.
- **Graph traversal**: BFS/DFS/SCC over program graphs, CSR-backed (ties to Innovation I.5 exploit-graph).
- **Arbitrary algorithm**: if a rule author writes a function, surgec compiles it to vyre IR.
- **Heuristics**: scoring, weighting, thresholding, top-K, percentile — composable.
- **Exemption system**: rule-level exemption grammar. Thousands of exemptions compile to a perfect-hash GPU lookup.

### B.2 Multi-file / multi-folder / multi-repo context + dependency graph
- Tool-aware dep graph (Cargo, npm, pip, go.mod, Maven, Gradle, PNPM) as a vyre Program over CSR of packages → modules → symbols.
- Multi-repo: one scan spans repos linked by manifest. Taint crosses boundaries.
- Language-aware call graph: virtual, overload, trait-impl resolution for Rust.
- Import graph distinguishes re-exports, macros, re-imports, wildcard imports.
- Whole-program symbol table indexed for O(1) cross-file lookup from inside a rule.
- Incremental rebuild: change one file, only the affected slice recompiles.

### B.3 Battle-tested frontends (source + binary + network + package)
Surgec is the universal vuln/malware detection engine. Every artifact class a security researcher analyzes needs a first-class frontend. See `libs/tools/surgec/SCOPE.md` for the full catalog; the shipping minimum is:

**Source languages** (each: GPU-dispatched lex + parse + AST; corpus test on a real codebase with parse-success rate published in CI)
- **C** — ISO C + preprocessor-faithful + bitfields + `_Atomic` + VLAs + GCC/Clang extensions (SURGE-C grammar gen). Corpus: Linux kernel + Chromium C extension.
- **C++** — C++20 subset covering real-world usage: templates, concepts, coroutines, modules. Corpus: Chromium.
- **Rust** — published grammar + best-effort macro expansion. Corpus: top-100 crates.io.
- **Go** — Go spec grammar. Corpus: kubernetes + docker/moby.
- **Python** — 3.12-complete (match, except*, type-params). Corpus: CPython.
- **JavaScript / TypeScript** — ES2024 + TSX + JSX + module resolution. Corpus: npm top-1000 + DefinitelyTyped.
- **Ruby** — parser.y equivalent. Corpus: top-1000 gems.
- **PHP** — PHP 8.x grammar. Corpus: WordPress + Drupal + Symfony.
- **Java / Kotlin** — JLS + Kotlin spec. Corpus: top-100 Maven Central.
- **Swift** — Swift 5.x grammar. Corpus: Alamofire / SwiftNIO / SwiftUI samples.
- **C#** — C# spec grammar. Corpus: dotnet/runtime + top-100 NuGet.
- **Zig** — Zig spec grammar. Corpus: ziglang/zig.
- **Solidity / Move** — smart-contract grammars for Web3 vuln classes.

**Binary / bytecode frontends**
- **ELF**, **PE**, **Mach-O**, **WASM**, **DEX**, **APK**, **IPA**, **COFF**, **AR**, raw **firmware** blobs. Each parsable into a GPU-resident structural AST (sections, symbols, imports, exports, relocations, code segments) with a corpus test.

**Network / wire frontends**
- **HTTP/1**, **HTTP/2**, **HTTP/3**, **TLS records**, **DNS**, **JSON**, **XML**, **protobuf**, **GraphQL**, **gRPC**, **WebSocket frames**. Each parseable from raw bytes into a GPU-resident structural AST.

**Package / config frontends**
- **package.json**, **Cargo.toml/lock**, **go.mod/sum**, **pip requirements**, **Gemfile/Gemfile.lock**, **Maven pom.xml**, **Gradle**, **NuGet**, **Dockerfile**, **Kubernetes manifests**, **Terraform**, **CloudFormation**, **Ansible**. Each parsed, resolved against ecosystem metadata, exposed as a rule-queryable dependency graph.

### B.7 Frontend coverage (gate requirement)
Every frontend listed in B.3 ships with:
- GPU-dispatched parser driven by `surgec-grammar-gen` (or equivalent for binary/network formats).
- CPU reference parser against which the GPU parser is byte-identical on a corpus.
- Parse-success rate ≥ 99.5% on the declared corpus. Regressions fail CI.
- Error-recovery: every parse failure reports file + byte range + expected tokens. No silent empty-tree.
- A SURGE `lang:` gate — `rust:unsafe_block`, `go:init_import`, `dex:export_method`, `terraform:public_ingress`, etc. — so rule authors attach rules to frontend-specific predicates without adapter plumbing.

### B.4 The 20 impeccable launch rules (shipping minimum — stdlib is open-ended)
Surgec covers every vuln/malware class enumerated in `libs/tools/surgec/SCOPE.md`. The 20 rules below are the **minimum shipping batch** — the stdlib keeps growing forever. Every rule: real CVE-class, 100s of SURGE lines allowed, cross-file, dataflow-driven, zero false negatives on known corpora, measured FP rate, ships with positive + negative test corpus, latency budget, published finding rate.

1. SQL injection (taint → SQL driver, sanitizer-aware, ORM-aware).
2. Command injection (→ exec/system/subprocess, `shell=True` aware).
3. Path traversal (→ filesystem open, canonical-path aware).
4. SSRF (→ HTTP client, allowlist/denylist, DNS-rebinding aware).
5. Deserialization of untrusted input (pickle, YAML.load, Java ObjectInputStream, PHP unserialize).
6. Template injection (Jinja/Twig/ERB/Handlebars — source → render).
7. XXE (XML parser config + untrusted input).
8. Open redirect (→ 3xx `Location`).
9. Hard-coded credential across multi-file reachability (not just single-line regex).
10. Weak crypto primitive reachable from crypto boundary (MD5 / SHA-1 / RC4 / DES on security-relevant path).
11. Insecure random reachable from token/ID generator.
12. TOCTOU on filesystem (stat → open gap).
13. Double-free / use-after-free (Rust `unsafe`, C/C++ ownership).
14. Integer-overflow → allocation (CVE-class: `length * elem_size` without `checked_mul`).
15. Unbounded recursion from untrusted input.
16. ReDoS — regex compiled from untrusted input or vulnerable pattern on untrusted input.
17. Prototype pollution (JS object merge, deep-assign).
18. Log injection (user input → logger without sanitization, CRLF-aware).
19. Race condition on shared state (concurrent access without synchronization).
20. Authorization bypass — reachability from authenticated endpoint to resource without permission check.

### B.5 Benchmark harness — prove the 1000× on every cell
- **Corpora**: Chromium (C++), CPython (Python), DefinitelyTyped (TS), 10K top crates (Rust), real CVE-positive corpora.
- **Competitors**: Nuclei, Semgrep OSS, Semgrep Pro, CodeQL, Snyk, ripgrep (pattern baseline), Hyperscan (throughput baseline).
- **Metrics**: rules/sec, bytes/sec scanned, cold-start, warm-start, full-corpus wall-clock, peak RAM, peak VRAM, findings/sec, precision, recall.
- **Publication**: `docs/BENCHMARKS.md` + `docs/BENCHMARK.md` with reproducible harness, adapter fingerprint, hardware spec. Numbers regenerate on `cargo bench`.
- **Gate**: smallest winning factor across rules × corpora ≥ 1000×. No cherry-picking.

### B.6 End-to-end demo (the ship criterion)
- One SURGE file (the 20 launch rules).
- One CLI (`surgec scan <corpus>`).
- One binary, no-config happy path.
- One output: structured findings + signed conformance cert.
- Live on a real corpus (Chromium or npm registry or top-10K crates), faster than anyone.
- Recorded, reproducible. Third-party rerun lands within noise. Notes in `docs/DEMO.md`.

---

## FRONT C — SURGE the language itself

### C.1 Richness
- First-class variables with common-case type inference.
- Callable functions (local + stdlib).
- Generic / parametric rules.
- `match` on AST shapes.
- Rule composition via predicate import.
- Exemption grammar `except when <pred>` attachable anywhere.
- First-class sanitizer / source / sink declarations.
- User-authored fixpoint with monotonicity proof.
- Cross-language rules (e.g. Python callers of a C extension).

### C.2 Organization to Linux subsystem standard
- `libs/surge/` = language only (AST, lexer, parser, type system, zero runtime/GPU/vyre deps).
- `libs/tools/surgec/` = compiler only (SURGE AST → vyre IR, no runtime/IO).
- Runtime (scan engine) separated from `compile/`.
- Stdlib rules in `surgec/rules/stdlib/*.srg` — houses the 20 launch rules.
- Pyrograph absorbed.
- Frontends in `surgec/src/lang/{c,cpp,py,js,ts,rs}/` — one dir per language.
- Every public item has a doc page; every internal item documented in-source.

### C.3 Tooling
- `surgec check` — fast parse + type-check.
- `surgec compile` — SURGE → vyre IR.
- `surgec bench` — rule set vs corpus with timing.
- `surgec lsp` — jump-to-def, find-references, hover, diagnostics.
- `surgec fmt` — canonical formatter.
- `surgec fuzz` — random-walk grammar to find rule-compiler crashes.
- Each: test suite + doc page + `--help` that answers every question.

---

## FRONT D — Flaw hunt (micro, macro, missing innovations)

### D.1 Micro-flaw sweep
- `cargo clippy --all-targets --all-features -- -D warnings` across workspace — every warning fixed.
- `cargo fmt --check` green, config committed.
- `println!` / `eprintln!` in library code replaced with `tracing`.
- Every user-facing string audited for tone and precision.
- Every CLI flag documented in `--help` with a concrete example.
- Every TOML config option validated at load with line + key error.
- Every public type impls `Debug` / `Clone` / `PartialEq` / `Eq` where meaningful, `Serialize` / `Deserialize` when on-wire.
- Every public fn has one compiling / running doc example (`cargo test --doc`).

### D.2 Confusing-to-user sweep
- First-run: `cargo install vyre && vyre --help` (and same for surgec) tells a new user the next step.
- Every error includes a concrete next step (no "invalid program" without byte range + expected token).
- README starts with a 30-sec working example.
- Running a CLI with no args prints top-level command list, not an error.
- `prove` / `run` / `scan` failures include the specific failing `(backend, op)` or rule, not a bool.

### D.3 Missing-innovation sweep — the general surface
(The ten load-bearing innovations are in [Section I](#section-i--the-ten-load-bearing-innovations). Additional user-facing innovations below.)
- Live GPU kernel hot-reload — change a rule, re-dispatch without a cold rebuild.
- Rule-differential replay — yesterday's rules vs today's code → only newly-introduced findings.
- Explainer mode — given a finding, print the data-flow path + every taint-touched node.
- Confidence score per finding (path length / sanitizer proximity / sink specificity).
- Corpus-wide TF-IDF of rule hits to surface unusual patterns.
- Auto-suppression proposal — 10K-hit rule proposes a TOML exemption with a one-line description.
- Watch mode — re-scan on file change, streaming findings.
- Distributed mode — scan a repo across N machines sharing the pipeline disk cache.
- Headless dispatch — CI job runs `surgec scan` against a PR diff, posts findings as review comments.
- Rule-provenance chain per finding (BLAKE3 of rule source + vyre program hash + adapter fingerprint).
- Offline `.surge-bundle` (rules + compiled pipeline + conformance cert) runnable air-gapped.

---

## FRONT E — Gate closure mechanics

### E.1 How we know it's done
- This document: zero unchecked boxes.
- All tests all platforms green.
- Benchmark gate ≥1000× on every named competitor × corpus cell.
- Clippy clean workspace-wide with `-D warnings`.
- No `todo!` / `unimplemented!` / `panic!("not impl")` / `#[ignore]` anywhere.
- Public-API diff vs 0.5.x reviewed and documented.
- Conformance cert emitted / signed / verified in-process / cross-process / on a different adapter.
- End-to-end demo recorded, reproducible, notes in `docs/DEMO.md`.

### E.2 If the gate cannot close
- 0.5 yanks from crates.io.
- Repos archive until the gate closes.
- No new release tagged. No "0.6" ever ships unless every box above is checked.

---

## PHASES P0–P7 — implementation order

Critical path: **P0 → P2 → P1 → P4** (hygiene → real lowering → scan execution → 1000× proof). P3 unblocks P2 completeness. P5/P6/P7/P8 run alongside once prerequisites exist.

### Phase 0 — Workspace hygiene (Torvalds-level organization)
Rule: if it doesn't run, it doesn't exist.

- **P0.1** Move `libs/performance/matching/vyre/surgec-grammar-gen/` → `libs/surge/grammar-gen/` or `libs/tools/surgec/grammar-gen/`. Cargo path fix, zero new code. Acceptance: `cargo test -p surgec-grammar-gen` passes at the new location.
- **P0.2** Prune hollow workspace crates:
  - Delete `vyre-libs-extern` (13 LOC empty lib.rs).
  - Delete or move-to-examples `vyre-libs-template` (134 LOC scaffold).
  - Merge `vyre-pipeline` (96 LOC thin wrapper) into `vyre-driver`.
  - Merge `vyre-pipeline-cache` (247 LOC duplicate of `pipeline_cache.rs`) into `vyre-runtime`.
  - Keep `vyre-driver-photonic` (252 LOC) only if hardware path exists, else delete.
  - Acceptance: `cargo test --workspace --exclude vyre-frontend-c` passes.
- **P0.3** Delete dead surgec code: `surgec/src/taint.rs` (deprecated; taint lowers via vyre dataflow) and `surgec/src/lower/stub_vyre_libs.rs` (only after P2 replaces every callsite).
- **P0.4** Purge "pattern matching" from live code/docs:
  - `surgec/ARCHITECTURE.md:45` → "bridges scan results"
  - `surgec/ARCHITECTURE.md:88` → "DFA-based GPU string scanning"
  - `surgec/future_work/README.md:4` → "rule DSL"
  - `vyre/README.md:168` → "string scanning composes ops"
  - `vyre/README.md:256` → "scanning now composes ops in vyre IR"
  - `surge/SPEC.md:606` → "Not a string-scanning language"
- **P0.5** Consolidate `.internals/{archive,planning,plans}` → single `.internals/archive/` with a README listing each file's purpose. Organization only.

### Phase 1 — Surgec scan execution path
Deliverable: `surgec scan <rules_dir> <target_dir>` produces findings on GPU.

- **P1.1** Add `scan` subcommand to `surgec/src/main.rs`. Flags: `-o findings.json`, `--format sarif|json`. Pipeline: parse → `surgec::compile_paths()` → `WgpuBackend::acquire()` → `scan::collector` walk → `scan::decode` decode → per-file `backend.dispatch()` → `output::sarif` or `output::vyre`. ~200-300 LOC + new `scan/dispatch.rs`. Acceptance: `surgec scan corpus/rules/surge/ test_corpus/ -o out.json` produces valid JSON.
- **P1.2** Build `surgec/src/scan/dispatch.rs` — the GPU execution bridge. `pub fn dispatch_rules(backend, compiled, file_bytes) -> Result<Vec<Finding>>`. Prepare inputs per `ir_emit.rs` layout, iterate `compiled.rules`, one `backend.dispatch` per rule, decode outputs → `Finding { rule_name, severity, offsets }`. ~150 LOC.
- **P1.3** Wire `scan::collector::Collector` to the dispatch path: `fn scan_gpu(&self, backend: &dyn VyreBackend) -> Result<Vec<ProjectFinding>>`. ~50 LOC.

### Phase 2 — Unify the lowering pipeline
Deliverable: one lowering path; every SURGE expression → real vyre IR.

- **P2.1** Connect `ir_emit` to the v3 lowerer. `surgec/src/lower/mod.rs::lower_call()` — when the predicate is one of the 25+ scanner predicates complete in `compile/ir_emit.rs`, delegate to `ir_emit::compile_scanner_predicate()`. ~30 LOC routing.
- **P2.2** Wire every inert `Expr` variant at `surgec/src/lower/mod.rs:131-151`:
  - `Comparison { lhs, op, rhs }` → `Expr::cmp(op, lower(lhs), lower(rhs))` (~15)
  - `BindingRef(name)` → `Expr::var(name)` (~5)
  - `LabelRef(label)` → `Expr::load("labels", label_index)` (~10)
  - `Conditional { cond, then, else_ }` → `Node::If(lower(cond), lower(then), lower(else_))` (~15)
  - `List(items)` → buffer init (~10)
  - `Literal(lit)` → `Expr::u32` / `Expr::f32` / string const (~10)
  - `Aggregate(fn, items)` → reduce op sum/count/min/max (~20)
  - `Fixpoint(body)` → `Node::Loop` with convergence check (~30)
  - `Path(segments)` → struct field-access chain (~15)
  - `Motif(spec)` → AST subtree match → graph traversal (~25)
  - `Comprehension(gen)` → parallel map via `gid_x()` (~20)
  - Plus the 4 previously missing variants: `IsMember`, `LetIn`, `Quantifier`, `Arrow`.
  - Total ~175 LOC.
- **P2.3** Wire `flows_to` to `bitset_fixpoint`. `surgec/src/compile/predicates/flows_to.rs::FlowsToPredicate::lower()` — replace the `LoweringError` with real lowering via `vyre_primitives::fixpoint::bitset_fixpoint`. ~50 LOC.
- **P2.4** Delete `surgec/src/lower/stub_vyre_libs.rs` in the same PR that replaces the last `inert_program()` callsite.

### Phase 3 — Vyre backend readiness
Deliverable: every IR node surgec can emit, the wgpu backend lowers.

- **P3.1** Audit naga lowering coverage at `vyre-driver-wgpu/src/lowering/naga_emit/node.rs`. Walk every `Node::*` returning `Err(unsupported)` and determine if any surgec-emitted Program can produce it. Priority:
  - `Node::Loop` with dynamic bound (needed for fixpoint).
  - `Node::If` with else branch (needed for conditionals).
  - `Node::Region` pass-through (confirm `region_inline` handles it).
  - ~100 LOC for Loop lowering. Feeds A.5 completeness.
- **P3.2** Security-op test fixtures at `vyre-libs/src/security/*.rs`. Add `test_inputs` to each of the 7 OpEntries:
  - `flows_to`: CSR `(edges_from, edges_to)` + reached bitset (~30)
  - `sanitized_by`: CSR + sanitizer bitset (~30)
  - `taint_flow`: sink set + reached set (~25)
  - `dominator_tree`: predecessor lists + idom buffer (~30)
  - `bounded_by_comparison`: access + bound + idom (~25)
  - `label_by_family`: edge list + family map (~20)
  - `path_reconstruct`: parent array + target (~20)
  - Total ~180 LOC. Closes A.4 for security ops.
- **P3.3** Wire-format coverage for new Expr variants at `vyre-foundation/src/serial/wire/encode/put_expr.rs`. ~30 LOC per new variant. Required so compiled Programs round-trip to disk/wire.

### Phase 4 — The 1000× benchmark (ship criterion)
- **P4.1** Build `benches/competition/`:
  - `corpus/` = 10K varied-size representative files.
  - `rules/surgec/` = 5 SURGE rules exercising GPU advantage.
  - `rules/semgrep/`, `rules/codeql/`, `rules/nuclei/` = same 5 rules per competitor.
  - `run_*.sh` scripts timing each.
  - Five seed rules: `decode_scan` (deflate + base64 decode + scan decoded content), `ast_heuristic` (parse C → 50 heuristics), `dataflow_taint` (CFG + fixpoint reachability), `multi_layer` (3 decode layers + 1000-exception cross-reference), `crypto_audit` (hash every function body vs known-bad set).
- **P4.2** `surgec/benches/vs_competition.rs`: one `WgpuBackend::acquire()`, one `surgec::compile_paths()`, `bench_function("surgec_10k_files", ...)` iterating `surgec::scan::dispatch_all` over the corpus.
- **P4.3** Publish `docs/BENCHMARK.md` with hardware spec (GPU, VRAM, CPU baseline), rules used, corpus size, wall-clock surgec vs each competitor, throughput (files/sec, MB/s), speedup per rule. Gate: smallest cell ≥1000×. Numbers regenerate on `cargo bench`.

### Phase 5 — surgec robustness
- **P5.1** Wire scan-filter tags to rule metadata. `surgec/src/scan/filter.rs:75` — currently hardcodes empty slice. Connect to `rule.tags` from AST at `rule.rs:422`.
- **P5.2** End-to-end integration tests:
  - `tests/scan_e2e.rs` (compile → scan → verify findings)
  - `tests/scan_decode.rs` (base64 content → decoded findings)
  - `tests/scan_empty.rs` (empty dir → zero findings + no crash)
  - `tests/scan_big_file.rs` (100 MB → GPU handles)
- **P5.3** Delete `surgec/src/taint.rs` and `pub mod taint` from lib.rs. Dead once P2.3 lands real `flows_to` lowering.

### Phase 6 — Vyre infrastructure polish
- **P6.1** GPU differential test serialization. `vyre-driver-wgpu/tests/cat_a_gpu_differential.rs` — static `Mutex`/`OnceLock` to serialize GPU access across workspace test binaries, or run with `--test-threads=1`.
- **P6.2** Remove feature-unification leak from `xtask` — either move `xtask` to workspace `exclude` or restructure so `c-parser` doesn't bleed workspace-wide.
- **P6.3** C11 parser test fixtures at `vyre-libs/src/parsing/c/parse/structure.rs` — `c11_extract_functions` and `c11_extract_calls` need ≥24-byte buffers (currently 8).
- **P6.4** `vyre-frontend-c/src/elf_linux.rs` — replace host-stub `ET_REL` fallback with real GPU ELF emission OR escalate to CEO for explicit scope call (LAW 9).

### Phase 7 — Release quality (closing pass)
- **P7.1** Walk every `#[ignore]` vs `findings.toml`. Verify a finding entry exists. Either fix the test or document the concrete blocker.
- **P7.2** Walk every public-fn `///`. If "WIP" / "stub" / "not yet" / "placeholder" — implement or delete (LAW 9).
- **P7.3** `surgec/ARCHITECTURE.md` and `vyre/docs/ARCHITECTURE.md` accuracy pass — verify every referenced file path exists and every description matches reality.
- **P7.4** CI enforcement gates:
  - `cargo test --workspace`
  - `cargo clippy --workspace -- -D warnings`
  - The 3 composition-discipline gates.
  - `grep -r 'todo!' --include='*.rs' src/ | grep -v test` must be empty.

### Phase 8 — line-level finding closure
Every `F-*` in Section F lands or gets an explicit CEO scope call (LAW 9: no documented surrender without one).

### Phase 9 — innovation delivery
Every `I.*` in Section I lands. Each innovation carries a bench proving its claimed factor. No innovation ships on a README — it ships with numbers.

### Sizing (P0–P7, pre-innovations)
| Phase | New LOC | Delete LOC | PRs |
|---|---|---|---|
| 0 — Hygiene | 0 | ~500 | 3 |
| 1 — Scan Exec | ~400 | 0 | 2 |
| 2 — Lowering | ~260 | ~65 | 3 |
| 3 — Backend | ~310 | 0 | 3 |
| 4 — Benchmark | ~400 | 0 | 2 |
| 5 — Robustness | ~300 | ~80 | 3 |
| 6 — Vyre Polish | ~100 | 0 | 4 |
| 7 — Release | ~50 | ~200 | 3 |
| **P0–P7 total** | **~1,820** | **~845** | **23** |
| 9 — Innovations | ~1,690 | 0 | 10 |
| **Grand total** | **~3,510** | **~845** | **33** |

---

## SECTION F — line-level findings (every flaw, every file)

Source: line-by-line read of ~50 source files across `libs/surge/`, `libs/tools/surgec/`, `libs/performance/matching/vyre/`. Each finding has file + line. Findings that a Phase task already covers are cross-referenced; new findings get `F-*` IDs and map into Phase 8.

### A — Architectural cracks
- **A1 🔴 Two disconnected lowering pipelines.** `compile/compile.rs:153` calls `ir_emit::compile_scanner_predicate()`; `lower/mod.rs:177` ignores `ir_emit` and dispatches to `vyre-primitives` + `vyre-libs::security`. Two independent compilation backends for the same input. → **P2.1.**
- **A2 🔴 v3 lowerer returns empty Programs for 11 Expr variants.** `lower/mod.rs:131-151` — `Comparison | Literal | BindingRef | LabelRef | List | Aggregate | Fixpoint | Path | Motif | Conditional | Comprehension` → `stub_vyre_libs::inert_program()`. Missing entirely: `IsMember`, `LetIn`, `Quantifier`, `Arrow`. → **P2.2 (extended).**
- **A3 🟠 `lower_call()` catch-all silently produces empty Programs.** `lower/mod.rs:297` — `_ => stub_vyre_libs::inert_program()`. → **F-A3.**
- **A4 🔴 `scan-project` removed, no replacement.** `main.rs:278-289` tells users to use `warpscan` which doesn't exist. → **P1.1.**
- **A5 🟠 Security ops exempt themselves from GPU differential testing.** `vyre-libs/src/security/mod.rs:42-48` — seven `UniversalDiffExemption`s with "needs driver convergence loop" and no plan. → **F-A5.**
- **A6 🟠 `CompiledDocument` has no Programs inside.** `compile/types.rs:224-235` — `CompiledRule` has only `scanner_rule_name: Option<String>`, no direct Program handle. → **F-A6.**

### B — Dead code & stubs
- **B1 🟡 `stub_vyre_libs.rs` — 7 dead functions.** → **P2.4.**
- **B2 🟡 `taint.rs` entirely deprecated.** → **P5.3.**
- **B3 🟡 `sketch::compile_sketch_predicate()` always errors.** `compile/sketch.rs:301-309`. → **F-B3.**
- **B4 🟡 `output/mod.rs` only declares `pub mod vyre`** — `output/sarif.rs` orphaned. → **F-B4.**
- **B5 🟡 `ir_emit` is `pub(crate)` but has pub-fn surface.** `compile/mod.rs:233`. → **F-B5.**

### C — Logic & validation gaps
- **C1 🟠 Validation doesn't check most predicate types.** `compile/validate.rs:348-351`. → **F-C1.**
- **C2 🟡 `collect_referenced_zones()` returns empty.** `compile/validate.rs:396-402`. → **F-C2.**
- **C3 🟡 `collect_signals_in_predicate` covers 5 variants only.** `compile/validate.rs:373-392`. → **F-C1 (joint fix).**
- **C4 🟡 Filter tags never wired.** `scan/filter.rs:334-339`. → **P5.1.**
- **C5 🟡 `require_enabled` has no effect.** `scan/filter.rs:370-374`. → **F-C5.**
- **C6 🟡 `main.rs` `unwrap_or_default()` outside `deny(clippy::unwrap_used)`.** `main.rs:209,253-254`. → **F-C6.**
- **C7 🟡 Silent catch-all in `referenced_signals()`.** `compile/patterns.rs:429-430`. → **F-C7.**
- **C8 🟡 Lossy catch-all in `From<surge::Error>`.** `error.rs:146-149`. → **F-C8.**

### D — Performance & correctness
- **D1 🟠 `Program::new` called with `[1,1,1]`.** `lower/mod.rs:309`. → **F-D1.**
- **D2 🟠 `lower_call` ignores bindings + args.** `lower/mod.rs:154-165, 234-294`. → **F-D2.**
- **D3 🟡 `merge_programs` clones entire entry Arc vec.** `lower/mod.rs:323-330`. → **F-D3.**
- **D4 🟡 `absorbs()` missing dual absorption.** `compile/optimize.rs:415-420`. → **F-D4.**
- **D5 🟡 `magic_len as u8` unchecked cast.** `compile/applicability.rs:557`. → **F-D5.**

### E — Organizational
- **E1** grammar-gen location → **P0.1.**
- **E2** `future_work/` directory → **F-E2.**
- **E3** hollow crates → **P0.2.**
- **E4** `.internals/` sprawl → **P0.5.**
- **E5** missing `pub mod sarif` → **F-B4.**

### F — Naming & language
- **F-F1** remaining "pattern matching" refs → **P0.4.**
- **F-F2 🟠 `Legacy` variants still load-bearing** (8+ references: `Expr::Legacy`, `Legacy(Box<LegacyPredicate>)`, "legacy zone", "legacy spatial-proximity anchors"). → **F-F2.**
- **F-F3 🔴 `Predicate` vs `Expr` duality.** Both production, different lowering paths. → **F-F3.**

### G — Test coverage gaps
- **F-G1 🔴 No integration test compile → lower → dispatch on GPU.** Golden tests verify wire roundtrip only.
- **F-G2 🟠 No `CompiledDocument.save/load` roundtrip with real Programs.** `compile/bundle.rs` uses empty stubs.
- **F-G3 🟡 `batch_compile_corpus.rs` compiles but never dispatches.**
- **F-G4 🟡 `parity/cpu_eval.rs` CPU-side only.** No GPU parity counterpart.
- **F-G5 🟡 `conformance.rs` covers v1 score-based rules only.** No v3 `let/require/report`.

---

## SECTION I — the ten load-bearing innovations

Each innovation builds on infrastructure that already exists. Each ships with a bench proving its factor. No README-only innovations. Combined multiplicatively, these don't just beat the competition — they do things the competition cannot do at all.

### I.1 — GPU-fused decode → scan pipeline
**What exists.** `surgec/src/scan/decode.rs` decodes on CPU (base64/hex/URL-encoding/gzip). `vyre-libs::matching::dfa` Aho-Corasick on GPU. `vyre-primitives::text::char_class` byte classification on GPU. Megakernel ring buffer keeps data GPU-resident.

**The innovation.** Move decode into the GPU dispatch:
```
raw_bytes (VRAM)
  → GPU base64 decode → decoded_bytes (VRAM)
  → GPU DFA scan     → hit buffer (VRAM)
  → GPU rule eval    → findings (readback)
```
Zero host↔device copies between stages. Competitors round-trip through host memory every stage — they *cannot* pipeline decode+scan without a host copy. Megakernel ring buffer + Innovation I.3 zero-copy NVMe-DMA = pure VRAM pipeline.

**Work.**
- `vyre-libs::decode::base64` GPU composition — ~80
- `vyre-libs::decode::hex` — ~40
- `vyre-libs::decode::inflate` (hardest — parallel DEFLATE) — ~200
- Fused decode → scan slot chaining in megakernel — ~100
- **Total ~420 LOC.** Expected factor 5–10× on obfuscated content.

### I.2 — Cross-rule CSE via fused megakernel
**What exists.** `surgec/src/compile/fuse.rs` concatenates rule bodies. `vyre::optimize` runs canonicalize → region_inline → cse → dce. Today CSE operates *within* each fused rule arm, not across arms (fuse wraps arms in `Region` first).

**The innovation.** Run global CSE *before* `Region` wrapping. If 50 rules all check `call_to(@malloc_family)`, the vyre Program computes it once and every arm reads the cached result. **Link-time optimization for security rules.** LLVM does this for code; nobody does it for security analysis. Result: sub-linear rule scaling.

**Work.** Restructure optimization ordering in `fuse.rs`: inline → CSE → re-wrap. ~50 LOC. Expected factor 2–10× at ≥100 rules.

### I.3 — Zero-copy NVMe → GPU DMA via io_uring
**What exists.** `vyre-runtime/src/uring/stream.rs` `GpuMappedBuffer`, `AsyncUringStream`, `submit_nvme_passthrough` (IORING_OP_URING_CMD). The megakernel has an `io_queue` buffer (64 slots) declared but the host-side SQE-submission loop isn't wired.

**The innovation.**
```
NVMe SSD
  → io_uring SQE (read_fixed / uring_cmd)
  → kernel DMA into GpuMappedBuffer
  → megakernel picks up bytes from ring buffer slot
  → GPU decode → scan → evaluate → results
```
Three copies today (typical pipeline). Zero copies with this. Saturate PCIe Gen5 (~14 GB/s) with raw file data flowing directly into GPU compute. No security scanner on earth does DMA-direct file ingestion to GPU.

**Work.** Wire the host-side io_uring driver loop submitting SQEs based on `io_queue` entries. ~150 LOC in `vyre-runtime`. Expected factor 3× on I/O-bound workloads.

### I.4 — Neural-net suspicion pre-filter
**What exists.** `vyre-libs::nn` has real `relu`, `linear`, `softmax`, `attention`, `layer_norm`, `moe` — all Category-A compositions returning `vyre::Program`.

**The innovation.** Tiny network (~100 KB weights) over per-256-byte-window statistics (entropy, byte-frequency histogram, structural features). Outputs per-region suspicion score. High-suspicion regions get full rule evaluation; low-suspicion regions are skipped.
```
file bytes → GPU entropy + histogram → 256 features
  → linear(256→64) → relu → linear(64→1) → sigmoid
  → suspicion_map[region_id]
  → only evaluate rules where suspicion > threshold
```
Every other scanner runs every rule against every byte. Learned skip-predictor fused into the same GPU dispatch. False-negative rate controlled by threshold.

**Work.**
- `vyre-libs::security::suspicion_classifier` composition — ~100
- `surgec/src/compile/prefilter.rs` classifier dispatch — ~80
- Offline training + committed weights.
- **Total ~180 LOC.** Expected factor 10–100× on large mostly-benign files.

### I.5 — Exploit graph reconstruction on GPU
**What exists.** `vyre-primitives::graph::csr_forward_traverse`, `scc_decompose`, `path_reconstruct`, `toposort`. `vyre-primitives::fixpoint::bitset_fixpoint`. `surgec/src/compile/chains/` has chain composition on CPU as a post-processing step.

**The innovation.** Build the exploit graph *on GPU* as findings accumulate. When rule X fires at offset 100 and rule Y fires at offset 250, the GPU computes reachability between them using the CSR graph primitives already in vyre.
```
findings buffer (GPU)
  → build_graph → CSR adjacency (GPU)
  → csr_forward_traverse → reachability bitset
  → scc_decompose → exploit clusters
  → path_reconstruct → concrete attack paths
  → readback: structured exploit graph
```
Output: not "47 individual vulns" but "3 exploit chains, max blast radius 12 files." No scanner produces exploit graphs in real time.

**Work.**
- `surgec/src/scan/exploit_graph.rs` — CSR-from-findings + graph dispatch — ~200
- SARIF extension (`relatedLocations` / `codeFlows`) — ~80
- **Total ~280 LOC.** Qualitative leap (new capability).

### I.6 — Adaptive workgroup sizing
**What exists.** `ir_emit.rs` hardcodes `[64,1,1]`. `v3` lowerer hardcodes `[1,1,1]` (catastrophic; see F-D1). `build_program_sharded` defaults `[256,1,1]`. `VyreBackend` exposes capability query; `vyre-intrinsics::hardware` exposes subgroup size.

**The innovation.** Runtime auto-tune. Query subgroup size, max workgroup, SM count. Set `max(subgroup_size, 64)`. For rules with high register pressure, reduce to avoid spills.
```rust
fn optimal_workgroup_size(backend: &dyn VyreBackend, program: &Program) -> [u32; 3] {
    let subgroup = backend.capabilities().subgroup_size.unwrap_or(32);
    let max_wg   = backend.capabilities().max_workgroup_x;
    let locals   = program.entry.iter().filter(|n| matches!(n, Node::Let { .. })).count();
    let pressure = if locals > 24 { 64 } else { 256 };
    let size     = pressure.min(max_wg).max(subgroup);
    [size, 1, 1]
}
```
SmartNIC/mobile GPUs have subgroup 16; desktop 32/64; server 128. Hardcoding wastes or over-subscribes. Adapts to the hardware like a JIT.

**Work.** ~40 LOC in `surgec/src/scan/dispatch.rs`. Expected factor 2–4× on varied hardware.

### I.7 — GPU-resident AST as first-class buffer
**What exists.** `vyre-libs::parsing::c` has `c11_extract_functions`, `c11_extract_calls`, `gnu_builtins`, `inline_asm`. `vyre-primitives::parsing::ast_ops` + `ssa_dominance_scan` do phi-placement on GPU. `vyre-primitives::predicate::{call_to, arg_of}` + `label::resolve_family` exist.

**The innovation.** Treat the GPU-parsed AST as a first-class scan buffer. When a rule references `call_to(@malloc_family)`:
1. Dispatch `c11_extract_calls` → GPU-resident call-site buffer.
2. Dispatch `resolve_family("malloc_family", call_sites)` → match bitset.
3. Dispatch the rule body against the matched call sites.

The rule operates on semantic program structure, not byte offsets. **This is what makes surgec categorically different from a GPU grep.** Semgrep matches syntax. CodeQL builds a DB then queries it. Surgec builds the AST on GPU, analyzes on GPU, never round-trips until findings are ready.

**Work.** Emit a two-phase dispatch (parse preamble → rule body) from `surgec/src/compile/ir_emit.rs`. ~120 LOC. Expected factor ≥100× vs string matching for semantic rules.

### I.8 — Incremental fixpoint cache
**What exists.** `bitset_fixpoint` runs from scratch every time. 10 000 files → 10 000 full convergences even when the taint graph barely changed.

**The innovation.** Cache the converged fixpoint state in GPU-persistent memory. For the next file, warm-start from the cached state. No new edges → zero iterations.
```
File 1: 0        → iterate 12 → converged, cache
File 2: cached   → iterate 2  → converged (only new edges)
File 3: cached   → iterate 0  → no new edges, instant
…
File 10 000: amortized ~0.1 iterations/file
```
**Incremental computation for GPU fixpoint analysis.** Salsa/rustc do this for type-checking. Nobody does it for GPU taint analysis. Megakernel persistent memory is the perfect substrate — state survives between dispatches with no host↔device transfer.

**Work.** Warm-start variant of `bitset_fixpoint` loading from cached buffers (~60). Driver-loop convergence detection + re-dispatch skip (~40). **Total ~100 LOC.** Expected factor 10–100× on incremental scans.

### I.9 — Subgroup-cooperative multi-string scan
**What exists.** `vyre-libs::matching::dfa` does Aho-Corasick DFA scanning on GPU — each thread processes one byte position independently. `vyre-intrinsics::hardware::subgroup_shuffle` is wired. The megakernel lane structure is persistent-threads + subgroup-aware.

**The innovation.** Share DFA state between neighboring threads in the same subgroup via `subgroup_shuffle_down`. Thread `N` passes its DFA state to thread `N+1`. Speculative execution for DFA scanning: each thread starts with `state_init`, then corrects as the real state arrives via shuffle. After `log₂(subgroup_size)` rounds every state is correct. After Mytkowicz et al. "Data-Parallel Finite-State Machines" (ASPLOS 2014) — nobody has deployed it inside a persistent megakernel for security scanning.

**Work.** `vyre-libs::matching::cooperative_dfa` — new scan composition using shuffle-based state forwarding. ~150 LOC. Expected factor 4–8× on multi-string workloads.

### I.10 — Compile-time rule specialization
**What exists.** Every rule compiles to the same generic Program shape regardless of rule complexity. `compile/optimize.rs` + `compile/applicability.rs` + `compile/fuse.rs` exist.

**The innovation.** Specialize emitted Programs by rule shape.

| Rule Shape | Specialization |
|---|---|
| Single-signal count check | Collapse to 1 load + 1 compare. No loops. |
| All-of with N signals | Unroll N loads if N ≤ 8, loop if N > 8. |
| Zone-constrained scan | Pre-filter: skip workgroups outside the zone radius. |
| Taint/fixpoint rule | Emit dedicated convergence loop + warm-start (I.8). |
| Multi-layer decode | Fuse decode + scan into single dispatch (I.1). |

`gcc -O3 -march=native` for security rules.

**Work.** `specialize_program` pass in surgec examining the emitted Program pre-fusion. ~200 LOC. Expected factor 2–5×.

### Combined impact
| Innovation | Factor | Infra reuse | New LOC |
|---|---|---|---|
| I.1 GPU decode fusion | 5–10× on obfuscated | decode.rs + megakernel | ~420 |
| I.2 Cross-rule CSE | 2–10× at ≥100 rules | fuse.rs + vyre CSE | ~50 |
| I.3 Zero-copy NVMe→GPU | 3× on I/O-bound | io_uring + GpuMappedBuffer | ~150 |
| I.4 Neural pre-filter | 10–100× on large files | vyre-libs::nn | ~180 |
| I.5 Exploit graph | Qualitative leap | vyre-primitives::graph | ~280 |
| I.6 Adaptive workgroup | 2–4× on varied hardware | VyreBackend caps | ~40 |
| I.7 GPU-resident AST | ≥100× vs string matching | vyre-libs::parsing | ~120 |
| I.8 Incremental fixpoint | 10–100× on incremental | bitset_fixpoint | ~100 |
| I.9 Subgroup DFA | 4–8× on multi-string | subgroup_shuffle | ~150 |
| I.10 Rule specialization | 2–5× compile-time | optimize.rs + fuse.rs | ~200 |

These stack multiplicatively. Not 1000× faster — doing things the competition cannot do at all. I.1 × I.2 × I.3 × I.7 × I.9 alone gives a five-stage multiplicative factor before any of the other innovations stack on top.

---

## SECTION M — master execution order

The critical path is the chain whose completion dates determine the gate close date. Everything else runs alongside once prerequisites exist.

### Stage 0 — parallel hygiene & lowering kickoff (no hard prereqs)
Run in parallel. Each fronts by itself.

1. **P0.1** grammar-gen relocation
2. **P0.2** prune hollow crates
3. **P0.4** pattern-matching purge
4. **P0.5** `.internals/` consolidation
5. **F-E2** future_work → ARCHITECTURE / .internals
6. **F-C1**, **F-C2**, **F-C5**, **F-C6**, **F-C7**, **F-C8** validation + visibility hygiene
7. **F-B3**, **F-B4**, **F-B5** surgec dead-surface hygiene
8. **F-D1** v3 workgroup size `[1,1,1] → [64,1,1]` (quick, critical-perf)
9. **F-D4**, **F-D5** optimize + cast hygiene
10. **F-F2** delete Legacy variants (uncovers non-exhaustive matches downstream — drives F-F3 scope)
11. **P3.1** naga `Node::Loop` dynamic bound (unblocks **P2.2** Fixpoint variant)
12. **P3.2** security-op test fixtures (unblocks **F-A5** convergence lens)
13. **P3.3** wire-format tags for any new Expr variant added this cycle

### Stage 1 — unify lowering (critical path)
Depends on Stage 0 hygiene landing cleanly and P3.1/P3.2.

14. **P2.1** connect `ir_emit` to v3 lowerer
15. **P2.2** wire every inert Expr variant (15 total incl. `IsMember`, `LetIn`, `Quantifier`, `Arrow`)
16. **F-A3** replace `lower_call` catch-all with explicit `LoweringError::UnknownPredicate`
17. **F-D2** thread bindings + argument buffer names through `lower_call`
18. **F-D3** kill `merge_programs` quadratic clone
19. **P2.3** wire `flows_to` to `bitset_fixpoint`
20. **P5.3** + **P0.3** delete `surgec/src/taint.rs`
21. **P2.4** delete `stub_vyre_libs.rs` (same PR as last callsite replacement)
22. **F-A6** `CompiledRule` Program accessor
23. **F-F3** unify `Predicate` vs `Expr` (follow-up wave; big)

### Stage 2 — surgec scan execution (critical path)
Depends on Stage 1 emitting real IR.

24. **P1.2** build `scan/dispatch.rs`
25. **P1.1** add `scan` subcommand
26. **P1.3** wire `Collector::scan_gpu`
27. **P5.1** wire filter tags
28. **P5.2** end-to-end integration tests
29. **F-G1** compile → lower → dispatch GPU test
30. **F-G2** bundle roundtrip with real Programs
31. **F-G3** batch_compile_corpus dispatches
32. **F-G4** GPU parity companion to `parity/cpu_eval.rs`
33. **F-G5** conformance.rs v3 coverage

### Stage 3 — backend + vyre polish (parallel with Stage 2)
34. **A.1** zero-stub sweep
35. **A.2** organization sweep (file splits, READMEs, module DAG)
36. **A.3** hot-path O(1) proofs
37. **A.4** per-op surface complete (every registered op)
38. **A.5** naga lowering completeness audit
39. **A.6** runtime-innovation bench verification
40. **A.7** SQLite/NASA/Linux/Chromium testing pass
41. **A.8** error-surface catalog
42. **A.9** docs pass
43. **A.10** release engineering
44. **P6.1** GPU differential test serialization
45. **P6.2** xtask feature-unification leak
46. **P6.3** C11 test fixtures ≥24 bytes
47. **P6.4** vyre-frontend-c ELF linker — real emission or CEO scope call
48. **P7.1** `#[ignore]` vs findings.toml audit
49. **P7.2** stale doc comments
50. **P7.3** ARCHITECTURE accuracy
51. **P7.4** CI enforcement gates
52. **F-A5** real CPU↔GPU convergence lens for security ops (delete the 7 `UniversalDiffExemption`s)

### Stage 4 — surgec language richness (parallel with Stage 2+3)
53. **C.1** first-class variables, functions, generics, match-on-AST, rule composition, exemption grammar, sanitizer/source/sink, user-authored fixpoint, cross-language rules
54. **C.2** organization to Linux subsystem standard
55. **C.3** tooling: check/compile/bench/lsp/fmt/fuzz

### Stage 5 — compilation-target surface (prereq for the 20 rules)
56. **B.1** every capability as a vyre intrinsic + CPU ref + WGSL lowering + proptest: decode/hash/regex/tokenize/AST-walk/dataflow/dominator/fixpoint/state-machines/graph/arbitrary-algo/heuristics/exemptions
57. **B.2** multi-file / multi-folder / multi-repo context + dependency graph
58. **B.3** battle-tested C / C++ / Python / JS / TS / Rust frontends

### Stage 6 — innovations (parallel with Stage 5 wherever infra exists)
Order chosen so each innovation feeds the next:

59. **I.6** adaptive workgroup (smallest, unblocks I.1/I.9 perf claims) — ~40 LOC
60. **I.10** rule specialization — ~200 LOC (feeds I.1)
61. **I.2** cross-rule CSE — ~50 LOC (feeds I.1)
62. **I.7** GPU-resident AST — ~120 LOC (feeds I.4 features, I.5 graph nodes)
63. **I.1** GPU-fused decode → scan — ~420 LOC
64. **I.9** subgroup-cooperative DFA — ~150 LOC
65. **I.8** incremental fixpoint cache — ~100 LOC
66. **I.4** neural suspicion pre-filter — ~180 LOC (depends on I.7 feature extraction)
67. **I.5** exploit-graph reconstruction on GPU — ~280 LOC (depends on I.7, Stage-2 scan)
68. **I.3** zero-copy NVMe → GPU DMA — ~150 LOC

Each innovation PR carries a bench proving its claimed factor and a documentation note in `docs/BENCHMARK.md`.

### Stage 7 — the 20 launch rules
69. **B.4** write, corpus-test, FP/FN-measure, latency-budget, publish finding rate for each of the 20 rules (SQLi, cmd-injection, path-traversal, SSRF, deserialization, template-injection, XXE, open-redirect, hard-coded-cred cross-file, weak-crypto-reachable, insecure-random-reachable, TOCTOU, double-free/UAF, int-overflow→alloc, unbounded recursion, ReDoS, prototype-pollution, log-injection, race-on-shared-state, authz-bypass)

### Stage 8 — the 1000× benchmark (ship criterion)
70. **P4.1** corpus + 5 seed rules per competitor
71. **P4.2** criterion vs-competition harness
72. **P4.3** publish `docs/BENCHMARK.md` — gate ≥1000× smallest cell

### Stage 9 — end-to-end demo and gate
73. **B.6** end-to-end demo on a real corpus, recorded, reproducible
74. **E.1** every box above checked; public-API diff reviewed; signed cert verified in-process, cross-process, different adapter
75. **E.2** contingency: if gate does not close, yank 0.5 from crates.io

### Front D (micro-flaw / confusing-to-user / missing-innovation) runs continuously alongside every stage
- **D.1** clippy `-D warnings` never regresses, `cargo fmt --check` always green, every public fn has a doctest
- **D.2** first-run experience, error messages, CLI help, README 30-sec example, specific failure modes
- **D.3** the general user-facing innovations (hot-reload, diff-replay, explainer, confidence scoring, TF-IDF, auto-suppression, watch mode, distributed mode, headless dispatch, provenance chain, offline bundle)

---

## Working posture (self-directive)

- **No short-burst wakeups.** One continuous pass.
- **Audit and repair interleave** — no "audit now, fix later."
- **Split and land, never queue.** If a surface is too broad for one session, commit the next slice now.
- **"Tests pass" is not a stopping condition.** The stopping condition is "every box on this page is checked."
- **No celebratory summaries** (anti-flaunting). Report only when a stage completes end-to-end or an item requires CEO input.
- **No agent reliance.** Claude works alongside Gemini/Codex as peers, not as an orchestrator.
- **LAW 9.** No documented surrender without an explicit CEO scope call. "Out of scope" is never self-granted.

**Next action.** Stage 0 items 1–13 in parallel. Stage 1 critical path (P2.*, F-A3, F-D2, F-D3) starts the moment naga Loop-dynamic-bound (P3.1) and security-op fixtures (P3.2) land.
