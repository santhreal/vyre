# vyre MASTER PLAN — 2026-04-17

> Historical import. This file is retained as evidence of the
> 2026-04-17 audit state, but it is not the current plan of record.
> Use `../../../audits/RELEASE_GATE.md` for the active release gate and
> `../../../docs/DOCUMENTATION_GOVERNANCE.md` for precedence. Historical
> claims below that call this file the source of truth are superseded.

One source of truth for every rot, gap, and deviation-from-vision across the vyre workspace. This is the plan only. No execution. Every entry is a finding that must close before v0.4.0 publishes. Goal: **ONE sweep from here to published v0.4.0 — no shims, no half-migrations, no debt carried forward.**

## 0.0 Workspace reality (from `cargo check --workspace`, 2026-04-17)

- vyre-conform: **279 compile errors, 122 warnings**. Workspace does NOT compile. Every downstream finding is moot until this is green.
- vyre (core): compiles, **735 warnings** (mostly missing_docs + dead_code + pub-use glob).
- vyre-std, vyre-spec, vyre-build-scan: compile.
- Primary error clusters:
  - `vyre-conform/src/vyre-spec/string/tokenize/mod.rs:79-258` — `E0639` non-exhaustive struct constructions (34 sites). ConformSpec shape drifted under the rename; constructor must switch to builder.
  - `$OUT_DIR/archetypes_registry.rs:7` — `&'static dyn Archetype` doesn't impl `Archetype` (missing blanket impl on `&dyn`).
  - `$OUT_DIR/gates_registry.rs:17-19` — `BeamSearch` + `NullBackend` rows don't implement `EnforceGate`; build_scan is emitting rows for non-gates.
- Workspace layout realities not in the rest of the plan:
  - `vyre_conform::MutationClass` is the public product enum from `vyre-conform/src/vyre-spec/types/vyre-conform/mutation_class.rs`; the adversarial mutation extension trait is internal-only as `Mutator`.
  - `vyre-conform/src/` has **15 top-level dirs** (target is 8 per ARCHITECTURE). Extras on disk: `algebra/`, `backend/`, `backends/`, `comparator/`, `framework/`, `generated/`. `vyre-spec/` and `framework/` BOTH exist — the "rename spec → framework" is half-done.
  - `vyre-conform/src/backends/` = `mod.rs` only; O6 NOT removed.
  - `vyre-conform/src/generated/` contains `coverage_report.md`, `defender_corpus.rs`, `manifest.toml`, `ops.rs`, `primitive_ops.rs`; H10 NOT moved to `$OUT_DIR`.
  - `vyre-core/src/compiler/` exists with 10 files (`buffer_layouts`, `dataflow_fixpoint`, `dominator_tree`, `recursive_descent`, `string_interner`, `typed_arena`, `visitor_walk`, `wgsl/`, `wgsl_backend.rs`, `mod.rs`). The `ops/compiler_primitives/` dir is gone. Status: **partially moved**; either canonical new home or mid-move residue — must resolve (see C14).
  - `vyre-core/src/reference/` still present; C4 OPEN.
  - `vyre-conform/fuzz` is `exclude = ["vyre-conform/fuzz"]` in workspace Cargo.toml — NOT compiled under `cargo check --workspace`. Hidden rot.
  - 14 zombie branches from last session (`vyre/*`, `jules-rescue/*`, `backup/*`, `t17-*`, `temp-push-branch`) — unreviewed. Must be audited + deleted/merged.
  - 14 sibling submodules show dirty content — out-of-scope for vyre release but flag before any multi-repo push.

## 0.1 NO-SHIMS LAW (ABSOLUTE, applies to every finding below)

Every migration below completes as a **full API move**. The following are ALL banned — treat as LAW 9 evasion:

- **No alias re-exports.** `pub use old::Thing as NewThing` or `pub use newpath::Thing` from the old path is a shim. Move the definition, update every call site.
- **No `pub(crate) use oldmod as newname`.** `vyre-conform/src/lib.rs` currently has six of these (`framework::types::conform as types`, `spec as specs`, `adversarial::mutations::catalog as mutations`, `enforce::layers`, `meta::observe`, and `mod ops { pub(crate) use vyre::ops::*; }`). Delete every one and rewrite every internal caller to the canonical path.
- **No thin shim module.** `pub(crate) mod ir { pub use vyre::ir::*; }` is banned — callers import `vyre::ir::Foo` directly.
- **No "legacy" / "compat" / "old_" / "_v1" names.** Dying APIs are deleted in the same commit that introduces the replacement.
- **No `#[deprecated]` waiting rooms.** If the API is wrong, change it; if it's right, leave it. `#[deprecated]` on pre-1.0 alpha code is cowardice.
- **No dual-path for the same concept.** One canonical path per responsibility. `framework` vs `spec`, `backend` vs `backends`, `algebra` vs `proof/algebra`, `comparator` vs `proof/comparator` — pick one and delete the other in the same PR.
- **No "old" dir + "new" dir cohabitation.** Moves are `git mv` + edit call sites + commit in one atomic change.
- **No publish with unpublishable path-only deps.** `vyre-conform/Cargo.toml` build-dep on `vyre-conform-codegen` (publish = false) is a shim-by-Cargo. Either codegen publishes, or its logic inlines into `build.rs`.
- **No feature-gated CPU/GPU parity.** `parity-oracle` feature in `vyre-core/Cargo.toml` gates an execution path. Kill the feature, move the oracle to conform.

Every rename in the plan uses `git mv` + global-grep rewrite + delete old dir + commit in **one atomic step**. If a step leaves the workspace red, fix forward in the same PR — never land a half-move on main.

## 0.2 NUKE ORDER (executes before any other stage)

CEO directive 2026-04-17: "prefer compiler errors to the current bullshit." Every stub, alias, shim, "temporary" helper, and `#[cfg(not(..))]` fallback is deleted in a single sweep ACROSS THE ENTIRE WORKSPACE before any repair work begins. Compile errors after the nuke are the honest state — they name every real gap. We fix forward against the real gaps; we do not restore the shims.

Items to nuke, scope = the whole vyre workspace (core, conform, spec, std, build_scan, demos, examples, vyre-conform/fuzz, vyre-conform/xtask, vyre-conform/codegen, vyre-conform/sigstore):

1. **Every alias re-export.** `pub use X as Y`, `pub(crate) use X as Y`, `use X as Y;` inside library code. Replace every call site with the canonical name.
2. **Every thin-shim module.** A module whose body is `pub use other::*;` and nothing else.
3. **Every stub function body.** `todo!()`, `unimplemented!()`, `panic!("not implemented")`, `panic!("TODO")`, `unreachable!("not yet")`, empty `Ok(())` bodies where the name promises work, functions returning `vec![]` / `HashMap::new()` / `Default::default()` when the name promises computation.
4. **Every `// TODO`, `// FIXME`, `// placeholder`, `// stub`, `// temporary`, `// HACK` in non-test code.** Delete the comment AND the hollow body it sits in. If the surrounding function is hollow, delete the function. The compiler will list everyone who depended on it — fix those call sites.
5. **Every `#[allow(dead_code)]` in non-test code.** Either wire the item or delete it. No middle state.
6. **Every `#[allow(unused_imports)]` / `#[allow(unused_variables)]` in non-test code.**
7. **Every `#![allow(missing_docs)]` crate-level and module-level.** Every `//! Doc.` placeholder. Missing docs become compile warnings → CI failures.
8. **Every `#[cfg(not(feature = "gpu"))]` CPU fallback that silently skips a GPU test.** Per CLAUDE.md: GPU exists on every machine in the fleet. A "GPU-unavailable" fallback is a silent correctness regression. Delete the fallback; let the test fail loudly on GPU absence.
9. **`parity-oracle` feature in `vyre-core/Cargo.toml`.** Feature-gating CPU reference is a shim for "we haven't moved it yet." Delete the feature AND all `#[cfg(feature = "parity-oracle")]`. The reference interpreter moves to `vyre-conform/src/proof/reference/` (C4) — core never has a CPU execution path.
10. **`decode-only`, `hash-only`, `primitive-only` features in `vyre-core/Cargo.toml`.** These exist so somebody could ship a "minimal" vyre. Nobody ships that. Delete. One crate, one feature set, no à la carte gating.
11. **Every `pub use X::*` glob.** H9/Kimi-2. Explicit named re-exports only.
12. **`vyre-core/src/ops.rs` Cat-B CPU re-exports** (`CategoryAOp`, `CpuOp` from the non-test path). Gone — reference lives in conform.
13. **`vyre-conform/src/lib.rs` six internal aliases** (`types`, `specs`, `ops`, `mutations`, `layers`, `observe`). Gone. Callers import the canonical path.
14. **`vyre-conform/src/backends/mod.rs`** (O6 residue). Whole dir deleted.
15. **`vyre-conform/src/generated/` committed content** (H10). `git rm -r` the tree; `build.rs` writes to OUT_DIR.
16. **`vyre-core/src/compiler/`** — either fold into `vyre-core/src/ir/transform/` + `vyre-core/src/lower/wgsl/` (see C14), or delete if redundant. No third dir that "compiles IR".
17. **`framework/` vs `vyre-spec/`** — delete `framework/` top-level; content moves to `vyre-spec/` + `pipeline/`. All callers rewritten.
18. **Every `#[deprecated]` attribute on pre-1.0 code.** Pre-release deprecation is theater. Delete the deprecated API, update callers, ship one clean surface.
19. **Every `process::exit` in library code** (H14). Functions return `Result` or propagate.
20. **Every `.unwrap()` / `.expect("")` with empty or non-`Fix:` message in non-test code.** Replace with `Fix:`-prefixed message or propagate.
21. **`vyre-conform/fuzz` excluded-from-workspace marker.** Include it; let it fail under `cargo check --workspace`; fix forward.
22. **Dead-branch zombie state.** Every `vyre/*`, `jules-rescue/*`, `backup/*`, `t17-*`, `temp-push-branch` — audit contents against main; if unique work exists, cherry-pick or re-dispatch; otherwise `git branch -D` (with user confirm).

**Acceptance test for stage 0.2:** `grep -r "pub use .* as " vyre-conform/src vyre-core/src vyre-std/src vyre-spec/src vyre-build-scan/src` returns zero hits. `grep -r "todo!()\|unimplemented!()" --include=*.rs vyre-conform/src vyre-core/src vyre-std/src vyre-spec/src vyre-build-scan/src` returns zero hits in non-test paths. `cargo check --workspace` may be red — that is fine and expected; the red is the real backlog.

## 0.4 CONFORM IS MAINTAINER-ONLY — audience rule, not quality rule

CEO directive 2026-04-17 (clarified): "cold/hot only refers to what the average user uses. it doesn't mean maintainers should have to live in shit."

Conform is **maintainer-facing surface**, not end-user-facing surface. End users consume `vyre` + `vyre-reference` + a backend (`vyre-wgpu` today). Maintainers — backend implementers, spec authors, release engineers — consume `vyre-conform`. That defines the **audience**. It does NOT soften the **quality bar**.

Every LAW still applies to conform in full:

- LAW 8 is absolute — every finding, low or high, fixed. No "cold path exemption." Maintainers get the same pristine surface end users get.
- Every allocation fix (Kimi-4), every concurrency fix (Kimi-3), every determinism fix (Kimi-6), every `Fix:`-prefix fix (Kimi-8), every glob-removal fix (H9/Kimi-2), every SemVer rename (Kimi-1), every must_use (Kimi-5), every file split (H5), every path flatten (H6) applies inside `vyre-conform/` exactly as it does inside `vyre/` or `vyre-reference/` or `vyre-wgpu/`.
- Certificate determinism and byte-identity are the functional contract. Perf inside conform is a quality contract — a slow certify run wastes maintainer time, and maintainer time is not cheap.
- The NO-SHIMS rule (§0.1) and NUKE ORDER (§0.2) apply to conform identically.

What the audience rule **does** change:

1. **No runtime consumer depends on `vyre-conform`.** If a non-conform crate pulls conform in `[dependencies]` (not `[dev-dependencies]`), that is a finding. End-user apps never transitively load conform.
2. **`vyre-conform`'s README** carries a banner: "Maintainer harness. For runtime, consume `vyre` + `vyre-reference` + a backend." No other documentation consequence.
3. **`vyre-core/src/vyre-conform/` module inside `vyre` core — DELETE.** Core must not know conform exists. The `pub mod conform` entry in `vyre-core/src/lib.rs:105` is a backwards shim; nuke it with §0.2.
4. **`vyre-conform/xtask`** stays `publish = false` as documented dev-tooling, relocated to workspace-root `xtask/` per cargo community convention.
5. **`vyre-conform/sigstore`** is a product feature, not conform-internal. Rename to `vyre-sigstore`, move to workspace root, publish. Certificate verification must work without pulling all of conform.
6. **`vyre-conform/codegen`** inlines into `vyre-conform/build.rs` so conform publishes without a `publish = false` build-dep.
7. **`test-infra/mutations/`** + **`test_catalog.toml`** move under `vyre-conform/rules/mutations/` and `vyre-conform/rules/test_catalog.toml`.

## 0.5 FINAL CRATE LAYOUT — the seven crates we ship

One crate, one purpose. This answers §8.a and §8.b decisively.

| Crate | Purpose | Pulls | Audience | Published? |
|-------|---------|-------|----------|------------|
| `vyre` | IR + ops catalog + lowering + `VyreBackend` trait. Backend-agnostic. | `vyre-spec` | end user | crates.io |
| `vyre-spec` | Frozen data contracts: `ConformSpec`, `AlgebraicLaw`, `Category`, `Invariant`, `IntrinsicTable`. 5-year SemVer. | — | end user | crates.io |
| `vyre-reference` | **NEW.** Pure-Rust CPU reference interpreter. One implementation per op. Serves (a) conform parity oracle, (b) downstream `.execute_cpu()` fallback for small data, (c) property-test double for consumers. | `vyre`, `vyre-spec` | end user | crates.io |
| `vyre-wgpu` | **NEW.** wgpu backend. Implements `VyreBackend`. Owns `runtime/`, device/queue/buffer-pool/pipeline-cache, and the wgpu-specific lowering glue. | `vyre`, `vyre-spec`, `wgpu` | end user | crates.io |
| `vyre-std` | Higher-level composites: DFA assembly, Aho-Corasick construction, compositional arithmetic, regex → NFA → DFA. | `vyre`, `vyre-spec` | end user | crates.io |
| `vyre-sigstore` | **RENAMED from vyre-conform/sigstore.** Keyless sigstore signing/verification for conformance certificates. Small, focused, publish it. | `vyre-spec` | end user + maintainer | crates.io |
| `vyre-conform` | Maintainer harness. Pulls every other crate. Runs certification. Dev-tool — not a runtime dep for end-user apps. Quality bar identical to end-user crates. | all of above | maintainer | crates.io (dev-tool banner) |
| `vyre-build-scan` | build.rs filesystem scanner. | — | build-time only | crates.io |

Workspace-only (not published): `xtask/` (dev tasks). Everything else under workspace root that isn't in this list gets deleted by §0.6.

Answer to §8.a (C4 destination) — **`vyre-reference`, a new crate, NOT `vyre-conform/src/proof/reference/`**. Rationale: CPU reference is useful beyond conform (small-data fallback, downstream property-test oracle, teaching tool). Putting it inside conform's private tree buries it.

Answer to §8.b (C8 destination) — **`vyre-wgpu`, a new crate**. Rationale: "one purpose per crate." Keeps `vyre` pure IR/ops and lets the next backend (CUDA/Metal/Vulkan-native) slot in as a sibling `vyre-cuda` / `vyre-metal` without touching core.

## 0.6 Workspace-root nuke list (previously uncataloged rot)

Deep scan 2026-04-17 surfaced non-crate debris at the workspace root. Each item below is deleted outright unless it has a defined home in §0.5. No dry-run; `git rm -r` is the answer.

- `archive/RELEASE_PLAN.md`, `archive/RELEASE_PLAN_V2.md`, `archive/coordination/`, `archive/README.md` — historical planning artifacts. **Delete.** Plan state lives in this MASTER_PLAN.md only.
- `tasks/00-index.md` through `tasks/05-trust-signals.md` — superseded by this plan. **Delete.**
- `docs/audits/` — 19 historical audit reports (agent_damage_audit, architecture_deep_audit, arith-kimi-AUDIT, catalog_audit, CONFORM_ARCHITECTURE_AUDIT_20260415, conform-gate-kimi-AUDIT, conform_stub_audit_pc1, dfa_minimize-kimi-AUDIT, docs-kimi-AUDIT, duplicate_neg, duplication_audit_2026_04_16, hash-kimi-AUDIT, ir-validate-wire-kimi-AUDIT, perf-kimi-AUDIT, primitive-laws-kimi-AUDIT, regex_to_nfa-kimi-AUDIT, toml_loader_naming, workgroup-primitives-kimi-AUDIT). Every finding worth keeping is folded into this plan. **Delete everything except MASTER_PLAN.md and the `audits/` dir itself.**
- `docs/internal/`, `docs/internals/`, `docs/generated/`, `docs/fossil-record.md`, `docs/migration.md`, `docs/planning/`, `docs/release/`, `docs/RELEASE_NOTES_v0.4.0-alpha.2.md` — audit each; generated + internal + premature release notes **delete**; keep only `docs/thesis.md`, `docs/roadmap.md`, `docs/stability.md`, `docs/trust-model.md`, `docs/OPS.md`, `docs/PRIMITIVES.md`, `docs/santh-standard.md`, `docs/support.md`, `docs/faq.md`, `docs/RELEASE_CHECKLIST.md`, `docs/architecture.svg`. Any doc that's stale rewrites or deletes on the spot.
- `scripts/fix_missing_docs.py`, `scripts/generate-ops-md.sh`, `scripts/check-ops-md.sh`, `scripts/ops-md-check.yml`-referenced helpers, `scripts/adversarial-cycle*.{sh,py}`, `scripts/dispatcher-templates/`, `scripts/fetch-regressions.sh`, `scripts/vyre-dispatcher.py`, `scripts/sleep-allowlist.txt`, `scripts/workflow-permission-*.txt` — every Python migration one-off, every dispatcher artifact, every "allowlist for lint" file. **Delete.** The lint rules move into the lint code itself; migration scripts are one-shot and gone. Keep only `scripts/publish-dryrun.sh` (used by release), `scripts/apply-branch-protection.sh` (used by repo setup), and `scripts/run-benchmarks.sh` (wired into bench.yml).
- `scripts/blake3-hasher/` subcrate — purpose unclear. If it's used by `build.rs` somewhere, inline. If not, **delete**.
- `tests/launch_smoke_test.rs` at workspace root — relocate under the crate it smoke-tests OR delete. No loose root-level tests.
- `target/` — confirm `.gitignore` covers it at every crate level; agent commits have previously leaked target artifacts.
- `docs/audits/audits/` nested dir if it exists — **delete** unconditionally.
- `Cargo.lock` — committed (correct for a workspace with binaries), but verified that it doesn't record path deps for deleted crates.

## 0.7 Core public surface audit (from `vyre-core/src/lib.rs`)

Current `pub mod` in `vyre-core/src/lib.rs`: **11 modules** (`backend`, `conform`, `engine`, `error`, `ir`, `lower`, `match_result`, `ops`, `compiler`, `reference`, `runtime`).

After the nuke + crate split, `vyre`'s public surface reduces to exactly:

- `pub mod ir` (Program, validate) — the IR data model.
- `pub mod ops` (op trait, op catalog) — the extensible op set.
- `pub mod lower` (WGSL emission for in-crate use; re-exported to `vyre-wgpu`) — keep public for backend authors.
- `pub mod error` — canonical error types.
- `pub use backend::{BackendError, DispatchConfig, VyreBackend}` — the backend trait + glue. `backend` itself is `pub(crate)`; only the trait leaks.
- `pub use ir::{Program, validate}` — re-exported for ergonomic `vyre::Program`.
- `pub use match_result::Match` — re-exported.
- `pub use error::{Error, Result}` — re-exported.

Everything else goes:

- `pub mod conform` → **DELETE** (see §0.4 rule 5).
- `pub mod engine` → **MOVE to `vyre-wgpu`** or `pub(crate)` if engine is pure IR transform (then sits in `core`). Audit each subsystem: `dataflow/`, `decode/`, `decompress/`, `dfa/` are IR-level algorithms; `prefix.rs`, `token_match_filter.rs`, `tests.rs` likely public. Per-subsystem decision required.
- `pub mod compiler` → **MOVE to `vyre-core/src/ir/transform/compiler/` as `pub(crate)`**. Compiler primitives are IR infrastructure; consumers use `vyre::Program::optimize()`, not the primitives directly.
- `pub mod reference` → **MOVE to `vyre-reference` crate** (answers C4).
- `pub mod runtime` → **MOVE to `vyre-wgpu` crate** (answers C8).

Acceptance: `vyre`'s `pub mod` list is `ir`, `ops`, `lower`, `error`. Four modules. Four re-exports at the root. Zero anything else.

## 0.3 Release-beyond-this-plan checklist (items to close before v0.4.0 final)

These are gaps found in the 2026-04-17 deep scan that the previous plan did not enumerate. Each is a release blocker under the one-sweep rule.

1. **`cargo-public-api` baseline snapshot** committed at `docs/public-api/vyre.txt`, `…/vyre-conform.txt`, `…/vyre-spec.txt`, `…/vyre-std.txt`, `…/vyre-build-scan.txt`. CI fails if the snapshot diffs without a corresponding `CHANGELOG.md` bump.
2. **`cargo-semver-checks`** wired into release CI. Runs on every tag.
3. **Per-crate `CHANGELOG.md`** following Keep-a-Changelog format. Every `pub` change lands with a changelog entry in the same PR.
4. **Per-crate `LICENSE-APACHE` + `LICENSE-MIT`** at crate root (docs.rs and crates.io require them or inherited). vyre-std currently missing both.
5. **`[package.metadata.docs.rs] all-features = true, rustdoc-args = ["--cfg", "docsrs"]`** on every publishable crate. Verified on vyre; missing on conform, std, spec, build_scan.
6. **Workspace-inherited `edition`/`rust-version`/`license`/`authors`/`repository`/`homepage`** on every crate (`vyre-std/Cargo.toml` hardcodes these).
7. **`cargo-deny.toml`** at workspace root with license allowlist + advisory-db gate + source allowlist + banned-crate list.
8. **`cargo-udeny`** run in CI against every crate.
9. **Loom CI target** invoking `cargo test --test loom_*` with `RUSTFLAGS=--cfg loom` on every concurrency primitive in conform.
10. **Miri CI target** running `cargo miri test` on any `unsafe` block surviving the nuke (should be exactly one: the `wgpu::Device::create_pipeline_cache` site).
11. **Coverage target** via `cargo-llvm-cov` with a minimum floor (SQLite aspires to 100%; start gate at 85% lines / 75% branches for conform, increase each release).
12. **`#![deny(missing_docs)]` on every publishable crate's lib.rs** after doc-sweep. Currently `#![warn(missing_docs)]` — promote to deny.
13. **`#![deny(warnings)]` on every publishable crate's lib.rs under a `strict` feature** + CI runs `cargo build --features strict`. Forces zero-warning shipping.
14. **GitHub Actions workflows** audited: `.github/workflows/ci.yml` + `release.yml` + `bench.yml` + `coverage.yml` + `public-api.yml`. Each green on main before tag.
15. **Per-crate `README.md`** with installation, three consumption modes (tool / lib / subcrate), example, feature table, MSRV, license. Today vyre-spec + vyre-build-scan likely skeletal.
16. **`rules/` Tier-B TOML directory** verified complete: at minimum every op in `vyre-core/src/ops/` has a matching `rules/op/<category>/<op>.toml` shell (community can extend). Per CLAUDE.md: hardcoded lists banned.
17. **Bench matrix**: every primitive in `benches/primitives_showcase.rs` runs on N=1K/10K/100K/1M with actual GPU dispatch + readback (P7). CI posts a gist with results on every main push.
18. **`vs_cpu_baseline.rs` bench un-`#[ignore]`ed** and asserts a minimum speedup per primitive (e.g. `assert!(gpu_ns * 10 < cpu_ns)`).
19. **`vyre-conform/sigstore`** + **`vyre-conform/xtask`** + **`vyre-conform/codegen`** decisions: either each is publishable and documented, or each collapses into the parent. Mixed publishable-vs-non-publishable subcrates are a shim by Cargo.
20. **`examples/hello_vyre`** compiles against the final 10-item public API and is documented in the README as the canonical entry point.
21. **`demos/rust_lexer_gpu` + `demos/rust_parser_gpu`** compile, run green, and have `RESULTS.md` documenting measured throughput. If they can't reach a real benchmark number, delete them — no "demo that kind-of works."
22. **`tasks/`, `test-infra/`, `archive/`, `scripts/`** dirs at workspace root audited: every file either has a documented purpose in `docs/` OR is deleted. No mystery tooling at root.
23. **Deterministic cert hash**: the certificate's `monotonic_sequence` is replaced by `blake3(canonical_inputs)`; two runs with identical inputs produce byte-identical certs (Kimi-6).
24. **`RELEASE.md`** at workspace root documenting the topological publish order (§7 Stage 7 of this plan) + the exact sequence of commands.
25. **`SECURITY.md`** names a published GPG key + contact + triage SLA.
26. **`CITATION.cff`** version field synced with `Cargo.toml` workspace version at tag time.
27. **`CODE_OF_CONDUCT.md`** + `CONTRIBUTING.md` reviewed by user before public release.
28. **Submodule-drift gate**: the 14 sibling submodules showing dirty content must be clean (either committed-in-their-own-repos or discarded) before a vyre tag push that touches the parent Santh repo.
29. **Zero shims in Cargo.toml**: no `path = "..."` to a `publish = false` crate that survives on crates.io (C19b — vyre-conform/codegen build-dep).
30. **Post-nuke compile count**: record the red-error count after §0.2. Track it weekly; it must reach zero before tag. No unreviewed restoration of deleted items.

Inputs merged:
1. User audit #1 — CRITICAL C1-C9, HIGH H1-H12, MEDIUM M1-M11, PERFORMANCE P1-P7 (39 findings).
2. User audit #2 — CRITICAL C10-C13, HIGH H13-H20, MEDIUM M12-M18, PERFORMANCE P8-P10 (18 findings).
3. User audit #3 — ORGANIZATION O1-O21 (migration graveyard, 21 findings).
4. Kimi audits 1-10 — SemVer, Cat-B, concurrency, allocation, must_use, non-determinism, fuzz coverage, error-message Fix-prefix, op-registry completeness, publish readiness.
5. Direct filesystem scan — 2228 .rs files, 189,288 LOC, 30+ files >500 LOC, 30+ dirs deeper than 4 levels, ~20 `pub use X::*` globs.

## 0. The vision — what every finding is measured against

- README: "There is no bytecode VM, no opcode interpreter, and no execution path that bypasses IR."
- README: "Determinism is achieved via restriction, not elimination."
- README: "Every layer is a strategic black box for the layer above."
- ARCHITECTURE: **exactly 10 public items** in `vyre-conform`: `certify`, `Certificate`, `Violation`, `VyreBackend`, `Finding`, `EnforceGate`, `Oracle`, `Archetype`, `MutationClass`, and the `vyre_spec::{Category, AlgebraicLaw, ConformSpec}` re-export.
- ARCHITECTURE: **8 modules** per crate — `spec`, `proof`, `enforce`, `pipeline`, `generate`, `verify`, `adversarial`, `meta`.
- ARCHITECTURE: responsibility dirs flat; **absolute max depth 4** under src/.
- ARCHITECTURE: no `util/`, `helpers/`, `misc/`. One file = one responsibility.
- ARCHITECTURE: filesystem IS the registry. No central REGISTRY/ARCHETYPES/etc. arrays.
- ARCHITECTURE: frozen traits = 5-year SemVer. Signatures never change after publish.
- ARCHITECTURE: every Err starts with `Fix: `. No swallowed errors.
- SANTH_STANDARD: `explicit module list` is the auto-registration mechanism.
- THESIS: `certify()` runs parity, law, mutation gate, AND adversarial gauntlet. No skipped layers.
- LAW 1: no stubs. LAW 8: every finding critical at internet scale. LAW 9: never weaken a doc to match broken code.

Anything in the tree that deviates from the above is a finding below.

## 1. Inventory at planning time

- `vyre-core/`, `vyre-conform/`, `vyre-std/`, `vyre-spec/`, `vyre-build-scan/` under `libs/performance/matching/vyre/`.
- 2228 `.rs` files, 189,288 LOC across `src/` of all five crates.
- Files >500 LOC: 30 (worst: `vyre-conform/src/enforce/enforcers/float_semantics.rs` at 2730 LOC).
- `mod.rs` files >400 LOC: 22 (worst: `vyre-conform/src/algebra/audit/mod.rs` at 775 LOC).
- Directory depth >4: 30+ (worst: `vyre-conform/src/verify/properties/tests/declared_laws/demorgan/checker/` at 8).
- `pub use X::*` globs: 20 occurrence sites.
- `#![allow(missing_docs)]` + `//! Doc.` placeholders across core: 20+ files.
- `as <numeric>` casts (likely silent truncation): hundreds; audit every one at `len()` / slot / shader-uniform boundaries.
- `.unwrap()` / `.expect()` / `panic!()` / `process::exit()` in non-test code: hundreds; many missing `Fix:` prefix.
- Conform top-level dirs today: `adversarial, algebra, backend, comparator, enforce, framework, generate, lib.rs, meta, pipeline, proof, spec, verify` (13; target 8).

## 2. Critical findings (block publish; compile or SemVer-freeze hazards)

### 2.1 Public API shape (C1, C2, H1, H2)

- **C1 — `pub fn certify(backend: &dyn WgslBackend, …)`** (`vyre-conform/src/pipeline/certify/mod.rs:255`). Must be `&dyn vyre::VyreBackend`. Internal machinery currently threads `&dyn WgslBackend` through ~20 call sites (`pipeline/certify/parity.rs`, `engine.rs`, `ops/mod.rs`, `streaming/orchestration/mod.rs`, `streaming/batch_execution/mod.rs`, `algebra/gpu_checker/mod.rs`, `meta/canary/mod.rs`, `pipeline/streaming/test_support.rs`, `certify/tests.rs` mocks). Refactor every call site to take `&dyn vyre::VyreBackend` and use `backend.dispatch(&program, &inputs, &vyre::DispatchConfig::default())` instead of the WGSL-string path. WgpuBackend's `VyreBackend::dispatch` impl internally lowers the Program to WGSL. No adapter/shim — pure Program flow.
- **C2 — `WgslBackend`** is already `pub(crate)` in `vyre-conform/src/lib.rs:157`. Keep it that way. The public surface must not re-export it.
- **H1 — 69-item public API** in `vyre-conform/src/lib.rs`. Reduce to the 10 items listed in §0. Every other `pub use` demoted to `pub(crate)`. Callers inside the crate use the internal path; external consumers use only the 10.
- **H2 — Trait signatures** (Archetype uses `instantiate` returning Vec, Enforcer uses const ID + `Result<(), Finding>`, Oracle uses `check`). Rewrite to ARCHITECTURE shape: `Archetype::materialize(&ConformSpec) -> Option<TestInput>`, `EnforceGate { fn id/name/run }`, `Oracle { fn id, kind, applicable_to, verify }`. Update every impl.

### 2.2 Bytecode VM and CPU interpreter (C3, C4, M1)

- **C3 — `vyre-core/src/bytecode.rs`** (637 LOC stack VM). Already deleted 2026-04-17 in commit `1d920554a5`. Verify Cat-B tripwire still flags any reintroduction and the conform tripwire text_scan list names `bytecode::Program` and `bytecode::Instruction` as forbidden (it does).
- **C4 — `vyre-core/src/reference/`** (full CPU interpreter). Currently gated behind the `parity-oracle` feature. User rejected feature-gating: must MOVE out of core entirely. Target: fold into `vyre-conform/src/proof/reference/` so the conform parity oracle owns it. Core never carries an execution path.
  - Callers that currently use `vyre::reference::…`: `vyre-conform/src/oracles/law_independent.rs`, `vyre-conform/src/bin/verify-cert.rs`, `vyre-conform/src/pipeline/bin/verify_cert.rs`, `vyre-conform/src/enforce/gate_7_coverage.rs`, `vyre-conform/src/enforce/enforcer_gpu_mandatory/scan.rs`, `vyre-conform/src/specs/primitive/mod.rs`, `vyre-conform/src/verify/golden/util/mod.rs`, `vyre-conform/src/verify/golden_samples/mod.rs`, `vyre-conform/tests/adversarial/gate7_omitted_rows.rs`, `vyre-core/tests/gap/test_primitive_math_gap.rs`, `vyre-core/fuzz/gpu/src/…`. Rewrite every import to `vyre_conform::proof::reference::…`. Core tests that need it add `vyre-conform` as a `[dev-dependencies]` path dep.
  - Delete the `parity-oracle` feature from `vyre-core/Cargo.toml` once the move completes.
- **M1 — README claim** ("no bytecode VM") now matches reality after C3. Keep the README note; verify no new bytecode code lands.

### 2.3 `unsafe` and workspace lints (C5, C6, C7)

- **C5 — `vyre-core/src/runtime/shader/compile_compute_pipeline.rs:118`** — `#[allow(unsafe_code)]` already scoped to the single `wgpu::Device::create_pipeline_cache` expression with a SAFETY note. Keep; any other unsafe block in core must follow the same tight pattern.
- **C6 — `vyre-conform/src/meta/oom/alloc.rs`** — `#[global_allocator] static GLOBAL_ALLOCATOR: OomAllocator`. Already guarded behind the `oom-injection` feature and default set to empty. No further change; verify no CI target turns it on for release.
- **C7 — `vyre-conform/Cargo.toml`** — `[lints] workspace = true` already added. Verify the workspace lints actually include `unsafe_code = "deny"` and extend to every publishable crate.

### 2.4 Backend dependency layering (C8, C9)

- **C8 — `vyre-core/Cargo.toml`** currently depends on `wgpu`, `pollster`, `bumpalo`. Vision: core is the backend-agnostic IR + ops + lowering crate. Fix: extract every wgpu-using module (`vyre-core/src/runtime/`, `vyre-core/src/engine/`) into a new crate `vyre-wgpu` (or keep the existing `vyre` name and introduce a separate `vyre-runtime-wgpu`). Core's Cargo.toml then only carries IR-level deps.
- **C9 — `vyre-std/Cargo.toml`** previously listed `wgpu` directly. Commit `4860ee4b35` dropped the direct dep. Verify no indirect pull via other deps and that `vyre-std` builds with `--no-default-features`.

### 2.4b Compile-error clusters currently holding the workspace red (C14-C18)

- **C14 — `vyre-core/src/compiler/` vs `vyre-core/src/engine/` vs `vyre-core/src/ops/`**. New `compiler/` dir appeared last session with 10 files (see §0.0). Decision: compiler primitives (dataflow, dominator, typed_arena, string_interner, visitor_walk, recursive_descent, buffer_layouts) are IR-transform infrastructure — they belong in `vyre-core/src/ir/transform/` (subdirs by responsibility). `wgsl_backend.rs` + `wgsl/` subtree inside compiler/ duplicates `vyre-core/src/lower/wgsl/` — fold into `lower/wgsl/`. After this: `vyre-core/src/compiler/` deleted, one canonical home per responsibility, zero dual paths.
- **C15 — `tokenize/mod.rs` 34 `E0639` sites**. `ConformSpec` is `#[non_exhaustive]` (vyre-spec) yet tokenize constructs it with `ConformSpec { field: val, … }` syntax. Fix: `ConformSpec::new(...).with_law(...).with_archetype(...)` builder. Apply to every spec constructor in conform (tokenize is the loudest; there are others). This also locks in LAW 2 evolvability.
- **C16 — Build-scan emits non-gate rows into gates_registry.rs**. `BeamSearch`, `NullBackend` are scanned from a dir inside `enforce/` but aren't `impl EnforceGate`. Fix build_scan's gate-detection to filter on `impl EnforceGate for …` rather than "file exists in dir", AND move the two non-gate items out of `enforce/gates/`.
- **C17 — `&'static dyn Archetype` not `Archetype`**. Archetype trait needs `impl<T: Archetype + ?Sized> Archetype for &T` (blanket) OR registry emits owned `Box<dyn Archetype>`. Pick blanket impl — zero allocation, matches `Oracle` pattern.
- **C18 — `vyre-conform/fuzz` excluded from workspace**. Re-include as workspace member. If it breaks, fix it — hidden fuzz rot is worse than visible red.

### 2.5 Duplicate definitions that cause or will cause hard compile errors (C10-C13, M17)

- **C10 — Duplicate `impl Display`**: `StructuralFinding`, `OverflowFinding`, `NoSilentWrongFinding`, `ParityFailure`, `ClosureError`, `CostFinding`, `ArchetypeError`, `MutationError`, `DataType`. For each: keep the impl in `enforce/enforcers/…` or `framework/types/vyre-conform/…` (canonical), delete the old orphan copy.
- **C11 — `default_composers()`** in both `vyre-conform/src/generate/composers/default_composers.rs` AND `vyre-conform/src/generate/composers/mod.rs:20`. Delete the mod.rs copy.
- **C12 — `impl Drop for SourceSnapshot`** in both `vyre-conform/src/verify/harnesses/mutation/impl_sourcesnapshot_drop.rs:9` AND `vyre-conform/src/verify/harnesses/mutation/mod.rs:250`. Already folded into mod.rs during the `O14` merge — but duplicate likely remains until one body is deleted. Verify.
- **C13 — `archetypes_registry.rs`** at both `vyre-conform/src/archetypes_registry.rs` AND `vyre-conform/src/generate/archetypes_registry.rs`. Top-level version deleted in commit `42ba0c1e0c`. Verify.
- **M17 — `vyre-conform/src/types/failure.rs`** vs `vyre-conform/src/framework/types/vyre-conform/failure.rs`. Top-level deleted in `42ba0c1e0c`; `lib.rs` re-exports `ParityFailure` from the framework path in `83e2dc0188`. Verify.

### 2.5b Conform 15→8 top-level-dir collapse (C19)

Target per ARCHITECTURE: `spec, proof, enforce, pipeline, generate, verify, adversarial, meta`. Current extras and their canonical homes:

- `algebra/` → fold into `proof/algebra/` (algebraic-law engine is part of the proof layer).
- `backend/` (wgpu dispatch machinery) → when C8 extracts the runtime into `vyre-wgpu`, the remaining conform-side wrapper moves to `pipeline/backend/` (wired from the public `VyreBackend` trait; internal only).
- `backends/` → DELETE; content is a vestigial `mod.rs` per O6.
- `comparator/` → `proof/comparator/` (byte-identity comparator is a proof primitive).
- `framework/` vs `vyre-spec/` — ONE survives. The public item is `ConformSpec`; re-export from the new canonical home `vyre-spec/`. Delete `framework/` as a top-level dir; its sub-responsibilities (`loader`, `registry`, `types`, `value`, `builder`) move under `vyre-spec/` + `pipeline/`. Update every `conform::framework::*` call site to the new path. No `pub(crate) use framework as spec` or inverse.
- `generated/` → content moves to `build.rs` OUT_DIR emissions (H10). Source tree has zero committed generated code.

Post-collapse: `ls vyre-conform/src/` returns exactly `adversarial enforce generate meta pipeline proof spec verify lib.rs` and no others.

### 2.6 Release-blocking publish hygiene (Kimi-10)

- **`vyre-core/Cargo.toml:53`** build-dep on unpublished `vyre-build-scan`. Publish order must be `vyre-build-scan` → `vyre-spec` → `vyre` → `vyre-std` → `vyre-conform`.
- **`vyre-conform/Cargo.toml:67`** build-dep on `vyre-conform-codegen` (publish = false). Either make codegen publishable or inline the codegen logic. A publishable crate cannot build-depend on an unpublishable workspace member.
- **`vyre-std/Cargo.toml:4-11`** hardcodes `edition`, `rust-version`, `license`, `authors`, `repository`, `homepage`. Change every one to `.workspace = true`.
- **All publishable crates** missing `[package.metadata.docs.rs]` block with `all-features = true` and `rustdoc-args = ["--cfg", "docsrs"]`. Add.
- **`vyre-std/`** missing `LICENSE-APACHE` and `LICENSE-MIT` at crate root. Copy from workspace root or switch to workspace-inherited licensing.

## 3. High findings (structural rot and layering)

### 3.1 Migration graveyards (O1-O10, O11, O12, O13, O15, H13)

Status after commits `e8f13ae21f` (graveyard purge) and `4cfb8725dd` + `8f84ee8ecb` (enforce orphans):

- O1 `vyre-conform/src/oracles/` (→ `proof/oracles/`) — removed.
- O2 `vyre-conform/src/mutations/` (→ `adversarial/mutations/`) — removed.
- O3 `vyre-conform/src/observe/` (→ `meta/observe/`) — removed.
- O4 `vyre-conform/src/corpus/` (→ `verify/corpus/`) — removed.
- O5 `vyre-conform/src/contribute/` (→ `meta/contribute/`) — removed.
- O6 `vyre-conform/src/backends/` (thin stub vs `backend/`) — removed.
- O7 `vyre-conform/src/types/` (→ `framework/types/vyre-conform/`) — removed.
- O8 `vyre-conform/src/vyre-spec/` vs `vyre-conform/src/specs/` — collapsed into `vyre-spec/` (commit this session).
- O9 `vyre-conform/src/reference/` vs `vyre-core/src/reference/` — **STILL OPEN**. Fold both into `vyre-conform/src/proof/reference/` per C4.
- O10 `vyre-conform/src/bin/` (→ `pipeline/bin/`) — removed.
- O11 `enforce/` triple registry — legacy subdirs + top-level `*.rs` deleted; still **verify** `enforcers/` + `gates/` cover every gate and no caller imports from the removed paths.
- O12 `adversarial/defender/` vs `adversarial/defenders/` — **STILL OPEN**. `defenders/` was a thin `pub use generated::*;` re-export; delete and fold into `defender/`.
- O13 `verify/golden/` vs `verify/golden_samples/` — **STILL OPEN**. Delete `golden_samples/` and merge into `golden/`.
- O15 `docs/audits/rescue/` — deleted in commits this session.
- H13 — 19 legacy `enforce/*.rs` files + subdirs — removed.

### 3.2 Filesystem-is-registry (H3, H4, M5)

- **H3 — build_scan wiring**: original coverage was 1 of 7 responsibility directories. `fix(vyre): H3 — wire build-scan responsibility directories` (commit `7a08ba9b3f`) addresses this. Verify every ARCHITECTURE table entry has a scan: `enforce/gates/`, `proof/oracles/`, `generate/archetypes/`, `adversarial/mutations/`, `backends/` (now folded), `vyre-spec/findings/`, `vyre-core/src/ops/{category}/`.
- **H4 — Central registries**: `ARCHETYPES` array + core `REGISTRY` const. Replaced with generated scans in commit `432ea51a8c`. Verify no remaining hand-listed registry constant anywhere.
- **M5 — `explicit module list`**: ARCHITECTURE + SANTH_STANDARD claim it is used. Verify every responsibility `mod.rs` invokes `explicit module list` rather than hand-listing children.

### 3.3 File size + nesting (H5, H6, H18)

Files that must split below 500 LOC (each listed with size; split by responsibility into siblings + `explicit module list`):

| LOC | file |
|-----|------|
| 2730 | vyre-conform/src/enforce/enforcers/float_semantics.rs |
| 1098 | vyre-conform/src/enforce/enforcers/category_b.rs |
| 926 | vyre-conform/src/enforce/enforcers/reference_trust.rs |
| 912 | vyre-conform/src/enforce/enforcers/atomics_race.rs |
| 775 | vyre-conform/src/algebra/audit/mod.rs |
| 762 | vyre-conform/src/vyre-spec/primitive/mod.rs |
| 757 | vyre-conform/src/enforce/enforcers/signature_match.rs |
| 756 | vyre-conform/src/verify/properties/tests/declared_laws.rs |
| 739 | vyre-core/src/lower/wgsl/expr/mod.rs |
| 737 | vyre-conform/src/enforce/enforcers/zero_stubs.rs |
| 737 | vyre-conform/src/enforce/enforcers/structural_rules.rs |
| 705 | vyre-conform/src/vyre-spec/string/tokenize/mod.rs |
| 677 | vyre-conform/src/framework/builder.rs |
| 675 | vyre-conform/src/algebra/gpu_checker/mod.rs |
| 656 | vyre-conform/src/algebra/mandatory_inference/cross/mod.rs |
| 614 | vyre-conform/src/enforce/enforcers/decomposition.rs |
| 607 | vyre-conform/src/enforce/enforcers/engine_composition.rs |
| 606 | vyre-conform/src/enforce/enforcers/composition_closure.rs |
| 594 | vyre-conform/src/enforce/enforcers/category_a.rs |
| 588 | vyre-core/tests/adversarial/float/common.rs |
| 582 | vyre-conform/src/algebra/formal_vyre-spec/mod.rs |
| 579 | vyre-conform/src/enforce/enforcers/category_c.rs |
| 570 | vyre-conform/src/generate/emit/independence/mod.rs |
| 556 | vyre-conform/src/enforce/enforcers/barrier_placement.rs |
| 540 | vyre-core/src/reference/typed_ops/mod.rs |
| 539 | vyre-core/src/ops.rs |
| 538 | vyre-conform/src/enforce/enforcers/overflow_contract.rs |
| 515 | vyre-conform/src/vyre-spec/engine/eval/vm.rs |
| 501 | vyre-core/build.rs |

Paths that violate the 4-level cap. Flatten via rename + `explicit module list` wiring:

- `vyre-conform/src/verify/properties/tests/declared_laws/{associative,identity,demorgan/checker,distributive,complement,…}` (8+ levels).
- `vyre-core/src/ops/compression/ops/{deflate_decompress,gzip_decompress,zlib_decompress,zstd,lz4}/{implementation,metadata,fixtures,lowering}/` (7 levels).
- `vyre-core/src/ops/workgroup/primitives/{stack,queue_fifo,queue_priority}/{metadata,fixtures}/` (6 levels).
- `vyre-core/src/ir/transform/optimize/tests/unit/cse/` (6 levels).
- `vyre-conform/src/generate/emit/cross_product/codegen/context/` (6 levels).

### 3.4 `util/`, mod.rs logic, pub-use globs (H7, H8, H9)

- **H7 — `vyre-core/src/util/`**: renamed to `vyre-core/src/bytemuck_safe/` (commit `823da7212b`). Verify no `use crate::util::` remains.
- **H8 — `mod.rs` heavy logic**: 22 `mod.rs` files >400 LOC (see table above). Each must become: doc comment + `explicit module list` + re-exports. Logic moves to sibling files.
- **H9 — `pub use X::*` globs**: 20 occurrence sites across core ops (string_matching kernels, hash kernels, workgroup primitives specs, security_detection catalog). Replace each with explicit named re-exports.

### 3.5 Examples, CODEOWNERS, require_gpu (H11, H12, H19)

- **H11 — `require_gpu()` panics** in `vyre-conform/src/backend/mod.rs:371`. Returns `Result<WgpuBackend, String>` after commit `575fc419fc`. Verify every caller uses `.expect("…")` in test context only.
- **H12 — `examples/hello_vyre/src/main.rs`** used `vyre_conform::algebra::verify_laws`, `enforce_signature`, `reference::run`, `framework::Value`. Rewritten to use only the 10 public items in commit `f8c3d6747c`. Verify the example still compiles under `H1`'s demoted public surface.
- **H19 — CODEOWNERS dead paths**: `/vyre-conform/src/vyre-spec/`, `/vyre-conform/src/reference/`, `/vyre-conform/src/mutations/`. After this session: `vyre-spec/` exists again (good), `reference/` becomes `proof/reference/` (update CODEOWNERS), `mutations/` lives at `adversarial/mutations/` (update CODEOWNERS).
- **H20 — pub use * glob remnants** in `vyre-conform/src/framework/loader/toml/mod.rs` (10), `vyre-conform/src/framework/value/mod.rs` (3), `vyre-conform/src/framework/registry/mod.rs` (2). Replace with explicit re-exports.

### 3.6 Generated code (H10)

- **H10 — `vyre-conform/src/generated/`** on disk. Some remnants deleted this session; remaining build.rs emission must land in `$OUT_DIR` with `include!(concat!(env!("OUT_DIR"), "/…"))` pulls. Delete any committed `src/generated/` content.

### 3.7 Orphan bin files (M18, H15)

- **M18** — 9 `fn main()` files under `vyre-conform/src/pipeline/bin/` without `[[bin]]` declarations in `vyre-conform/Cargo.toml`. Either declare each with a `[[bin]] name = … path = …` or delete as dead code.
- **H15** — `vyre-conform/src/bin/phase6_calibrate.rs` duplicated `vyre-conform/src/pipeline/bin/phase6_calibrate.rs`. The `bin/` dir was deleted in the graveyard purge; verify.

### 3.8 Process::exit in library code (H14)

- `vyre-conform/src/enforce/registry.rs:218`, `vyre-conform/src/enforce/enforce_all.rs:218`, `vyre-conform/src/pipeline/bin/phase6_calibrate.rs:458,481` call `std::process::exit`. Library code must return `Result` or propagate. Two of those enforce files were deleted in commit `8f84ee8ecb`; the pipeline bin remains — fix it.

### 3.9 SemVer hazards (Kimi-1)

From `vyre-spec`:

- `lib.rs:22-111` every `pub mod X; pub use X::Type;` pair creates dual identity. Make every child module `pub(crate)` and expose types exclusively via the crate root.
- `verification.rs:6,34` `pub enum Verification` missing `#[must_use]`; `count(self) -> Option<u64>` collides with `Iterator::count` if `Verification` ever becomes iterable. Rename to `witness_count()` and add `#[must_use]`.
- `layer.rs:8,40` `pub const fn description(self)` collides with deprecated `std::error::Error::description`. Rename to `layer_description()`.
- `metadata_category.rs:8,23` `pub const fn id(self)` collides with future `Id` traits. Rename to `category_id()`.
- Every `Copy + non_exhaustive` enum/struct (`IntrinsicTable`, `BinOp`, `UnOp`, `DataType`, `AtomicOp`, `BufferAccess`, `Convention`, `AdversarialInput`, `GoldenSample`, `KatVector`, `Invariant`, `EngineInvariant`, `FloatType`, etc.) — drop `Copy` so future variants may carry owned payloads without a major bump.
- `pub const INVARIANTS`, `pub const ALL_ALGEBRAIC_LAWS`, `pub const LAW_CATALOG` expose length + ordering as stable contract. Replace with `pub fn invariants()`, `pub fn all_algebraic_laws()`, `pub fn law_catalog()`.
- `by_id::by_id` hardcodes match on 15 variants; `catalog_is_complete` hardcodes `15`. Derive dynamically from `EngineInvariant`.
- `missing_backends() -> Vec<&'static str>` and `by_category() -> Vec<&'static Invariant>` — return `impl Iterator<Item = …>` instead of forcing a heap-allocated Vec into the public signature.
- `BackendAvailability = fn(&str) -> bool` type alias — rigid. Switch to a trait-object wrapper so predicates may capture state in the future.

### 3.10 Cat-B violations (Kimi-2)

- Every `pub use <mod>::*` in core is flagged as a Cat-B glob. See §3.4 H9. Enumerated at `vyre-core/src/ops/compression/ops/{zstd,lz4,zlib_decompress,gzip_decompress,deflate_decompress}/{implementation,mod}.rs`, `vyre-core/src/ops/crypto/chacha20_block/mod.rs`, `vyre-core/src/ops/hash/{entropy,rolling}/kernel.rs`, `vyre-core/src/ops/security_detection/{catalog.rs, mod.rs}`, `vyre-core/src/ops/string_matching/{aho_corasick_scan,glob_match,kmp_find,nfa_scan,rabin_karp_find,wildcard_match}/{kernel.rs, mod.rs}`, `vyre-core/src/ops/workgroup/primitives/{hashmap,queue_fifo,queue_priority,stack,state_machine,string_interner,typed_arena,union_find,visitor}/mod.rs`.
- CPU execution paths reachable without `parity-oracle`: `vyre-core/src/ops.rs:43` re-exports `CategoryAOp, CpuOp`; `vyre-core/src/ops/cpu_op.rs:{6,18,21}` pub trait/type/fn. Gate the entire trait behind `#[cfg(any(test, feature = "parity-oracle"))]` OR, per C4, move into `vyre-conform`.
- `vyre-core/src/ops/security_detection/catalog/detect_*.rs` each does `pub use super::super::super::lowering::source` — six sites. Remove the re-exports; callers import `lowering::source` from its canonical path.

### 3.11 Concurrency hazards (Kimi-3)

CRITICAL:
- `vyre-conform/src/algebra/checker/verify_one_law_witnessed.rs:31` — `persist_violation(…)` from inside rayon `par_iter` races on SHA-named file and `.expect`s.
- `vyre-conform/src/pipeline/streaming/regression_sinking.rs:91-97,133-140` — temp file names only `pid`; concurrent flushes clobber.
- `vyre-conform/src/meta/harness/mod.rs:114-154` — mutates source files without file locking.
- `vyre-conform/src/meta/harness/mod.rs:295-319` — last-writer-wins on META_FINDINGS_*.md.
- `vyre-conform/src/corpus/witness.rs:160-163` — append without file locking; concurrent processes interleave.

HIGH:
- `vyre-conform/src/pipeline/notify.rs:16-20,26-29,36-39,52-55` holds `MutexGuard<dyn Reporter>` across user-defined trait calls — reentrance deadlock.
- `vyre-conform/src/backend/wgpu/dispatch.rs:28-40` uses `RwLock::read/write().expect(…)` — poisoned lock panics the process.
- `vyre-conform/src/backend/wgpu/context.rs:38-47` holds global mutex across multi-second `pollster::block_on`.
- `vyre-conform/src/verify/golden/freeze_goldens.rs:31-44` TOCTTOU on `path.exists() → fs::write`.

MEDIUM:
- `vyre-conform/src/algebra/checker/mod.rs:183-185` synchronous filesystem I/O inside `par_iter`.
- `vyre-conform/src/{bin,pipeline/bin}/phase6_calibrate.rs:184-185` + `vyre-conform/src/pipeline/bin/core.rs:178-179` `HashSet → Vec` non-determinism.
- `vyre-conform/src/meta/harness/mod.rs:168-172` `HashSet::difference().cloned().collect()` non-determinism in `MetaGateReport`.
- `vyre-conform/src/observe/subscriber.rs:61` global `Arc<Mutex<RecorderState>>` serializes every tracing event.
- `vyre-conform/src/enforce/enforcers/layer8_feedback_loop.rs:11-12` + `vyre-conform/src/algebra/mandatory_inference/mod.rs:15` unbounded global caches with no eviction.
- `vyre-conform/src/framework/loader/toml/load.rs:18-22`, `vyre-conform/src/meta/oracle.rs:239-244` TOCTTOU patterns.

### 3.12 Non-determinism in certificates (Kimi-6)

CRITICAL:
- `vyre-conform/src/pipeline/execution/mod.rs:87-88` elapsed-ms from `Instant::now()` formatted into `ParityFailure.message` and serialized into the cert.
- `vyre-conform/src/pipeline/certify/parity.rs:118-128` total-bomb path embeds elapsed-ms into cert message.
- `vyre-conform/src/verify/budget/mod.rs:34-42` `ReferenceBombDetected` `Display` includes elapsed-ms.
- `vyre-conform/src/pipeline/certify/track.rs:126` `CERTIFICATE_SEQUENCE` atomic counter — identical inputs produce different certs across runs. Remove `monotonic_sequence` from the cert OR derive deterministically from a hash of the inputs.

MEDIUM:
- `vyre-conform/src/pipeline/bin/contribute.rs:54` serializes `ContributeReport` with `duration_ms` / `total_duration_ms`.
- `vyre-conform/src/meta/harness/mod.rs:168-172` `adversaries_killed/survived` built from `HashSet<String>` iteration.

### 3.13 Allocation paths (Kimi-4)

LAW 8 — every finding is critical, no cold/hot split. Every site below is fixed. Pre-size + borrow + arena across every IR transform + validate + lowering + conform orchestration site. Representative entries:

Pre-size + borrow + arena across every IR transform + validate + lowering site. Full list lives in the Kimi-4 output; representative entries:

- `vyre-core/src/ir/validate/validate.rs:45-73` unsized `FxHashSet`/`FxHashMap`; pre-size from `program.buffers.len()`.
- `vyre-core/src/ir/validate/validation_error.rs:10` `pub message: String` — switch to `Cow<'static, str>`.
- `vyre-core/src/ir/transform/optimize/{dce,cse}/*.rs` clone program buffers + entry-op-id; move instead.
- `vyre-core/src/ir/transform/optimize/cse/impl_exprkey.rs:{12,14,16}` `to_string()` per variable/buffer — use `&str` or `Arc<str>` in `ExprKey`.
- `vyre-core/src/ir/transform/inline/impl_inlinectx.rs:229-256` `format!`/`to_string` per call; reuse a scratch `String`.
- `vyre-core/src/ir/model/program/mod.rs:409` rebuild `buffer_index`; reuse when buffers unchanged.
- `vyre-core/src/lower/wgsl/impl_lowerctx.rs:{6,17}` `"  ".repeat()` + `format!` per node; write into output buffer.
- `vyre-core/src/lower/wgsl/node.rs:{55,154}` clone variable names into `TypeScope`; store `&str`.
- `vyre-core/src/lower/wgsl/analysis/atomic_buffers.rs:7` unsized HashSet; pre-size.
- `vyre-core/src/engine/{decode,decompress,dfa}/mod.rs` `to_vec()` / `VecDeque::from(..to_vec())` on hot buffers; borrow.

Includes P2 `emit_expr_string`, P3 `Program::Clone`, P4 `certify()` Vec/BTreeMap/String churn, P5 byte_words padding, P6 reference interp HashMap (audited against vyre-reference after C4 move), P8 902 `.clone()` calls across the workspace (≥30% cut), P9 `parking_lot` swap everywhere `std::sync::Mutex` survives in non-test library code, P10 `default_generators()` one-shot `LazyLock<&'static [&'static dyn InputGenerator]>`. Every site — core, vyre-reference, vyre-wgpu, vyre-std, vyre-conform — is fixed.

### 3.14 Missing `#[must_use]` (Kimi-5)

Report types that silently drop important state: every `pub enum / pub struct` whose name ends in `Error`, `Certificate`, `Report`, `Finding`, `Violation`, `Result`, `Outcome`, or any builder chain returning `&mut Self` / `Self`. Full list in Kimi-5 output — apply across vyre-spec/, vyre-core/, vyre-conform/.

### 3.15 Error-message `Fix:` prefix (Kimi-8)

Every `panic!`, `.expect`, `bail!`, `Err(thiserror_variant { message: … })`, `map_err(|_| …)` must begin with `Fix: …`. Hot sites (truncated):

- `vyre-conform/src/determinism_stress.rs:33,39` `panic!("{op_name}: ...")` without `Fix:`.
- `vyre-conform/src/vyre-spec/engine/mod.rs:35,138` panics missing `Fix:`.
- `vyre-conform/src/meta/oom/alloc.rs:89` `panic!("{OOM_SENTINEL}")`.
- `vyre-conform/src/backend/gpu_parity.rs:41`, `vyre-conform/src/meta/harness/mod.rs:257,259`.
- Every `.expect(…)` under `vyre-conform/src/{framework,bin,pipeline/bin,backend/wgpu,oracles,meta,specs}/…`.
- `map_err(|_| …)` sites at `vyre-std/src/pattern/cache.rs:193`, `vyre-std/src/pattern/dfa_pack.rs:{147,156,157,158,177,178,179,185,199}`.
- `vyre-core/src/reference/value/numeric_views.rs:{11,24}`, `vyre-core/src/engine/dfa/buffers/mod.rs:77` `try_from(…).ok()` collapsing Result to Option.
- Every workgroup primitive Error enum missing a `Fix:`-prefixed `Display` impl (`FifoError`, `StackError`, `VisitError`, `ArenaError`, `HashmapError`, `PriorityError`, `UnionFindError`, `StateMachineError`, `InternError`).
- Swallowed results at `vyre-std/src/pattern/cache.rs:{58,78,104}`.

### 3.16 Fuzzing + property coverage (Kimi-7)

- Add unit tests for `arithmetic_mean`, `variance`, `std_dev`, `byte_histogram`, `chi_square`, `sliding_entropy` in `vyre-conform/tests/unit/stats.rs` (draft lives in Kimi-7 output).
- Add float-op integration tests `f32_{abs,add,ceil,clamp,cmp,cos,div,eq,floor,fma,is_finite,is_inf,is_nan,le,lt,max,min,neg,rem,round,sign,sin,sqrt,trunc}` in `vyre-core/tests/integration/primitive_ops/float/`.
- Extend `vyre-core/tests/integration.rs` `mod` list to include the new float tests.
- Add fuzz targets for any op in vyre-core/src/ops that lacks a target in vyre-core/fuzz/.
- Un-`#[ignore]` `benches/vs_cpu_baseline.rs`; rewrite `benches/primitives_showcase.rs` so every iter body runs actual GPU dispatch + readback, not `black_box(rows.len())` (P7).

### 3.17 Op-registry completeness (Kimi-9)

~90 ops registered in core but missing a conform spec, including `compression.zlib_decompress`, `compression.zstd_decompress`, `crypto.chacha20_block`, `data_movement.*`, `decode.*`, `encode.*`, `graph.dfs`, every `hash.*`, `match.dfa_scan`, `match.scatter`, every `reductions.*`, every `rule.*`, `scan.prefix_sum_inclusive`, `sort.bitonic_sort_u32`, `string.prefix_brace`, `string.tokenize_gpu`, every `workgroup.*`, every `compiler_primitives.*`. Each needs a `vyre-conform/src/vyre-spec/<category>/<op>.rs` with CPU reference, WGSL declaration, laws, archetypes, KAT vectors, adversarial inputs.

Ops whose `cpu_fn` is stubbed at `structured_intrinsic_cpu`: all `workgroup.*`. Implement a flat-byte CPU adapter for each.

Engines in conform without a declared WGSL lowering: `engine.dfa`, `engine.eval`, `engine.scatter`. Either add the WGSL path or mark CPU-only explicitly in the spec.

### 3.18 Dead + suppressed code (M14, M15, M16)

- `#![allow(missing_docs)]` surviving in core under `ir/serial/wire/{tags,decode,encode,framing}/`, `ir/transform/inline/`, `ir/transform/optimize/{dce,cse}/`, `reference/hash/{hmac,blake3,sha3}/`. Remove every one; write real docs.
- `#[allow(dead_code)]` at 18 non-test sites (enumerated in user's second audit, M15). Delete dead items or wire in the caller.
- `#[allow(clippy::too_many_arguments)]` at `vyre-conform/src/types/failure.rs:39` (already deleted), `vyre-conform/src/framework/types/vyre-conform/failure.rs:39`, `vyre-core/src/engine/dataflow/mod.rs:215`. Split the offending functions into builder-shaped types.
- `#[allow(clippy::all)]` blanket at `vyre-conform/src/generated.rs:5`. Narrow to specific lints.
- `#[allow(clippy::expect_used, clippy::panic)]` at `vyre-conform/src/algebra/checker/persist_violation.rs:7` and `checker/mod.rs:258`. Either fix the code or annotate with a documented `// Reason:`.

### 3.19 Perf (P1-P10)

- P1 — `vyre-conform/src/backend/wgpu/dispatch.rs` caches validation + routes through `vyre::runtime::shader::compile_compute_pipeline` (done in `01feff825a`). Keep.
- P2 — `vyre-core/src/lower/wgsl/expr/mod.rs:133` `emit_expr_string` allocates String per sub-expression — rewrite to `&mut dyn fmt::Write`.
- P3 — `vyre-core/src/ir/model/program/mod.rs:51` deep-copies entire IR; switch to `Arc<[BufferDecl]>` + `Arc::make_mut`.
- P4 — `vyre-conform/src/pipeline/certify/mod.rs` Vec/BTreeMap/String churn; pre-size + arena.
- P5 — `vyre-conform/src/backend/wgpu/byte_words.rs:8` pad-to-words copies; zero-copy upload via aligned slice.
- P6 — reference interpreter storage HashMap move/replace; moot after C4 moves the oracle to conform, but audit the surviving `proof/reference/` for the same pattern.
- P7 — benchmarks measure `black_box(rows.len())`; rewrite every iter to actually run the primitive + GPU dispatch/readback, un-ignore `vs_cpu_baseline.rs`, add CI bench job.
- P8 — 902 `.clone()` calls in non-test code; audit hot paths (`certify`, backend dispatch, spec registry iteration) and cut ≥30%.
- P9 — replace every `std::sync::Mutex` in conform non-test library code with `parking_lot::Mutex` (no-poisoning, faster); same for `RwLock`.
- P10 — `vyre-conform/src/generate/generators/mod.rs` `default_generators()` allocates a fresh `Vec<Box<dyn InputGenerator>>` per call; switch to `LazyLock<&'static [&'static dyn InputGenerator]>` so allocation happens once.

## 4. Medium findings (docs + CI)

- M2 — README `cargo add vyre vyre-conform` command will work only after publish; leave as aspirational until all blockers close.
- M3 — ARCHITECTURE migration status section deleted (commit `3e6ff10d89`). Keep that way.
- M4 — CONTRIBUTING documents all 5 contributor flows (commit `809f84721a`).
- M6 — THESIS `layer4/5` claim matches code (`layer4_mutation_gate::validate_catalog` + `adversarial::run_gauntlet` both wired into certify). Keep.
- M7 — CODEOWNERS updated to reference `vyre-spec/`, `proof/reference/`, `adversarial/mutations/` after C4 completes and other dirs stabilize.
- M8 — CI `cargo test` job added + matrix (commit `0a8d6bb0b4`). Keep.
- M9 — Clippy errors to resolve in `vyre-core/src/engine/dataflow/mod.rs`, `vyre-core/src/engine/decompress/dispatch_kernel/mod.rs`, `vyre-core/src/ir/validate/nodes.rs` (once structural + perf passes complete).
- M10 — 215+ `#![allow(missing_docs)]` in vyre-core: the Kimi reports narrow this to ~18 files under `ir/serial/wire/*` + `ir/transform/inline|optimize` + `reference/hash/{hmac,blake3,sha3}/`. Remove each and write docs.
- M11 — `CertificateLevels` `#[non_exhaustive]` added (commit `575fc419fc`).
- M12-M16 — listed in 3.18.
- M18 — listed in 3.7.

## 5. Low / residual

- `fix_warnings.py`, `fix_missing_docs.py`, one-off agent scripts — already deleted.
- `docs/audits/audits/{logs,…}` — deleted.
- `docs/audits/rescue/` — deleted.
- Empty `tests.rs` stubs (`vyre-core/src/ops/security_detection/tests.rs`) — deleted.
- `vyre-conform/src/runtime/cache/tests/mod.rs` was never wiring `unit/`; fixed in `e67267d47d`.
- Every remaining `.rej`, `.orig`, `.bak`, `.rs.bk` — final grep sweep once structural work lands.

## 6. Execution plan — dependency order (strict)

Stage 1 — unblock compile after the deletions this session:
1.1 Run `cargo check --workspace --all-targets --all-features`. Every E0252 duplicate-name error traces to C10/C11/C12/C13/M17 duplicate-removal follow-ups. Close each by deleting the remaining stale definition that still re-exports the same name.
1.2 Fix any import that still points at a deleted `vyre-conform/src/{oracles,mutations,observe,corpus,contribute,reference,types,spec,bin,backends}` path.
1.3 Ensure `build_scan` fully compiles: `vyre-build-scan/src/rust_specs.rs:64` usage of `unwrap_or_else(fatal)` must either take `PathBuf` or wrap in Result per the existing `fatal::{fatal, required_env_path}` contract.

Stage 2 — public-API lockdown (blocks publish):
2.1 C1 + C2 public certify signature + WgslBackend pub(crate).
2.2 H2 trait signature alignment (EnforceGate, Archetype, Oracle).
2.3 H1 shrink conform public API to 10 items.
2.4 H12 rewrite `examples/hello_vyre` against the 10 items.

Stage 3 — structural invariants (blocks 5-year SemVer confidence):
3.1 C4 move `vyre-core/src/reference/` → `vyre-conform/src/proof/reference/`; delete `parity-oracle` feature.
3.2 C8 extract wgpu from core into `vyre-wgpu` crate.
3.3 O9, O12, O13 final migration-graveyard removals (reference/, defenders/, golden_samples/).
3.4 H5 split every file >500 LOC (30 sites).
3.5 H6 flatten every path >4 levels (30 sites).
3.6 H8 thin every mod.rs >400 LOC (22 sites).
3.7 H9, H20 replace every `pub use X::*` glob with explicit re-exports.
3.8 H10 move remaining `vyre-conform/src/generated/` content to `$OUT_DIR`.
3.9 M5 introduce `explicit module list` in every responsibility dir.

Stage 4 — correctness + SemVer polish:
4.1 Kimi-1 vyre-spec SemVer: drop `Copy` on growing enums/structs, rename colliding methods, replace `pub const` catalogs with `pub fn`, hide child modules.
4.2 Kimi-5 `#[must_use]` pass across spec, core, conform.
4.3 Kimi-3 concurrency fixes (CRITICAL before publish).
4.4 Kimi-6 determinism fixes (CRITICAL — cert must hash-identical for identical inputs).
4.5 Kimi-8 `Fix:` prefix pass across every `panic!` / `.expect` / `Err` / `map_err`.
4.6 Kimi-2 Cat-B glob removal + CPU-path gating (overlaps with H9 + C4).
4.7 Kimi-9 fill the ~90 missing conform specs + implement workgroup `cpu_fn` bodies + declare engine WGSL paths.

Stage 5 — hygiene + perf:
5.1 M14, M15, M16 delete dead_code + remove clippy allow-blankets + split too_many_arguments.
5.2 M10 remove every remaining `#![allow(missing_docs)]` + `//! Doc.` placeholder.
5.3 P2-P10 allocation/perf passes.
5.4 M18 declare every `pipeline/bin/` file as a `[[bin]]` or delete.
5.5 H14 convert every `process::exit` to Result.

Stage 6 — tests + CI:
6.1 Kimi-7 add missing unit/fuzz/bench targets.
6.2 P7 real benchmarks + bench CI job.
6.3 M8 + M9 publish-dry-run + doc-clean + strict-clippy CI jobs (already landed in `0a8d6bb0b4`).

Stage 7 — publish readiness (Kimi-10):
7.1 `vyre-std/Cargo.toml` switch to `.workspace = true` inheritance; copy LICENSE files or inherit.
7.2 Every publishable crate adds `[package.metadata.docs.rs]` stanza.
7.3 `vyre-conform/Cargo.toml` eliminates the build-dep on unpublishable `vyre-conform-codegen` (inline the codegen OR publish codegen).
7.4 `vyre-core/Cargo.toml` builds-dep on `vyre-build-scan` stays once build_scan publishes first.
7.5 Topological publish: `vyre-build-scan` → `vyre-spec` → `vyre` → `vyre-std` → `vyre-conform`.

## 7. Per-file coverage map

The full list of 2228 `.rs` files is too long for this plan document. Every file falls into one of:

- (A) File is canonical, documented, <500 LOC, flat, all Err prefixed `Fix:`, no `#[allow]` blanket, no `pub use *`, no `Copy` on growing types, no orphan — PASSES.
- (B) File is at a deprecated path that was removed this session — should no longer exist; if it does, blocked by a deletion follow-up.
- (C) File contains one or more of: LOC >500, missing docs, `pub use *`, missing `Fix:` prefix, unchecked cast, unguarded `unwrap/expect/panic`, held `Mutex` across user callback, `HashSet` iteration into serialized output, `#[allow]` without documented reason, dead code, `#[allow(clippy::all)]`, inline `edition/license` metadata where workspace inheritance is expected.

Every file in category C must be fixed per the specific remedies in §2-§5 above.

For the release gate, each stage's completion condition is: **zero files in category B**, and every file in category C has a closed finding traced to a commit on main.

## 9. Parallelization plan — one sweep to v0.4.0

Agents are team members, not employees. I work alongside them — I author the hard architectural moves, they fan out on the mechanical work and deep scoped implementations. I review every diff against this plan before merge.

### Topology

- **Trunk-only.** Every agent commits to `main` (or a short-lived branch that merges within the hour). No long-lived worktrees. No rescue branches.
- **One tracker file** `docs/audits/SWEEP.md` — every agent appends a one-line status when they land or fail. I scan it between waves.
- **Shared nuke baseline**: stage 0 lands first. No downstream work starts on a pre-nuke tree.

### Waves (each row = one time slot; agents within a row run in parallel)

| Wave | Me (orchestrator alongside) | Agent fanout (parallel) |
|------|------------------------------|-------------------------|
| 0.A — Zombie audit | Diff every open branch vs main; merge unique work, delete the rest | 1× copilot-mini read-only: catalog what each branch contains, surface losses |
| 0.B — Nuke sweep | Author the nuke: aliases, shims, stubs, feature gates, dead-code allows | 4× Codex parallel: (i) delete every alias/shim in core+std, (ii) delete every alias/shim in conform+spec+build_scan, (iii) delete every stub body + TODO/FIXME/HACK site, (iv) delete silent CPU fallbacks + `parity-oracle` feature |
| 0.C — Post-nuke red audit | Run `cargo check --workspace 2>&1 | tee docs/audits/post-nuke-errors.txt`. Classify every error into (a) rewire to canonical path, (b) real gap needing implementation | 3× copilot-mini read-only: cluster errors by file + by root cause |
| 1.A — Layout lockdown (C14-C19, C4, C8) | Author the 8-module conform collapse + vyre-core/src/compiler disposition + vyre-wgpu extraction plan | 3× Codex parallel (all multi-crate): A) C4 move vyre-core/src/reference/ → vyre-conform/src/proof/reference/; B) C8 extract vyre-core/src/runtime + engine/wgpu → new `vyre-wgpu` crate; C) C19 conform 15→8 top-level collapse (algebra→proof, comparator→proof, framework→spec+pipeline, backends DELETE, generated→OUT_DIR, backend internalized) |
| 1.B — Public-API pristine (H1, H2, C1, C2) | Review the final conform lib.rs against the 10-item list; write the public-api snapshot baseline | 1× Codex: rewrite every `pub(crate) use … as …` into direct canonical paths + update every internal call site; 1× Codex: swap certify signature to `&dyn VyreBackend` end-to-end |
| 2.A — File size + path depth (H5, H6) | Author the split rules for the top 5 worst offenders (2730-LOC float_semantics → by-law subdirs) | 8× Kimi parallel, 1 file each (one-file-one-Kimi): float_semantics, category_b, reference_trust, atomics_race, primitive/mod, signature_match, declared_laws, expr/mod. Next batch: 8 more. 30 total files. |
| 2.B — Path flattening (H6) | Author the target depth-≤4 layout for each 5+ level path | 6× Kimi parallel: declared_laws/*, compression/ops/*, workgroup/primitives/*, ir/transform/optimize/tests/*, generate/emit/cross_product/*, any straggler |
| 2.C — Globs + Fix: prefix + must_use (H9, Kimi-5, Kimi-8) | Spot-check a random 20 sites per wave | 4× Kimi parallel: (i) core globs→explicit, (ii) conform globs→explicit, (iii) Fix: prefix sweep core, (iv) Fix: prefix + must_use sweep conform |
| 3.A — Correctness-critical (Kimi-1, 3, 6) | Review each patch line-by-line — these are the ones that matter for 5-year SemVer + determinism | 3× Codex parallel: A) Kimi-1 vyre-spec SemVer rename + drop Copy + `pub fn` catalogs; B) Kimi-3 concurrency (rayon race, parking_lot swap, file locks, poison removal); C) Kimi-6 determinism (blake3(inputs) cert hash, elapsed-ms out of cert messages) |
| 3.B — Spec fill (Kimi-9, ~90 ops) | Draft the spec template + KAT vector format | 1× Codex 5.4 solo on the template; then 6× Kimi parallel by category: hash, compression, crypto, string/match, reductions, workgroup. Real KATs, no vacuous placeholders (LAW 9). Real tokenize KAT in this wave. |
| 3.C — Fuzz + bench (Kimi-7, P7) | Write the CI bench workflow + the perf floor assertions | 2× Kimi parallel: (i) fuzz targets for every op in vyre-core/src/ops, (ii) real bench iter bodies + un-ignore vs_cpu_baseline |
| 4.A — Doc sweep + `#![deny(missing_docs)]` | Review doc quality on the 10 public items (these must be pristine) | 3× Kimi parallel: core doc sweep, conform doc sweep, std+spec+build_scan doc sweep |
| 4.B — Hygiene: cargo-public-api, semver-checks, deny, udeny, loom, miri, coverage, CHANGELOGs | Author each CI workflow; decide license allowlist; decide coverage floor | 2× copilot-mini read-only audits: surface any public item that would regress the snapshot; list every untested conform module |
| 5 — Publish dry-run + topological publish + tag + release notes | I run `cargo publish --dry-run` in topo order; I author release notes; I push the tag | Nobody else — this is a one-person atomic step |

### Per-wave gating

Before wave N+1 starts, wave N's merges are all green under `cargo check --workspace --all-targets --all-features` and a fresh public-api snapshot is taken. No wave starts before the previous one closes. No "run everything at once"; staged waves keep the tree sane.

### Agent hygiene

- `batch_dispatch` 3 at a time max (MCP limit).
- Every prompt names THE ONE FILE or THE ONE DIRECTORY the agent owns. Never "audit the workspace."
- Every prompt includes the §0.1 NO-SHIMS rules + §0.2 NUKE ORDER verbatim at the top.
- Every prompt includes the specific finding ID being closed (C-, H-, M-, P-, O-, Kimi-) so the commit message threads back to this plan.
- No agent dispatches a second task for itself. One task per agent per wave; I dispatch the next one.

### Communication cadence

- I read `agent_status` at the start of each wave + at 10-min intervals while agents run.
- I read every agent diff in full before `approve_merge`. No auto-merge of writer output.
- Any cursor/copilot auto-review goes to me, not to auto-merge.
- When an agent returns a plan instead of a diff, I either redispatch with clearer scope OR take the task myself if the agent has failed twice.

## 8. Open questions for the user

The plan above requires decisions on four items:

a. **C4 destination** — move `vyre-core/src/reference/` into `vyre-conform/src/proof/reference/` (preferred, matches ARCHITECTURE §8-Module) OR into a separate `vyre-reference` crate? The former keeps conform self-contained; the latter lets non-conform consumers use the reference oracle if they ever need it.

b. **C8 wgpu extraction** — new `vyre-wgpu` crate vs merging the runtime into `vyre-conform/src/backend/wgpu/`? New crate matches the "one purpose per crate" rule; merging shrinks the published-crate count.

c. **Bench strategy** — run real GPU benchmarks on every PR (slower CI, higher confidence) vs only on main post-merge?

d. **Publish cadence** — one-shot alpha.2 with every finding closed (longer wait), or split into alpha.2 (structural + SemVer) → alpha.3 (perf + remaining cleanups)?
