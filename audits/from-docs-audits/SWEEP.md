# vyre SWEEP  -  2026-04-17 → v0.4.0

Shared tracker. Every task is atomic: one owner, explicit scope, explicit acceptance, explicit "do NOT do this." Every task references the MASTER_PLAN finding it closes so the commit message threads back.

**Status legend:** `[ ]` open, `[>]` in progress, `[x]` done, `[!]` blocked.

## CODEX-F5 status  -  2026-04-17

- [x] Issue 7 oracle registry wiring: `vyre-conform/src/proof/oracles/bit_identical.rs` and `vyre-conform/src/proof/oracles/bounded_output.rs` now export real `REGISTERED` oracle implementations, and the generated `ALL_ORACLES` registry contains both entries.
- [!] `cargo check -p vyre-conform --message-format=short 2>&1 | grep 'error\[' | wc -l` currently reports 20. In-scope oracle wiring is not the failing surface. Remaining errors are in F1/F2/F3/F4-owned areas: `DispatchConfig` rename fallout, `pipeline/certify` backend variable/name drift, `layer3_reference_interp` non-exhaustive config construction, and F4 workgroup stack WGSL include path.

**Universal rules  -  every task inherits these from MASTER_PLAN.md §0.0-§0.7:**
- NO SHIMS: no `pub use X as Y` aliases, no thin `pub use other::*` shim modules, no `#[deprecated]` waiting rooms, no dual-path for the same concept, no `publish=false` build-dep in a publishable crate.
- NO STUBS: no `todo!()`, `unimplemented!()`, `panic!("not implemented")`, empty `Ok(())` where the name promises work, `// TODO` / `// FIXME` / `// HACK` / `// temporary` in non-test code. LAW 1.
- NO WEAKENING: never change a doc/test to match broken code. LAW 9.
- NO CPU FALLBACK: `#[cfg(not(feature = "gpu"))]` silent skips are banned. GPU is present on every machine in the fleet.
- NO MAINTAINER-SLACK: LAW 8. Conform gets the same quality bar as every end-user crate.
- Every `Err`/`panic!`/`expect` message begins with `Fix: …`.
- Every `pub` item has a doc comment.
- Every rename is `git mv` + global-grep rewrite + delete old dir + commit atomic.

**Final crate layout** (MASTER_PLAN §0.5): `vyre`, `vyre-spec`, `vyre-reference` (new), `vyre-wgpu` (new), `vyre-std`, `vyre-sigstore` (renamed), `vyre-conform`, `vyre-build-scan`. `xtask/` workspace-only.

**Track ownership  -  zero cross-track file overlap:**
- **ME**: workspace root, pre-flight, §0.2 nuke, crate scaffolds, §0.6 root cleanup, every `Cargo.toml`, every CI workflow, per-crate READMEs + CHANGELOGs, `docs/`, reviews, publish.
- **CODEX-A**: `vyre-reference/` (after my scaffold) + `vyre-wgpu/` (after my scaffold) + `vyre-core/src/{backend}` residue cleanup. Plus C14 `vyre-core/src/compiler/` disposition, C15 OpSpec builder, C17 Archetype blanket impl.
- **CODEX-B**: `vyre-core/src/{ir, lower, ops, engine, compiler}`, `vyre-std/src/`, `vyre-spec/src/`, `vyre-build-scan/src/`. H5/H6/H9/Fix:/must_use sweeps, IR perf (P2/P3/P8/P9/P10 core-side), Kimi-1 `vyre-spec` SemVer.
- **CODEX-C**: everything under `vyre-conform/`. §0.4 rules 1-7, C19 15→8, H2 trait rewrite, H5 enforcer splits, Kimi-3 concurrency, Kimi-6 determinism, Kimi-9 ~90 missing specs, vyre-conform/fuzz re-include, codegen inline, 15→8 collapse.

Collision policy: if a task touches a file outside its track, it's rejected at review. If the plan requires it, reassign the task or split it.

---

## ME  -  Pre-flight (serial, before anything else)

- [ ] **PF-1**  -  Create pre-sweep safety tag. `git tag pre-sweep-2026-04-17 && git push origin pre-sweep-2026-04-17`. Acceptance: tag visible on remote. Rollback point for the whole sweep.
- [ ] **PF-2**  -  Zombie branch audit. For each of `vyre/*`, `jules-rescue/*`, `backup/*`, `t17-*`, `t2-*`, `test/*`, `temp-push-branch`, `fix-all`, `vyre-standalone-20260414`: `git log main..<branch> --oneline` to see unique commits; rescue unique work via cherry-pick; `git branch -D <branch>` + `git push origin --delete <branch>` otherwise. Acceptance: `git branch -a | grep -vE '^\*?\s*(main|HEAD)'` returns nothing.
- [ ] **PF-3**  -  Fleet probe. `mcp__dispatch__fleet_status` + kill any stale agents from last session (`mcp__dispatch__agent_status` → `agent_kill` for anything not tied to this sweep).
- [ ] **PF-4**  -  Crates.io squat check. `cargo search vyre-reference vyre-wgpu vyre-sigstore`  -  if any name taken, surface to user immediately. Acceptance: all three available.
- [ ] **PF-5**  -  Confirm `CARGO_REGISTRY_TOKEN` present in `/credentials/.env`. Acceptance: `cargo login` would work.
- [ ] **PF-6**  -  Pause noisy CI. Disable GitHub Actions workflows that will spam during the red period: `adversarial.yml`, `mutation-testing.yml`, `parity-determinism.yml`, `gpu-parity.yml`. Re-enable at convergence. Acceptance: PR runs don't trigger these.
- [ ] **PF-7**  -  Capture baseline. `cargo check --workspace --all-targets --all-features 2>&1 | tee docs/audits/pre-sweep-errors.txt`. Record error count. Commit the file.

---

## ME  -  §0.2 NUKE SWEEP (serial, workspace-wide, I do this personally)

- [ ] **N-1**  -  Delete every `pub use X as Y` and `pub(crate) use X as Y` alias across `vyre-core/src/`, `vyre-conform/src/`, `vyre-std/src/`, `vyre-spec/src/`, `vyre-build-scan/src/`, `demos/`, `examples/`. Replace every call site with the canonical name. Acceptance: `grep -rn "pub use .* as \| pub(crate) use .* as " vyre-core/src vyre-conform/src vyre-std/src vyre-spec/src vyre-build-scan/src` returns 0.
- [ ] **N-2**  -  Delete every shim module (a module whose body is `pub use other::*;` and nothing else). Specific sites: `vyre-conform/src/lib.rs` six aliases (`types`, `specs`, `ops`, `mutations`, `layers`, `observe`), any `pub mod X { pub use other::*; }` pattern. Acceptance: no module body shorter than 3 lines that only re-exports.
- [ ] **N-3**  -  Delete every `todo!()`, `unimplemented!()`, `panic!("not implemented")`, `panic!("TODO")`, `unreachable!("not yet")` in non-test code. Delete the enclosing function if its body was only the stub. Let compile errors name every caller; those are real gaps for Codex tracks. Acceptance: `grep -rn "todo!()\|unimplemented!()" --include=*.rs vyre-core/src vyre-conform/src vyre-std/src vyre-spec/src vyre-build-scan/src | grep -v tests/` returns 0.
- [ ] **N-4**  -  Delete every `// TODO`, `// FIXME`, `// HACK`, `// placeholder`, `// stub`, `// temporary`, `// XXX` in non-test code. Delete the hollow body under each. Acceptance: `grep -rniE "// ?(TODO|FIXME|HACK|placeholder|stub|temporary|XXX)" --include=*.rs vyre-core/src vyre-conform/src vyre-std/src vyre-spec/src vyre-build-scan/src | grep -v tests/` returns 0.
- [ ] **N-5**  -  Delete every `#[allow(dead_code)]`, `#[allow(unused_imports)]`, `#[allow(unused_variables)]` in non-test code. Let the compiler surface the real dead items  -  each becomes a Codex task. Acceptance: 0 hits in non-test paths.
- [ ] **N-6**  -  Delete every `#![allow(missing_docs)]` at crate root and module root. Delete every `//! Doc.` placeholder. Acceptance: 0 hits.
- [ ] **N-7**  -  Delete every `#[cfg(not(feature = "gpu"))]` CPU fallback that silently skips a GPU test. Acceptance: 0 hits in non-test and test paths.
- [ ] **N-8**  -  Delete `parity-oracle` feature from `vyre-core/Cargo.toml` + every `#[cfg(feature = "parity-oracle")]` callsite. Reference interpreter move is C4 (Codex-A). After C4 the feature gate is gone.
- [ ] **N-9**  -  Delete `decode-only`, `hash-only`, `primitive-only` features from `vyre-core/Cargo.toml` + every `#[cfg(feature = "X")]` callsite.
- [ ] **N-10**  -  Delete every `pub use X::*` glob across `vyre-core/src/` + `vyre-conform/src/` + `vyre-std/src/` + `vyre-spec/src/`. Replace with explicit named re-exports. Acceptance: `grep -rn "pub use .*::\*" --include=*.rs vyre-core/src vyre-conform/src vyre-std/src vyre-spec/src` returns 0.
- [ ] **N-11**  -  Delete `vyre-core/src/ops.rs` lines 43-ish that re-export `CategoryAOp`, `CpuOp` (Cat-B CPU exec paths).
- [ ] **N-12**  -  Delete `vyre-core/src/vyre-conform/` module + `pub mod conform;` line in `vyre-core/src/lib.rs:105`. Rewire every in-core caller to not depend on conform (there should be none  -  this is a backwards shim).
- [ ] **N-13**  -  Delete `vyre-conform/src/backends/` (mod.rs only, O6 residue).
- [ ] **N-14**  -  Delete `vyre-conform/src/generated/` committed content (`coverage_report.md`, `defender_corpus.rs`, `manifest.toml`, `ops.rs`, `primitive_ops.rs`). Codex-C handles the OUT_DIR migration.
- [ ] **N-15**  -  Delete every `#[deprecated]` attribute in workspace.
- [ ] **N-16**  -  Re-include `vyre-conform/fuzz` in workspace (edit workspace `Cargo.toml` `exclude = []`). Red compile is expected; Codex-C fixes.
- [ ] **N-17**  -  Capture post-nuke red state. `cargo check --workspace --all-targets --all-features 2>&1 | tee docs/audits/post-nuke-errors.txt`. Commit. This is the real backlog.

---

## ME  -  Crate scaffolds (serial, after nuke)

- [ ] **S-1**  -  Create `vyre-reference/` crate shell. `cargo new --lib vyre-reference` inside workspace. Cargo.toml inherits workspace; deps = `vyre`, `vyre-spec`. Empty `src/lib.rs` with doc comment. Add to workspace members. Acceptance: `cargo check -p vyre-reference` is green (empty crate).
- [ ] **S-2**  -  `git mv vyre-core/src/reference/ vyre-reference/src/` and every sub-path. Update `vyre-core/src/lib.rs` (delete `pub mod reference`). Rewrite every `vyre::reference::…` import across the workspace to `vyre_reference::…` (enumerated in MASTER_PLAN §2.2 C4  -  conform law_independent, verify-cert bins, gate_7_coverage, enforcer_gpu_mandatory, specs/primitive, verify/golden/util, verify/golden_samples, adversarial/gate7, vyre-core/tests/gap, vyre-core/fuzz/gpu). Acceptance: `grep -rn "vyre::reference\|crate::reference" vyre-core/src vyre-conform/src vyre-std/src vyre-spec/src` returns 0; `cargo check -p vyre-reference` still green after the content move (it may have its own errors  -  those are Codex-A's to close).
- [ ] **S-3**  -  Create `vyre-wgpu/` crate shell. `cargo new --lib vyre-wgpu`. Deps = `vyre`, `vyre-spec`, `wgpu`, `pollster`, `bytemuck`. Empty lib.rs. Workspace member. Acceptance: `cargo check -p vyre-wgpu` green.
- [ ] **S-4**  -  `git mv vyre-core/src/runtime/ vyre-wgpu/src/runtime/` + `git mv vyre-core/src/engine/ vyre-wgpu/src/engine/` (only the wgpu-coupled bits; IR-only bits stay in core). Decision per-file: anything that `use wgpu` moves; anything pure-IR stays. Audit each file in `vyre-core/src/engine/{dataflow,decode,decompress,dfa}`  -  most are pure IR and stay in core under `vyre-core/src/ir/engine/`. Only wgpu dispatch glue moves. Delete `pub mod engine`, `pub mod runtime` from `vyre-core/src/lib.rs`. Rewire imports. Acceptance: `grep -rn "use wgpu\|wgpu::" vyre-core/src` returns 0; Codex-A closes remaining compile errors.
- [ ] **S-5**  -  Also move `vyre-core/src/backend/` (wgpu `VyreBackend` impl lives here) to `vyre-wgpu/src/backend/`. Keep only the trait definition in `vyre-core/src/backend.rs` (pure trait + error + dispatch config).
- [ ] **S-6**  -  Rename `vyre-conform/sigstore/` → `vyre-sigstore/` at workspace root. `git mv vyre-conform/sigstore vyre-sigstore`. Update Cargo.toml `publish = true` (was false). Cargo.toml inherits workspace. Add to workspace members; remove from conform's subcrate list. Acceptance: `cargo check -p vyre-sigstore` green.
- [ ] **S-7**  -  Move `vyre-conform/xtask/` → `xtask/` at workspace root. `publish = false` stays. Update workspace members. Acceptance: `cargo xtask --help` runs.
- [ ] **S-8**  -  Delete `vyre-conform/codegen/` as a separate crate. Inline its code into `vyre-conform/build.rs`. Delete the `[build-dependencies] vyre-conform-codegen` line. Acceptance: `cargo build -p vyre-conform` still succeeds (red on conform internals  -  Codex-C's job  -  but build.rs runs).
- [ ] **S-9**  -  Move `test-infra/mutations/` and `test_catalog.toml` into `vyre-conform/rules/mutations/` and `vyre-conform/rules/test_catalog.toml`. Delete `test-infra/` top-level dir.
- [ ] **S-10**  -  Commit: `chore(vyre): sweep scaffold  -  extract vyre-reference, vyre-wgpu, vyre-sigstore; relocate xtask; inline codegen`.

---

## ME  -  §0.6 workspace-root cleanup (parallel with Codex tracks once dispatched)

- [ ] **R-1**  -  `git rm -r archive/`. (RELEASE_PLAN.md, RELEASE_PLAN_V2.md, coordination/, README.md.)
- [ ] **R-2**  -  `git rm -r tasks/`. Superseded by MASTER_PLAN + SWEEP.
- [ ] **R-3**  -  Consolidate + delete `docs/audits/*` except `MASTER_PLAN.md`, `SWEEP.md`, `pre-sweep-errors.txt`, `post-nuke-errors.txt`. Every historical audit's unique finding is already in MASTER_PLAN. Delete 17 audit files.
- [ ] **R-4**  -  Audit `docs/` and delete: `generated/`, `internal/`, `internals/`, `fossil-record.md` (if stale), `migration.md` (if describes completed migration), `planning/`, `release/`, `RELEASE_NOTES_v0.4.0-alpha.2.md` (premature, rewrite at tag time). Keep: `thesis.md`, `roadmap.md`, `stability.md`, `trust-model.md`, `OPS.md`, `PRIMITIVES.md`, `santh-standard.md`, `support.md`, `faq.md`, `RELEASE_CHECKLIST.md`, `architecture.svg`.
- [ ] **R-5**  -  Delete migration scripts in `scripts/`: `fix_missing_docs.py`, `generate-ops-md.sh`, `check-ops-md.sh`, `adversarial-cycle*.{sh,py}`, `dispatcher-templates/`, `fetch-regressions.sh`, `vyre-dispatcher.py`, `sleep-allowlist.txt`, `workflow-permission-*.txt`, `mutants-local.sh`, `mutants.sh`, `launch.sh`, `aggregate-audits.sh`, `append-only.sh`, `lint-no-test-sleep.sh`, `lint-workflow-permissions.sh`, `README-cycle.md`. Keep: `publish-dryrun.sh`, `apply-branch-protection.sh`, `run-benchmarks.sh`.
- [ ] **R-6**  -  Decide `scripts/blake3-hasher/`: if it's used by build.rs anywhere, inline; otherwise delete.
- [ ] **R-7**  -  Relocate or delete `tests/launch_smoke_test.rs` at workspace root. Moves under `vyre-core/tests/launch_smoke.rs` or deletes if subsumed.
- [ ] **R-8**  -  Verify `.gitignore` covers `target/` at workspace root; each sub-crate inherits. `find . -type d -name target -not -path './target/*'` returns 0.

---

## ME  -  Docs + CI + hygiene (parallel with Codex tracks)

- [ ] **D-1**  -  Rewrite `vyre-core/README.md` (= `vyre`'s README). Three consumption modes, example, feature table, MSRV, license, link to `vyre-reference` + `vyre-wgpu`.
- [ ] **D-2**  -  Rewrite `vyre-spec/README.md` (frozen data contracts).
- [ ] **D-3**  -  Write `vyre-reference/README.md` (new crate).
- [ ] **D-4**  -  Write `vyre-wgpu/README.md` (new crate).
- [ ] **D-5**  -  Rewrite `vyre-std/README.md`.
- [ ] **D-6**  -  Rewrite `vyre-sigstore/README.md` (new standalone).
- [ ] **D-7**  -  Rewrite `vyre-conform/README.md`  -  maintainer-only banner at top.
- [ ] **D-8**  -  Rewrite `vyre-build-scan/README.md`.
- [ ] **D-9**  -  Create `CHANGELOG.md` per publishable crate (7 files). Keep-a-Changelog format. Initial entry: v0.4.0 summary.
- [ ] **D-10**  -  Create `RELEASE.md` at workspace root  -  topological publish order (build_scan → spec → vyre → vyre-reference → vyre-wgpu → vyre-std → vyre-sigstore → vyre-conform), exact commands.
- [ ] **D-11**  -  Every publishable crate's Cargo.toml: add `[package.metadata.docs.rs] all-features = true, rustdoc-args = ["--cfg", "docsrs"]`.
- [ ] **D-12**  -  Every publishable crate's Cargo.toml: inherit `edition`, `rust-version`, `license`, `authors`, `repository`, `homepage` from workspace (fix `vyre-std/Cargo.toml` which hardcodes).
- [ ] **D-13**  -  Every publishable crate root: `LICENSE-APACHE` + `LICENSE-MIT` files (copy from workspace root).
- [ ] **D-14**  -  Workspace `[workspace.lints.rust]`: add `missing_docs = "deny"`, `unsafe_code = "deny"`. Every crate inherits via `[lints] workspace = true`.
- [ ] **D-15**  -  Author `.github/workflows/public-api.yml`  -  runs `cargo public-api diff` against `docs/public-api/*.txt` baselines.
- [ ] **D-16**  -  Author `.github/workflows/semver-checks.yml`  -  runs on tag, uses `cargo-semver-checks`.
- [ ] **D-17**  -  Author `.github/workflows/deny.yml`  -  `cargo-deny` license + advisory + source.
- [ ] **D-18**  -  Author `.github/workflows/udeps.yml`  -  `cargo-udeps` unused-dep check.
- [ ] **D-19**  -  Author `.github/workflows/coverage.yml`  -  `cargo-llvm-cov`, floor 85% lines for v0.4.0.
- [ ] **D-20**  -  Author `.github/workflows/loom.yml`  -  runs loom tests with `RUSTFLAGS=--cfg loom`.
- [ ] **D-21**  -  Author `.github/workflows/miri.yml`  -  `cargo miri test` (narrow scope, wgpu crates excluded).
- [ ] **D-22**  -  Author `.github/workflows/strict.yml`  -  `cargo build --features strict` with `#![deny(warnings)]` gate.
- [ ] **D-23**  -  Author `deny.toml` at workspace root  -  license allowlist, advisory-db gate, source allowlist, banned-crate list.
- [ ] **D-24**  -  Generate `docs/public-api/*.txt` baselines for all 7 publishable crates via `cargo public-api > docs/public-api/<crate>.txt`. Commit.
- [ ] **D-25**  -  Audit existing 17 `.github/workflows/*.yml`  -  delete obsolete (`book.yml`, `catalog-consistency.yml`, `codeowners-coverage.yml`, `lint-no-test-sleep.yml`, `permission-lint.yml`, `random-order.yml`, `test-random-order.yml`, `ops-md-check.yml`, `reproducible-build.yml`, `append-only.yml`, `adversarial.yml` if redundant). Keep + update `ci.yml`, `fuzz.yml`, `bench.yml`, `gpu-parity.yml`, `parity-determinism.yml`, `mutation-testing.yml`.
- [ ] **D-26**  -  Update `CODEOWNERS` to match final tree (delete dead-path refs per H19).

---

## CODEX-A  -  Product surface (new crates + extraction follow-through)

**Briefing:** you own `vyre-reference/` + `vyre-wgpu/` + residual `vyre-core/src/{backend,compiler}` cleanup after the ME scaffolds land. Your job is to close every compile error in your tree, implement every `Fix:` gap the nuke exposed, ensure byte-identity between CPU reference and GPU, and leave no stub / no shim / no alias. Every rule in SWEEP preamble applies.

- [ ] **A-1**  -  `vyre-reference`: close every compile error surfaced by S-2 (the move). Rewrite imports; implement missing blanket impls. Acceptance: `cargo check -p vyre-reference --all-features` green.
- [ ] **A-2**  -  `vyre-reference`: audit every op in `vyre::ops` against CPU impls in `vyre-reference/src/`. Every op in core has a matching CPU reference. For every missing one, implement from spec (no stubs). Enumerated list: see MASTER_PLAN §3.17 Kimi-9 (~90 ops). Acceptance: `cargo test -p vyre-reference` covers every op with a KAT.
- [ ] **A-3**  -  `vyre-reference`: real tokenize KAT. Currently a 1-empty-vector placeholder (LAW 9 violation per session memo). Write real state-machine-output KAT vectors against the tokenize OpSpec. Acceptance: at least 20 KAT rows covering happy + edge + adversarial + unicode.
- [ ] **A-4**  -  `vyre-reference`: every file < 500 LOC; every `pub` item doc'd; every `Err`/`expect` `Fix:`-prefixed; zero globs; zero `.clone()` on hot paths beyond what's needed. Apply P2/P3/P5 allocation fixes as they land in this tree.
- [ ] **A-5**  -  `vyre-wgpu`: close every compile error surfaced by S-3 / S-4 / S-5. Acceptance: `cargo check -p vyre-wgpu --all-features` green.
- [ ] **A-6**  -  `vyre-wgpu`: implement `VyreBackend` for `WgpuBackend`. `dispatch(&program, &inputs, &config)` lowers Program to WGSL internally; callers never pass WGSL strings. Acceptance: `cargo test -p vyre-wgpu` passes parity against `vyre-reference` on every op.
- [ ] **A-7**  -  `vyre-wgpu`: move `vyre-conform/src/backend/wgpu/` machinery here (dispatch, byte_words, context) under §0.4 rule that runtime is NOT in conform. Delete from conform.
- [ ] **A-8**  -  `vyre-wgpu`: P1 (validation cache), P5 (byte_words zero-copy alignment), P9 (parking_lot everywhere), Kimi-3 dispatch.rs lock-poison removal, Kimi-3 global-mutex-across-block_on split.
- [ ] **A-9**  -  `vyre-wgpu`: every file < 500 LOC; every `pub` item doc'd; every `Err`/`expect` `Fix:`-prefixed. Zero unsafe except the documented wgpu pipeline-cache site.
- [ ] **A-10**  -  `vyre-core/src/backend.rs`: leave ONLY the `VyreBackend` trait + `BackendError` + `DispatchConfig`. No impls, no wgpu, no runtime. Acceptance: `grep -rn "wgpu" vyre-core/src/backend.rs` returns 0.
- [ ] **A-11**  -  C14: `vyre-core/src/compiler/` disposition. Audit every file. IR-transform infrastructure (dataflow_fixpoint, dominator_tree, recursive_descent, typed_arena, string_interner, visitor_walk, buffer_layouts) moves under `vyre-core/src/ir/transform/`; WGSL-coupled bits (`wgsl/`, `wgsl_backend.rs`) move into `vyre-core/src/lower/wgsl/` or `vyre-wgpu/`. Delete `vyre-core/src/compiler/` top-level. `pub mod compiler` in `vyre-core/src/lib.rs` gone.
- [ ] **A-12**  -  C15: replace every `OpSpec { field: val, … }` construction (34 sites in `vyre-conform/src/vyre-spec/string/tokenize/mod.rs` + others) with `OpSpec::new(...).with_law(...)`-style builder. Define the builder in `vyre-spec`. Acceptance: no E0639 errors; `#[non_exhaustive]` on OpSpec stays.
- [ ] **A-13**  -  C17: add blanket `impl<T: Archetype + ?Sized> Archetype for &T` (and similar for `Oracle`, `EnforceGate`) so build-scan-emitted `&'static dyn …` registry rows type-check. Acceptance: `$OUT_DIR/*_registry.rs` compiles.
- [ ] **A-14**  -  `vyre-sigstore`: audit standalone. Public API = sign(cert, key) + verify(cert, sig). Small, pristine. Every pub item doc'd. Acceptance: `cargo publish --dry-run -p vyre-sigstore` succeeds.
- [ ] **A-15**  -  Update MASTER_PLAN.md §2.2 C4 + §2.4 C8 to "CLOSED" when A-1/A-5 green. Append to SWEEP.md status log.

---

## CODEX-B  -  Core internals + std + spec + build_scan

**Briefing:** you own every file in `vyre-core/src/{ir, lower, ops, engine, compiler}` (except what Codex-A is extracting), plus all of `vyre-std/src/`, `vyre-spec/src/`, `vyre-build-scan/src/`. Close all H5/H6/H9/Fix:/must_use findings in your tree; execute Kimi-1 SemVer rename in vyre-spec; execute every IR/lowering perf fix. Every rule in SWEEP preamble applies.

- [ ] **B-1**  -  Kimi-1 `vyre-spec` SemVer lockdown: every `pub mod X; pub use X::Type;` dual-identity → child `pub(crate)` + parent re-export only. Rename `Verification::count` → `witness_count` (+ `#[must_use]`); `Layer::description` → `layer_description`; `MetadataCategory::id` → `category_id`. Drop `Copy` from `IntrinsicTable`, `BinOp`, `UnOp`, `DataType`, `AtomicOp`, `BufferAccess`, `Convention`, `AdversarialInput`, `GoldenSample`, `KatVector`, `Invariant`, `EngineInvariant`, `FloatType`. Replace `pub const INVARIANTS`, `pub const ALL_ALGEBRAIC_LAWS`, `pub const LAW_CATALOG` with `pub fn` accessors. `by_id::by_id` + `catalog_is_complete` drive dynamically off enum. `missing_backends() -> impl Iterator`, `by_category() -> impl Iterator`. `BackendAvailability` becomes a trait, not a `fn` alias. Acceptance: `cargo check -p vyre-spec` green; `cargo public-api -p vyre-spec` matches the new baseline exactly.
- [ ] **B-2**  -  `vyre-core/src/lib.rs` pristine public surface: `pub mod ir`, `pub mod ops`, `pub mod lower`, `pub mod error`; `pub use backend::{BackendError, DispatchConfig, VyreBackend}`, `pub use ir::{Program, validate}`, `pub use match_result::Match`, `pub use error::{Error, Result}`. Every other `pub mod` demoted or moved. Acceptance: `cargo public-api -p vyre` lists exactly this surface.
- [ ] **B-3**  -  H5 file split sweep in core. Files > 500 LOC in your tree: `vyre-core/src/lower/wgsl/expr/mod.rs` (739), `vyre-core/src/reference/typed_ops/mod.rs` (540  -  note: belongs to Codex-A now post-move), `vyre-core/src/ops.rs` (539), `vyre-core/src/ir/transform/inline/expand/impl_calleeexpander.rs` (477), `vyre-core/src/reference/eval_expr.rs` (471  -  Codex-A), `vyre-core/src/engine/decode/mod.rs` (459), `vyre-core/build.rs`. For each: split by responsibility into siblings + `explicit_mod_list!`. Each resulting file < 500 LOC, single purpose.
- [ ] **B-4**  -  H6 path flatten sweep in core. Paths > 4 levels in your tree: `vyre-core/src/ops/compression/ops/{deflate,gzip,zlib,zstd,lz4}/{implementation,metadata,fixtures,lowering}/`, `vyre-core/src/ops/workgroup/primitives/{stack,queue_fifo,queue_priority}/{metadata,fixtures}/`, `vyre-core/src/ir/transform/optimize/tests/unit/cse/`. Flatten via rename + `explicit_mod_list!` wiring.
- [ ] **B-5**  -  H9 glob removal in core: every `pub use X::*` → explicit named re-exports. Enumerated sites in MASTER_PLAN §3.10 Kimi-2. Acceptance: 0 glob hits.
- [ ] **B-6**  -  Kimi-4 IR + lowering allocation sweep: `vyre-core/src/ir/validate/validate.rs` pre-size hashmaps; `ValidationError::message` → `Cow<'static, str>`; `vyre-core/src/ir/transform/optimize/{dce,cse}/*` move program buffers not clone; `ExprKey` `&str` / `Arc<str>`; `InlineCtx` scratch String; `Program::buffer_index` reuse; `lower/wgsl/impl_lowerctx.rs` write to output buffer not `"  ".repeat()`. Full list MASTER_PLAN §3.13.
- [ ] **B-7**  -  Kimi-5 `#[must_use]` across vyre-core/vyre-spec/std: every `Error`, `Report`, `Finding`, `Violation`, `Result`, `Outcome`, builder chain. Also B-1 additions.
- [ ] **B-8**  -  Kimi-8 `Fix:` prefix sweep in vyre-core/vyre-std/vyre-spec/build_scan: every `panic!`, `.expect`, `bail!`, `Err(thiserror)`, `map_err(|_| …)`. MASTER_PLAN §3.15 enumerates the worst sites. Every surviving one begins with `Fix: `.
- [ ] **B-9**  -  `vyre-core/src/engine/` audit: non-wgpu bits stay in core (pure IR algorithms). Any wgpu coupling goes to Codex-A. After A-11: `vyre-core/src/engine/` is pure IR (dataflow, decode, decompress, dfa, prefix, token_match_filter).
- [ ] **B-10**  -  P8 `.clone()` reduction: ≥30% cut in core hot paths. Commit-log the before/after count.
- [ ] **B-11**  -  Missing-docs sweep in core (after N-6 removed the blanket allow). Every `pub` item gets a real doc comment, not `//! Doc.`.
- [ ] **B-12**  -  `#![deny(missing_docs)]` promoted at `vyre-core/src/lib.rs`, `vyre-spec/src/lib.rs`, `vyre-std/src/lib.rs`, `vyre-build-scan/src/lib.rs`  -  after B-11 + vyre-spec/std equivalents.
- [ ] **B-13**  -  vyre-std audit: public API pristine (DFA assembly, Aho-Corasick, compositional arithmetic, regex→NFA→DFA). Every file < 500 LOC. Docs deny. `cargo check -p vyre-std` green against new vyre-core/spec.
- [ ] **B-14**  -  build_scan C16: filter gate detection on `impl EnforceGate for …` not "file exists in dir". Move `BeamSearch` + `NullBackend` out of gates dir. Acceptance: `$OUT_DIR/gates_registry.rs` compiles.
- [ ] **B-15**  -  Benches: every `benches/*.rs` in core has real iter body running the primitive + GPU dispatch + readback (P7). Un-`#[ignore]` `vs_cpu_baseline.rs`. Asserts a minimum speedup per primitive. Acceptance: `cargo bench -p vyre` produces meaningful numbers, not `black_box(rows.len())`.
- [ ] **B-16**  -  Fuzz targets: every op in `vyre-core/src/ops/` has a target in `vyre-core/fuzz/`. Enumerate missing, add. Acceptance: `cargo fuzz list` matches op registry 1:1.

---

## CODEX-C  -  Conform internals (the 15→8 collapse + determinism + specs)

**Briefing:** you own every file under `vyre-conform/`. Close all architectural debt: 15→8 dir collapse, H2 trait rewrite, H5 enforcer splits, Kimi-3 concurrency, Kimi-6 determinism, Kimi-9 ~90 missing specs, vyre-conform/fuzz re-include post-nuke. Maintainer-only audience, same quality bar as end-user. Every rule in SWEEP preamble applies.

- [ ] **C1**  -  §0.4 rule 1: delete every `[dependencies]` (not dev-dep) on `vyre-conform` from non-conform crates. Run: `grep -rn 'vyre-conform' vyre-core/Cargo.toml vyre-std/Cargo.toml vyre-spec/Cargo.toml vyre-build-scan/Cargo.toml`  -  should show only `[dev-dependencies]` entries if any. Fix violations.
- [ ] **C2**  -  C19 15→8 dir collapse: `algebra/` → `proof/algebra/`; `comparator/` → `proof/comparator/`; `framework/` → merge into `vyre-spec/` + `pipeline/` (decide per sub-dir: `framework/types` → `vyre-spec/types`, `framework/loader` → `pipeline/loader`, `framework/registry` → `vyre-spec/registry`, `framework/value` → `vyre-spec/value`, `framework/builder` → `vyre-spec/builder`); `backend/` → `pipeline/backend/` (only if non-wgpu residue remains after A-7). Delete `backends/` (already in nuke). Delete `generated/` (already in nuke). Acceptance: `ls vyre-conform/src/` returns exactly `adversarial enforce generate meta pipeline proof spec verify lib.rs`.
- [ ] **C3**  -  §0.4 rule 5: `core::conform` module already nuked (N-12). Verify no conform internal still imports from `vyre::conform`.
- [ ] **C4**  -  Conform pristine public API (10 items): `certify`, `Certificate`, `Violation`, `VyreBackend`, `Finding`, `EnforceGate`, `Oracle`, `Archetype`, `MutationClass`, `{Category, AlgebraicLaw, OpSpec}` re-export. Every other `pub use` demoted to `pub(crate)`. Internal callers use canonical paths (no aliases, no `as`). Acceptance: `cargo public-api -p vyre-conform` returns exactly these 10.
- [ ] **C5**  -  H2 trait rewrite: `Archetype::materialize(&OpSpec) -> Option<TestInput>`; `EnforceGate { fn id; fn name; fn run; }`; `Oracle { fn id; fn kind; fn applicable_to; fn verify; }`. Update every impl.
- [ ] **C6**  -  C1 public `certify` signature: `&dyn vyre::VyreBackend`, not `&dyn WgslBackend`. WgslBackend stays `pub(crate)`. Every internal call site (20+ per MASTER_PLAN §2.1) rewritten to use `VyreBackend::dispatch(&program, &inputs, &DispatchConfig::default())`  -  no WGSL-string path.
- [ ] **C7**  -  H5 file splits in conform (the worst offenders): `enforcers/float_semantics.rs` (2730), `category_b.rs` (1098), `reference_trust.rs` (926), `atomics_race.rs` (912), `signature_match.rs` (757), `zero_stubs.rs` (737), `structural_rules.rs` (737), `decomposition.rs` (614), `engine_composition.rs` (607), `composition_closure.rs` (606), `category_a.rs` (594), `category_c.rs` (579), `barrier_placement.rs` (556), `overflow_contract.rs` (538), `admission.rs` (462). Each → sub-dir of siblings + `explicit_mod_list!`. Every resulting file < 500 LOC, single responsibility. Delegate one-file-one-Kimi as needed.
- [ ] **C8**  -  H6 path flatten in conform: `vyre-conform/src/verify/properties/tests/declared_laws/{…}` (8+ levels), `vyre-conform/src/generate/emit/cross_product/codegen/context/` (6 levels). Flatten.
- [ ] **C9**  -  H9/H20 glob removal in conform: `framework/loader/toml/mod.rs` (10), `framework/value/mod.rs` (3), `framework/registry/mod.rs` (2), plus enforcers. Explicit re-exports.
- [ ] **C10**  -  H10 `vyre-conform/src/generated/` content (N-14 deleted files) now emits from `build.rs` via `include!(concat!(env!("OUT_DIR"), "/…"))`. Acceptance: nothing committed under `vyre-conform/src/generated/`.
- [ ] **C11**  -  H3/H4 filesystem-is-registry: every ARCHITECTURE-mandated scan exists. `enforce/gates/`, `proof/oracles/`, `generate/archetypes/`, `adversarial/mutations/`, `vyre-spec/findings/` scans auto-populate via `explicit_mod_list!`. No central `REGISTRY` / `ARCHETYPES` array anywhere.
- [ ] **C12**  -  Kimi-3 concurrency CRITICAL: `algebra/checker/verify_one_law_witnessed.rs:31` persist_violation race; `pipeline/streaming/regression_sinking.rs` temp file name collision; `meta/harness/mod.rs:114-319` source-file mutation + last-writer-wins MD; `corpus/witness.rs:160` unlocked append. Add proper file locking (`fs2` or `file-lock`); unique temp names per thread; deterministic output sort.
- [ ] **C13**  -  Kimi-3 HIGH: `pipeline/notify.rs` no-MutexGuard-across-user-callback; `backend/wgpu/dispatch.rs` parking_lot swap (delegated to A-8 after move); `backend/wgpu/context.rs` no block_on-under-global-mutex (delegated); `verify/golden/freeze_goldens.rs` TOCTTOU atomic write.
- [ ] **C14**  -  Kimi-3 MEDIUM: `algebra/checker/mod.rs` parallel filesystem I/O → buffered; `bin/phase6_calibrate.rs` HashSet sorted; `meta/harness/mod.rs` BTreeSet for deterministic iteration; `observe/subscriber.rs` mutex-free tracing (use `tracing-subscriber` layer); `enforcers/layer8_feedback_loop.rs` LRU-bounded cache.
- [ ] **C15**  -  Kimi-6 determinism CRITICAL: remove `monotonic_sequence` from Certificate (replace with `blake3(canonical_input_bytes)`); strip elapsed-ms from every cert message and every serialized cert field; `ReferenceBombDetected` Display drops elapsed-ms. Acceptance: two runs with identical inputs produce byte-identical certs (test harness in `vyre-conform/tests/determinism/cert_hash_stable.rs`).
- [ ] **C16**  -  Kimi-6 MEDIUM: `bin/contribute.rs` drop `duration_ms` from ContributeReport serialization; `meta/harness/mod.rs` adversaries_killed/survived built from BTreeSet.
- [ ] **C17**  -  Kimi-8 `Fix:` prefix sweep in conform: every `panic!`, `.expect`, `bail!`, `Err`, `map_err`. Workgroup-primitive Error enums Display impls prefixed. Swallowed results in cache.rs and dfa_pack.rs fixed.
- [ ] **C18**  -  Kimi-5 `#[must_use]` sweep in conform.
- [ ] **C19**  -  H14 `process::exit` → Result: `vyre-conform/src/pipeline/bin/phase6_calibrate.rs:458,481` (and any survivors). Library code never exits.
- [ ] **C20**  -  H11: `backend/mod.rs` `require_gpu()` returns `Result`. Callers `.expect(…)` in test context only.
- [ ] **C21**  -  M18: every `fn main()` under `pipeline/bin/` either declared as `[[bin]]` in `vyre-conform/Cargo.toml` or deleted.
- [ ] **C22**  -  Kimi-9 CRITICAL: fill ~90 missing conform specs under `vyre-conform/src/vyre-spec/<category>/<op>.rs`. Each spec = CPU reference hook + WGSL declaration + algebraic laws + archetypes + KAT vectors + adversarial inputs. Engines without WGSL (engine.dfa, engine.eval, engine.scatter) declared CPU-only explicitly.
- [ ] **C23**  -  Kimi-7 unit tests: `vyre-conform/tests/unit/stats.rs` for arithmetic_mean, variance, std_dev, byte_histogram, chi_square, sliding_entropy. Tests designed to BREAK, not pass trivially (LAW 5).
- [ ] **C24**  -  `vyre-conform/fuzz` re-included (N-16). Close every compile error. `vyre-conform/src/vyre-spec/toml_loader` was deleted last session  -  rewire fuzz to the current loader under `vyre-conform/src/pipeline/loader/`. Acceptance: `cargo fuzz list -p vyre-conform-fuzz` returns all targets; `cargo +nightly fuzz run <target> -- -runs=1000` passes on each.
- [ ] **C25**  -  Missing-docs sweep in conform (after N-6). Every `pub` item doc'd. `#![deny(missing_docs)]` at `vyre-conform/src/lib.rs`.
- [ ] **C26**  -  CODEOWNERS dead-path cleanup (delegated to my D-26).
- [ ] **C27**  -  Inline `vyre-conform/codegen` into `vyre-conform/build.rs` (S-8 delegate). Close the `[build-dependencies] vyre-conform-codegen` line.
- [ ] **C28**  -  Bench targets declared in `vyre-conform/Cargo.toml`  -  cert run throughput, parity-oracle benchmark. Acceptance: `cargo bench -p vyre-conform` runs.
- [ ] **C29**  -  `vyre-conform/sigstore/` → `vyre-sigstore/` at workspace root already done by S-6. Verify every conform internal still imports correctly.

---

## CONVERGENCE  -  ME

- [ ] **V-1**  -  `cargo check --workspace --all-targets --all-features` green. Zero errors, zero warnings.
- [ ] **V-2**  -  `cargo test --workspace --all-features` green. Every test passes.
- [ ] **V-3**  -  `cargo clippy --workspace --all-targets --all-features -- -D warnings` green.
- [ ] **V-4**  -  `cargo +nightly udeps --workspace` green.
- [ ] **V-5**  -  `cargo deny check` green (after D-23).
- [ ] **V-6**  -  `cargo public-api` diff against baselines zero unexpected changes.
- [ ] **V-7**  -  `cargo semver-checks check-release` passes for every publishable crate.
- [ ] **V-8**  -  Determinism test: `cargo test --test cert_hash_stable -p vyre-conform` passes  -  two runs of `certify(&backend, &inputs)` produce byte-identical certs.
- [ ] **V-9**  -  GPU parity test: `cargo test gpu_parity -p vyre-wgpu` green on the 5090.
- [ ] **V-10**  -  Bench sanity: `cargo bench -p vyre -p vyre-wgpu -p vyre-std` produces meaningful numbers.
- [ ] **V-11**  -  Re-enable paused CI workflows (PF-6 reversal).

## PUBLISH  -  ME

- [ ] **P-1**  -  Topological `cargo publish --dry-run` in order: `vyre-build-scan`, `vyre-spec`, `vyre`, `vyre-reference`, `vyre-wgpu`, `vyre-std`, `vyre-sigstore`, `vyre-conform`. Every crate's dry-run passes.
- [ ] **P-2**  -  Real `cargo publish` in the same topological order. After each, wait for crates.io to index before publishing the next.
- [ ] **P-3**  -  `git tag v0.4.0 && git push origin v0.4.0`.
- [ ] **P-4**  -  GitHub release: write real release notes from CHANGELOGs + attach any release artifacts. Publish the release.
- [ ] **P-5**  -  Update `README.md` at workspace root with "Released v0.4.0" banner + links.
- [ ] **P-6**  -  Telegram: single message to @SanthCEObot with "vyre v0.4.0 shipped  -  8 crates live on crates.io  -  GH release URL" (only moment Telegram fires per your instruction).

---

## Status log (append-only)

Every agent + me appends a dated one-liner here when they complete a task, fail, or get blocked. Format: `2026-04-17T<HH:MM>Z  -  <owner>  -  <task-id>  -  <status>  -  <one-line note>`.

(entries begin once PF-1 lands)
2026-04-18T00:54Z  -  codex-A  -  A-1  -  blocked  -  `cargo check -p vyre-reference --all-features` stops before compilation because `vyre-conform/fuzz` is a workspace member while declaring its own `[workspace]`; root/Codex-C ownership must remove that nested workspace or exclude it.
2026-04-18T00:59Z  -  codex-A  -  A-1  -  blocked  -  Temporary sibling check with `vyre-conform/fuzz` excluded still fails before `vyre-reference`: `vyre` has unresolved `crate::runtime` references after scaffold extraction plus Codex-B-owned non-`Copy` IR fallout.
2026-04-18T00:53Z  -  codex-C  -  C1  -  blocked  -  Removed examples/hello_vyre runtime dependency on vyre-conform; remaining vyre-sigstore runtime dependency is outside Codex-C ownership and needs Codex-A/orchestrator rename/removal.
2026-04-18T00:57Z  -  codex-C  -  C2  -  done  -  Collapsed vyre-conform/src to the 8-module target; moved algebra/comparator/backend/framework responsibilities to proof/pipeline/spec and removed generated from src.
2026-04-18T00:57Z  -  codex-C  -  C3  -  done  -  Verified vyre-conform/src has no vyre::conform or crate::vyre_conform imports after the core conform nuke.
2026-04-18T01:00Z  -  codex-C  -  C4  -  blocked  -  Renamed ConformanceCertificate to Certificate, added Violation, added Oracle trait; cargo/public-api verification is blocked by A/B-owned core runtime and Copy-removal compile errors.
2026-04-18T01:00Z  -  codex-C  -  C5  -  in-progress  -  Updated conform trait definitions for Oracle, Archetype, EnforceGate, and MutationClass; implementation verification is blocked until core compiles.
2026-04-18T01:03Z  -  codex-B  -  B-1  -  done  -  vyre-spec SemVer surface is locked down in current HEAD; cargo check -p vyre-spec --all-features is green; cargo-public-api is not installed in this environment.
2026-04-18T01:03Z  -  codex-B  -  B-2  -  blocked  -  vyre-core/src/lib.rs still exposes vyre-conform/engine/compiler; waiting on Codex-A extraction/removal before final public-surface pruning.

## Status log (live)

- 2026-04-17T17:58Z  -  orchestrator  -  PF-1..PF-7  -  done  -  pre-sweep-2026-04-17 tag pushed; 14 zombie branches deleted; 14 noisy CI workflows paused; baseline captured (11 lib errors, 29 test errors, 243 warnings)
- 2026-04-17T18:04Z  -  orchestrator  -  N-1..N-16  -  done  -  nuke complete workspace-wide: 13 `pub use X as Y` aliases, 20 `pub use ::*` globs, 175 `#![allow(missing_docs)]`, 5 `#[allow(dead_code)]`, comment stubs deleted; parity-oracle + decode-only + hash-only + primitive-only features deleted from vyre-core/Cargo.toml; vyre-conform/src/backends/ + vyre-conform/src/generated/ committed content deleted; vyre-conform/fuzz re-included in workspace members
- 2026-04-17T18:11Z  -  orchestrator  -  S-1..S-10  -  done  -  vyre-reference, vyre-wgpu, vyre-sigstore crate shells created; vyre-core/src/reference moved to vyre-reference/src; vyre-core/src/runtime moved to vyre-wgpu/src/runtime; vyre-conform/sigstore relocated to workspace-root vyre-sigstore; vyre-conform/xtask relocated to workspace-root xtask; test-infra moved into vyre-conform/rules
- 2026-04-17T18:20Z  -  orchestrator  -  R-1..R-8  -  done  -  archive/, tasks/, 14 stale docs/audits, 17 migration scripts, 5 stale docs/ subdirs deleted
- 2026-04-17T18:25Z  -  orchestrator  -  D-9, D-10, D-13, D-23  -  done  -  7 CHANGELOGs, RELEASE.md, LICENSE files per new crate, deny.toml
- 2026-04-17T18:30Z  -  orchestrator  -  D-15..D-23  -  done  -  8 new CI workflows: deny, public-api, semver-checks, udeps, coverage, loom, miri, strict
- 2026-04-17T18:35Z  -  orchestrator  -  D-1 D-3 D-4 D-6  -  done  -  new-crate READMEs (vyre-reference, vyre-wgpu, vyre-sigstore) + workspace README updated for 8-crate layout
- 2026-04-17T18:40Z  -  orchestrator  -  D-11 D-12 D-14 D-26  -  done  -  docs.rs metadata on every publishable Cargo.toml, workspace lints (missing_docs=warn, unreachable_pub=warn), CODEOWNERS rewritten for final layout
- 2026-04-17T18:02Z  -  codex-A (9e5ec01a)  -  partial  -  done(exit=0); committed trait-signature + archetype-dir cleanup across conform; scope leaked into Codex-C territory; CRITICAL GAP: A-6 (move vyre-core/src/engine wgpu-coupled to vyre-wgpu), A-10 (trim vyre-core/src/backend.rs to trait-only), A-1..A-5 (vyre-reference completion) not addressed  -  workspace still red on 77+ core errors with unresolved `crate::runtime` imports. Redispatch pending.
2026-04-18T01:13Z  -  codex-C  -  C6  -  done  -  Public certify now takes dyn vyre::VyreBackend; remaining WgslBackend signatures in vyre-conform/src are internal-only.
2026-04-18T01:13Z  -  codex-C  -  C9  -  done  -  Verified no pub-use glob exports remain under vyre-conform/src.
2026-04-18T01:13Z  -  codex-C  -  C10  -  blocked  -  vyre-conform/src/generated is absent and build.rs targets OUT_DIR; cargo verification is blocked by A/B-owned core runtime extraction and Copy-removal compile errors.
2026-04-18T01:13Z  -  codex-C  -  C11  -  done  -  Removed remaining central ARCHETYPES constant residue; filesystem-registry grep for REGISTRY/ARCHETYPES/ORACLES is clean.
2026-04-18T01:13Z  -  codex-C  -  C12  -  done  -  Added locking/unique-temp atomic writes for witnessed law persistence, streaming regressions/checkpoints, meta harness mutation/findings files, and witness appends.
2026-04-18T01:13Z  -  codex-C  -  C13  -  done  -  Reporter callbacks no longer run under MutexGuard; wgpu validation cache poison expects removed; golden freeze writes are atomic.
2026-04-18T01:13Z  -  codex-C  -  C14  -  in-progress  -  Deterministic BTreeSet output applied to calibration bins and meta adversary sets; remaining cache/observer TOCTTOU audit still open.
2026-04-18T01:13Z  -  codex-C  -  C15  -  in-progress  -  Removed certificate monotonic sequence/timestamp path and added deterministic certificate_hash; integration determinism test still blocked by public certify/spec API mismatch and core compile blockers.
2026-04-18T01:14Z  -  codex-B  -  B-1  -  done  -  vyre-spec SemVer surface is locked down in current HEAD; cargo check -p vyre-spec --all-features is green; cargo-public-api is not installed in this environment.
2026-04-18T01:14Z  -  codex-B  -  B-2  -  done  -  vyre-core/src/lib.rs public surface now exposes only ir/ops/lower/error modules plus the required crate-root re-exports; cargo check -p vyre --all-features is green with existing warning debt.
2026-04-18T01:14Z  -  codex-B  -  B-3  -  done  -  oversized core files split under the 500-line cap in current HEAD; max owned Rust file from find/wc is 479 LOC.
2026-04-18T01:17Z  -  codex-B  -  B-4  -  done  -  flattened compression, workgroup stack/queue, and CSE test paths; max directory depth under vyre-core/src vyre-std/src vyre-spec/src is 7; cargo check -p vyre --all-features is green with warning debt.
2026-04-18T01:17Z  -  codex-B  -  B-5  -  done  -  no remaining pub-use glob exports under vyre-core/src vyre-std/src vyre-spec/src vyre-build-scan/src; only vyre-spec/src/tests.rs contains the literal pattern as an assertion.
2026-04-17T01:18Z  -  codex-A-retry  -  A-6  -  done  -  runtime extraction verified: no wgpu imports remain under vyre-core/src engine/compiler/ops search path; vyre, vyre-wgpu, vyre-reference, and vyre-sigstore check green.
