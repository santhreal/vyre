# Vyre orphan-symbol + abandonment audit (2026-04-30)

Audit cleanup A15. Format per CLAUDE.md: `SEVERITY | path:line | symbol/file | finding | recommended action`.

## Summary

vyre's source code is genuinely well-disciplined on the dimensions CLAUDE.md cares most about:

- **Zero `TODO`/`FIXME` markers** across all 1736 `.rs` files.
- **Zero `unimplemented!()`/`todo!()`/`panic!("not implemented")` stubs** in any source file.
- **Zero `#[ignore]` markers** on any test.

This is a real signal that the codebase doesn't have the deferral culture A15 was set up to detect. The findings below are structural debts surfaced during A1–A14 that need explicit triage in A16, plus a small set of wildcard re-exports caught by the existing `vyre-foundation/tests/organization_contracts.rs` test.

## Pre-existing test failures (caught before A15 started)

**HIGH | `vyre-foundation/tests/bench_corpus_duplication.rs:21` | `checked_in_corpus_duplicates_match_manifest_policy`** | Test invokes `python3 benches/competition/scripts/check_corpora.py`, which doesn't exist on disk. Failed before A1 began. | **Re-wire**: either restore the missing script or delete the test (the latter requires user approval  -  see CLAUDE.md "two incidents on 2026-04-12 destroyed 1400 + 584 LOC of working code on bad 'dead' verdicts"). The script's git history (`git log --diff-filter=D --name-only -- benches/competition/scripts/check_corpora.py`) will tell us which.

**HIGH | `vyre-foundation/tests/organization_contracts.rs:56` | `foundation_wildcard_pub_reexports_are_baselined`** | `src/lib.rs:83 pub use crate::ir_eval::*;`  -  wildcard re-export not in baseline. | **Re-wire**: the test is enforcing "no new wildcard re-exports without baselining". The wildcard at lib.rs:83 (now `pub use crate::runtime::ir_eval::*;` after A12) needs to either (a) be added to the test's baseline allowlist, or (b) be replaced with explicit named re-exports. Option (b) is the structurally-correct answer.

**HIGH | `vyre-foundation/tests/organization_contracts.rs:452` | `workspace_wildcard_pub_reexports_are_baselined`** | Same `ir_eval::*` violation + `vyre-libs/src/lib.rs:302 pub use vyre_driver::self_substrate::*;`  -  wildcard re-export at the libs level. | **Re-wire**: A10 changed the underlying `vyre_driver::self_substrate` from `pub mod self_substrate;` to `pub use vyre_self_substrate as self_substrate;`. The downstream `vyre-libs/src/lib.rs:302` wildcard re-export still works but is even less specific now. Replace with explicit named re-exports.

**HIGH | `vyre-foundation/tests/organization_contracts.rs:?` | `foundation_inline_test_modules_are_baselined`** | Pre-existing inline-test-module violation. | **Re-wire**: identify the inline `mod tests { ... }` blocks the test flagged + either move them to sibling `_tests.rs` files (the project convention) or add to baseline.

## Crate naming + dep-direction debts

**HIGH | `vyre-runtime/src/megakernel/c_frontend.rs` (1299 LOC)** | The file's own header comment (line 2): "*COMPLETED: gemini-cli 2026-04-30 T-06* NOTE: this module is flagged for eviction from the substrate (Phase 1 substrate cut). C-language semantics belong in vyre-libs-c, not vyre-runtime. Do not extend this file; new C-frontend logic goes in the consumer crate." | **Re-wire**: A14 attempted the move but hit the substrate-dep-cycle blocker (parsing/core, region, harness need to hoist first). When A14's prerequisite is done, this file moves to `vyre-c-frontend` along with the rest. Until then the eviction marker is a known-open finding.

**MEDIUM | `vyre-libs/src/parsing/c/parse/vast.rs` (8721 LOC)** | C-parser code in the LEGO substrate's neighbor crate. Marker for the A14 substrate-frontend separation. | **Document**: open until A14's prerequisite (hoisting parsing/core etc) lands.

## Files-over-cap (open A13 work)

These are explicitly documented in A13's task description as known-open. Listing here for A16's triage queue:

- `vyre-foundation/src/optimizer/scheduler.rs` (1161 LOC)  -  needs functional split per impl-method-concern.
- `vyre-runtime/src/megakernel/planner.rs` (1387 LOC)  -  fusion / layout / dispatch_chain / cost.
- `vyre-runtime/src/pipeline_cache.rs` (1096 LOC)  -  key / store / eviction / telemetry.
- `vyre-driver-wgpu/src/emit/naga_emit/expr.rs` (1463 LOC)  -  per Expr variant family.
- `vyre-driver-wgpu/src/emit/naga_emit/mod.rs` (1271 LOC)  -  per emit phase.
- `vyre-driver-cuda/src/codegen/ctx.rs` (1538 LOC)  -  scope / register / expr_emit / node_emit.

## Apparent orphans (found during A15 sweep)

Caveat: `vyre-foundation/tests/organization_contracts.rs` has a `no_root_stray_plan_docs` style enforcement test for source-tree organization, but no test specifically enforces "every src/*.rs is mentioned in a parent mod.rs". The list below was found by walking `find . -path '*/src/*.rs'` and checking for `mod <basename>` references. Many false positives because the parent uses `pub(crate) mod <name>` or `#[path = "x_tests.rs"] mod tests;` which the trivial regex doesn't match.

**MEDIUM | `xtask/src/measurement_gate.rs`, `xtask/src/introspection.rs`, `xtask/src/perf_inventory_wave1.rs`** | Not visibly declared from xtask/src/main.rs. Likely orphan binaries from earlier xtask exploratory work. | **Investigate**: if `cargo run -p xtask -- <subcommand>` doesn't reach them, they're abandonded  -  re-wire into `main.rs` subcommand dispatch or remove with user approval.

**MEDIUM | `vyre-libs/src/test_migration.rs`** | Not visibly declared. | **Investigate**: likely a transient migration helper from a past refactor. Either re-wire to lib.rs or remove with user approval.

**MEDIUM | `vyre-libs/src/security/family_mask.rs`, `flow_composition.rs`** | Not visibly declared from `vyre-libs/src/security/mod.rs`. | **Investigate**: security module's mod.rs needs review.

**MEDIUM | `vyre-driver-wgpu/src/config.rs`, `vyre-driver-wgpu/src/pipeline_bindings.rs`** | Not visibly declared from vyre-driver-wgpu/src/lib.rs. | **Investigate**: lib.rs likely re-exports something else; verify or re-wire.

**MEDIUM | `vyre-reference/src/dual_impls.rs`** | Not visibly declared. | **Investigate** vyre-reference/src/lib.rs.

**MEDIUM | `xtask/src/quick_cache/{mutation_cache_json, json_escape, quick_mutation, cached_outcome, write_and_commit, nibble}.rs`** | The 6 files in xtask/src/quick_cache/ look like a self-contained subsystem. Verify quick_cache/mod.rs correctly declares them. | **Investigate**: if quick_cache/ exists at all and is wired from xtask/src/main.rs.

## Existing harmless warnings

**LOW | `vyre-self-substrate/src/zx_rewrite.rs:12` | `unused_imports: ZxColor and ZxSpider`** | Pre-existing unused-imports warning. Not a hard error. | **Sweep**: remove unused imports.

## Action items rolling into A16

A16 will take each MEDIUM/HIGH finding above and apply one of: (re-wire), (delete-after-proof + user approval), or (document-as-public-API). The re-wire is the default per CLAUDE.md "DON'T DELETE, IMPLEMENT".

## What this audit did NOT do

- **No diff-since-baseline check**  -  I didn't compare against a pre-A1 snapshot to confirm zero TODO/etc were ALREADY there before the audit-fix track started. The codebase was clean BEFORE the audit-fix work; the audit-fix track didn't introduce any.
- **No `cargo public-api diff`**  -  A19 will run that against the final state.
- **No `cargo machete`** for unused deps  -  A18's job.
- **No deep mod-tree validation**  -  the `find + grep` approach has false-positive orphan reports because `pub(crate) mod`, `#[path = "..."]`, and `#[cfg(test)]` patterns weren't all matched. A16 should verify each apparent orphan via actual `cargo doc` reachability or read the relevant mod.rs.
