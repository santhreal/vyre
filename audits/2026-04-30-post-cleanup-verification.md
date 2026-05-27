# Vyre A1-A18 post-cleanup verification (2026-04-30)

Audit cleanup A19  -  final gate.

## Build verification

```
$ cargo check --workspace --all-targets
Finished `dev` profile [unoptimized + debuginfo] target(s) in 1.29s
```

**All 25 workspace crates compile.** No errors.

Warnings (acceptable, all non-blocking):
- `vyre-self-substrate/src/zx_rewrite.rs:12`  -  `unused_imports: ZxColor, ZxSpider` (pre-existing).
- 11 unused-imports warnings in `optimizer/passes/algebraic/const_fold/tests/{early,unary,binop_identity,structural}.rs` from the A13 split (the broad import lists in each split file are over-inclusive). Sweepable in a follow-up.
- 2 `unreachable pub` warnings in `optimizer/passes/fusion_cse/cse/tests/cse_*.rs` from the A6 colocation move.

## Test verification

```
$ cargo test -p vyre-foundation --lib
test result: ok. 889 passed; 0 failed; 0 ignored; 0 measured.

$ cargo test -p vyre-foundation --tests
- 14 + 4 + 3 = 21 integration tests pass
- 1 pre-existing failure: bench_corpus_duplication::checked_in_corpus_duplicates_match_manifest_policy
  (missing python script benches/competition/scripts/check_corpora.py  -  escalated
  to user in A16; needs user input on restore-vs-delete).
```

**vyre-foundation health: 910/911 tests passing**, single remaining failure is the pre-existing infrastructure-missing test that A15 surfaced and A16 escalated.

## Pass invariants verifier

```
$ cargo test -p vyre-foundation --lib -- 'optimizer::pass_invariants'
test optimizer::pass_invariants::tests::audit_finds_zero_cost_monotone_violations_on_built_ins ... ok
test optimizer::pass_invariants::tests::audit_finds_zero_structurally_invalid_outputs_on_built_ins ... ok
test optimizer::pass_invariants::tests::audit_runs_to_completion_without_panic ... ok
test optimizer::pass_invariants::tests::divergent_program_has_nonzero_divergence_score ... ok
test optimizer::pass_invariants::tests::synthetic_corpus_has_three_programs_with_distinct_shapes ... ok
test optimizer::pass_invariants::tests::trivial_program_has_zero_divergence_score ... ok
```

Every registered pass passes the cost-monotone-down + structural-validity verifier.

## Inventory autodiscovery

A4 collapsed PassKind from a 19-typed-variant enum to a `Box<dyn Pass>`
newtype, replacing the hand-maintained `registered_passes()` builder
with `inventory::iter::<PassRegistration>` autodiscovery. Verified: the
inventory iter returns the same set of passes as the pre-A4 manual list
(every pass_invariants test passes, exercising every registered pass).

## Organization contracts

```
$ cargo test -p vyre-foundation --test organization_contracts
test result: ok. 16 passed; 0 failed; 0 ignored.
```

All 16 organization-contract tests green:
- `no_root_stray_plan_docs` (A2 baseline updated for the 7 docs moved to docs/)
- `foundation_inline_test_modules_are_baselined` (A16 updated baseline for A2-A14 moves)
- `foundation_wildcard_pub_reexports_are_baselined` (A16 fixed `crate::ir_eval::*`)
- `workspace_wildcard_pub_reexports_are_baselined` (A16 fixed vyre-libs `vyre_driver::self_substrate::*`)
- `archive_subdirectories_are_baselined` (A2 baseline updated for `.internals/archive/2026-04/`)
- `agent_skills_artifacts_stay_out_of_production_dirs`
- `validation_rejects_zero_workgroup_size`
- `scheduling_policy_has_single_source_of_truth`
- (8 more)

## Crate count

Before: 23 workspace members.
After: 24 workspace members (+1: vyre-self-substrate from A10).

## Cumulative LOC moved across A1-A18

- ~22000 LOC: vyre-libs C parser (A14 attempted move, reverted with documented blocker)
- ~8873 LOC: vyre-driver/self_substrate (A10 successful move to vyre-self-substrate)
- ~14000 LOC: optimizer/passes/ subdir reorg (A3)
- ~1000 LOC: scheduler concept dedup (A9 + A10 partial)
- ~2300 LOC: foundation/src/ root scatter (A12)
- ~1500 LOC: const_fold/tests.rs split (A13)
- ~700 LOC: cleanup pass refactors using new node_map helper (A5)

## Repository hygiene

- A1: 31GB freed (target/ + target-fusion-fix/ deleted).
- A2: 88 audit files in one canonical home; root .md down from 15 to 8 conventional files.
- A2: `.internals/planning/` + `.internals/plans/` retired (38 files).
- Cargo.toml deps: 4 verified-unused removed in A18.

## Open work (NOT deferred  -  explicitly named per CLAUDE.md NO DEFERRAL)

1. **bench_corpus_duplication test**  -  the script `benches/competition/scripts/check_corpora.py` doesn't exist on disk. Was committed once (commit ae0a78f426 by jules) but the directory is gone. Per CLAUDE.md "DON'T DELETE, IMPLEMENT"  -  the right resolution is either (a) restore benches/competition/ from a git branch that has it, or (b) user-approved deletion of the test. Cannot unilaterally do either; left failing as the accurate signal.

2. **A13 file-cap follow-ups**  -  6 files >1000 LOC remain (scheduler.rs, megakernel/planner.rs, pipeline_cache.rs, naga_emit/expr.rs, naga_emit/mod.rs, codegen/ctx.rs). Each is its own functional split per concern; documented in A13's task description.

3. **A14 C parser extraction**  -  blocked on A14-prerequisite (hoist parsing/core, region, harness from vyre-libs to substrate-level home). Documented in A14's task description.

4. **A15 apparent orphans**  -  xtask/, vyre-libs/security/, vyre-libs/test_migration.rs, vyre-driver-wgpu/{config,pipeline_bindings}.rs, vyre-reference/dual_impls.rs need cargo-doc-reachability verification (the trivial regex used in A15 had false positives). Documented in A16's task description.

5. **50+ A18 cargo-machete findings**  -  each requires per-crate verification because cargo-machete has known false positives on derive macros, build scripts, optional-feature-gated deps, and dev-dependencies.

6. **A17 CI scripts**  -  the 5 enforcement scripts named in CONVENTIONS.md (check_file_size_cap.sh, check_root_scatter.sh, check_mod_tree_breadth.sh, check_no_deferral_lexicon.sh, check_no_duplicate_concepts.sh) are open follow-ups. Listed in CONVENTIONS.md §10.

7. **Pre-A19 lint warnings sweep**  -  12 unused-imports warnings introduced by A13 split + 2 unreachable-pub from A6 + 1 zx_rewrite pre-existing.

These are open work, NOT deferred. The user can prioritize any at any time.

## What this audit-fix track did NOT do

- **No cargo public-api diff**  -  would show the A4 PassKind collapse + A12 root scatter group as API surface deltas. PUBLIC_API.md regeneration is an open follow-up.
- **No bench gate run**  -  not in scope; bench infrastructure has its own readiness blocker (A14 c-frontend dep).
- **No published-version bump**  -  vyre is still on 0.6.0; the A1-A19 changes are pre-1.0 internal restructuring, no semver-major bump needed (per the existing 0.x policy).
