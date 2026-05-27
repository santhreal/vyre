# vyre code organization conventions

Established by audit cleanup A17 (2026-04-30). These rules are the
long-term guardrails that prevent the entropy A1–A16 found from coming
back. Torvalds-style: small focused files, one concept per file, clear
directory taxonomy, no horizontal sprawl at the crate root, the mod
tree is the table of contents.

## 1. Crate boundaries

Each workspace member has one purpose. Crossing the boundary is a code
review issue.

| Crate | Purpose | Forbidden |
|---|---|---|
| `vyre-foundation` | IR + optimizer + lower (substrate-only) | language frontends, application code, demos, backend-specific emit |
| `vyre-primitives` | Tier 2.5 LEGO substrate (feature-gated per domain) | Pass-trait wrappers (those go in vyre-foundation/optimizer/passes/), backend-specific code |
| `vyre-self-substrate` | vyre using its own primitives on its own scheduler / dataflow / cost-model problems (recursion thesis layer) | non-substrate-self-uses, backend-specific code, frontend code |
| `vyre-driver` | backend abstraction (Pass trait dispatch, registry, capability negotiation) | backend-specific emit code, substrate self-uses |
| `vyre-driver-<backend>` | backend-specific dispatch + final emit | substrate-side IR transformations |
| `vyre-runtime` | host-side dispatch, scheduling, megakernel orchestration | substrate-side IR transformations, language frontends |
| `vyre-libs` | shared libs published independently to crates.io (`secfinding`, `multimatch`, `attackstr`, etc.) | substrate code, language frontend code |
| `vyre-c-frontend` (post-A14) | C language frontend (lex / preprocess / parse / sema / lower) | non-C-language code |

**Forbidden in substrate crates** (vyre-foundation, vyre-primitives,
vyre-self-substrate): language frontends, application code, demos,
backend-specific emit code.

## 2. File-size cap

Source code: **500 LOC**. Files over the cap split via the parent-as-dir
pattern: `foo.rs` (500 LOC) becomes `foo/{mod, concern_a, concern_b,
concern_c}.rs` with `mod.rs` declaring submodules + re-exports.

Test files: **1000 LOC**, then split per fixture group.

## 3. Directory taxonomy

Every `src/` follows:

```
src/
├── lib.rs              ← mod tree only; ≤8 mod lines at top level
├── error.rs            ← top-level error type if any
├── test_util.rs        ← top-level test helpers
└── <concept_dir>/
    ├── mod.rs          ← public API + sub-mod tree
    └── <feature>.rs
```

No loose `.rs` at crate root other than `lib.rs`, `main.rs`, `error.rs`,
`test_util.rs`. New work that doesn't fit an existing concept_dir gets
a new concept_dir, not a new loose root file.

## 4. No duplicate concepts across crates

One canonical home per concept (CSE engine, DCE engine, scheduler,
lower, etc.). When a concept could plausibly live in two places, pick
one and document the choice in this file. The other home is a
back-compat re-export only, marked with the audit tag that established
the canonical location.

Established canonical homes (post-A1-A16):

| Concept | Canonical home |
|---|---|
| Pass-scheduler | `vyre-foundation::optimizer::scheduler` |
| Megakernel-fusion-scheduler | `vyre-foundation::optimizer::megakernel::{matroid_subset, schedule_oracle}` |
| Dispatch-scheduler | `vyre-runtime::scheduler` |
| Megakernel runtime orchestrator | `vyre-runtime::megakernel::scheduler` |
| CSE engine + Pass wrapper | `vyre-foundation::optimizer::passes::fusion_cse::cse::{engine, wrapper}` |
| DCE engine + Pass wrapper | `vyre-foundation::optimizer::passes::fusion_cse::dce::{engine, wrapper}` |
| Substrate-side lowering | `vyre-foundation::lower` |
| Backend final emit | `vyre-driver-<backend>::emit` (cuda uses `codegen` for nvcc/PTX-tooling familiarity) |
| Backend trait boundary | `vyre-driver::backend::lowering` |
| Self-substrate primitives | `vyre-self-substrate::*` |
| C language frontend | `vyre-libs::parsing::c::*` (re-export shim), `vyre-c-frontend::*` (canonical, post-A14 prerequisite) |

## 5. Auto-discovery over hand-maintained registries

New passes / dialects / law registrations / extension hooks use
`inventory::submit!`  -  never hand-maintained `use foo::{...}` import
blocks. Adding a new pass should require ZERO edits to the parent
crate's `lib.rs` or `optimizer.rs`.

(A4 collapsed the previous 19-typed-variant `PassKind` enum into a
newtype `pub struct PassKind(Box<dyn Pass>);` and replaced the
hand-maintained `registered_passes()` body with a 4-line inventory
iter  -  this is the pattern.)

## 6. Naming convention

- `lower/`  -  substrate IR → backend-IR transformations.
- `emit/`  -  backend-IR → final source/binary output.
- `codegen/`  -  CUDA convention equivalent to `emit/` (kept for
  nvcc/PTX-tooling familiarity).
- `lowering.rs` (singular file at `vyre-driver/src/backend/`)  -  the
  cross-backend `LowerableOp` trait boundary.
- `transform/`  -  deprecated; merging into `optimizer/` over time.
- `pass_substrate/`  -  small shrinking module; megakernel scheduler
  pieces moved to `optimizer/megakernel/` in A9; remaining items are
  substrate-self-uses awaiting the A10/A14 hoist.

## 7. No deferral lexicon

Per AGENTS.md no-evasion policy, no file or comment uses "deferred" /
"out of scope" / "post-launch" / "v0.7" / "later" as a way to push work
off. Open or closed, nothing else. "Open and unfinished" is fine  - 
"deferred" is not.

## 8. Audit + planning files

Three canonical homes:

- `docs/`  -  consumer-facing documentation (architecture, vision,
  migration guides).
- `audits/`  -  historical record of audits we ran. Audit findings live
  here; once an audit closes, the findings file moves to
  `audits/closed/<year>/`.
- `.internals/`  -  agent-coordination + active plans + scratch +
  release-engineering. Anything in `.internals/` is NOT shipped and is
  out of scope for downstream consumers.

Anything at the crate root is README + LICENSE-* + CHANGELOG +
CONTRIBUTING + CODE_OF_CONDUCT + SECURITY (the conventional set)
plus AGENTS.md with GEMINI.md / CLAUDE.md redirect stubs (AI-tooling root conventions  - 
agents.md is industry standard).

## 9. Test layout

- Inline `#[cfg(test)] mod tests { ... }` blocks are forbidden in new
  source files. The project convention is sibling `_tests.rs` files
  declared via `#[path = "x_tests.rs"] mod tests;`.
- The pre-existing baseline of inline-tests violations is enforced via
  `vyre-foundation/tests/organization_contracts.rs::foundation_inline_test_modules_are_baselined`.
  Adding to the baseline requires explicit user approval.
- Wildcard `pub use ...::*;` is forbidden anywhere in src/. Use named
  re-exports. Enforced by
  `vyre-foundation/tests/organization_contracts.rs::*_wildcard_pub_reexports_are_baselined`.

## 10. CI enforcement (A17 follow-up)

These checks run on every PR and block merge if a violation is added:

- `vyre-foundation/tests/organization_contracts.rs` (existing)  - 
  `no_root_stray_plan_docs`, wildcard re-exports baselined, inline test
  modules baselined.
- `vyre-foundation/tests/archive_confusion.rs` (existing)  -  only
  baselined `.internals/archive/` subdirs allowed.

To add (open follow-up  -  not deferred):

- `scripts/check_file_size_cap.sh`  -  fail PR if any new/modified file
  exceeds 500 LOC (test files exempt up to 1000 LOC).
- `scripts/check_root_scatter.sh`  -  fail PR if a new `.rs` is added to
  any crate root that isn't lib/main/error/test_util.
- `scripts/check_mod_tree_breadth.sh`  -  fail PR if `lib.rs` mod tree
  breadth grows beyond 8 top-level entries.
- `scripts/check_no_deferral_lexicon.sh`  -  grep PR diff for the
  "deferred"/"post-launch"/"out of scope"/etc lexicon and reject.
- `scripts/check_no_duplicate_concepts.sh`  -  fail if a new file matches
  the name of a file in a different crate (modulo a documented
  allowlist).

## 11. The audit-fix track (A1-A19)

Audit cleanup A1-A19 (2026-04-30) executed against an audit findings
list discovered in conversation. The persistent results:

- A1: 31GB freed by deleting `target/` + `target-fusion-fix/`.
- A2: 88 audit files in one canonical home + 7 doc files moved to
  `docs/` + 13 durable design docs hoisted from `.internals/planning/`
  to `docs/` + 25 dated execution traces archived.
- A3: 25-file flat passes/ → 8 category subdirs (algebraic, loops,
  memory, sync, fusion_cse, cleanup, specialization, lowering/{wgpu,
  cuda, cpu}).
- A4: PassKind enum (19 typed variants + 6 method match arms) collapsed
  to newtype + `inventory::submit!` autodiscovery.
- A5: Node walker hoisted to `visit/node_map.rs`; 5 cleanup passes
  refactored from per-pass walkers to map_children + map_body composition.
- A6+A7: CSE + DCE engines colocated with their Pass-trait wrappers.
- A8: vyre-primitives parsing CSE renamed `ast_*` to disambiguate from
  IR-level CSE.
- A9: scheduler concepts deduplicated into 3 canonical homes.
- A10: vyre-driver/self_substrate (55 files, 8873 LOC) extracted to
  dedicated `vyre-self-substrate` crate.
- A11: `lower/` vs `emit/` naming convention established + applied.
- A12: vyre-foundation/src/ root scatter (22 .rs files) → 4 grouped
  subdirs (runtime/, dispatch/, algebra/, analysis/), root down to 9.
- A13: const_fold/tests.rs (1340 LOC, 108 tests) split into 5 files;
  scheduler.rs, naga_emit/expr.rs, naga_emit/mod.rs, codegen/ctx.rs,
  megakernel/planner.rs, pipeline_cache.rs left as known-open splits.
- A14: C parser extraction attempted, blocked on substrate-dep cycle
  (parsing/core, region, harness need to hoist first); reverted cleanly,
  prerequisite documented.
- A15: orphan-symbol audit; vyre source has zero TODO/FIXME/
  unimplemented!()/todo!()/#[ignore] markers across all 1736 .rs files.
- A16: 3 of 4 pre-existing org_contracts test failures resolved; 1 open
  (bench_corpus_duplication missing python script  -  escalated to user).
- A17: this file.
- A18: cargo-machete unused-deps audit (next).
- A19: workspace-wide build + test + pass_invariants verification (last).

Open work documented in each task description; not deferred, just not
done yet.
