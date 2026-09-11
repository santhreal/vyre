# vyre code organization conventions

**Status: Superseded.** Use [`docs/code-style.md`](code-style.md), the generated
crate guides, and `scripts/quality_gate.sh` for current contributor rules.

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
| Megakernel IR fusion oracles | `vyre-foundation::optimizer::megakernel::{matroid_subset, schedule_oracle, scratch_reuse}` |
| Whole-graph artifact compiler | `vyre-megakernel::{compile, Artifact, ArtifactEnvelope}` |
| Megakernel wave policy | `vyre-driver::{megakernel_execution, megakernel_barrier, megakernel_frontier}` |
| Dispatch-scheduler | `vyre-runtime::scheduler` |
| Megakernel runtime protocol/orchestrator | `vyre-runtime::megakernel::{protocol, scheduler, planner, resident}` |
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
## 7. Repository layout

Crate roots contain package metadata, licenses, and contributor-facing
documentation. Implementation, tests, fixtures, and generated evidence live
under their owning subsystem directories.

## 8. Test layout

- New behavioral tests live in crate `tests/` directories.
- Inline test modules are reserved for contracts that require private items.
- Public surfaces use named re-exports.

## 9. CI enforcement

- `scripts/check_max_file_size.sh` enforces the source-size budget.
- `scripts/check_repo_hygiene.sh` rejects repository hygiene regressions.
