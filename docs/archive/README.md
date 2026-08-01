# Archived documents

Nothing in this directory describes how vyre works today. Do not follow the
instructions in these files, and do not treat their claims as current.

These are finished plans, completed audits, and point-in-time status reports.
They are kept because they record why decisions were made, not what the code
does now. A file here was accurate when it was written and has not been updated
since.

For current documentation, start at [`../INDEX.md`](../INDEX.md). For the
architecture as built, read [`../ARCHITECTURE.md`](../ARCHITECTURE.md). The
day-to-day state of the release gates is tracked in a maintainer status file that
is excluded from the repository and does not ship, so there is no published
document to send you to for it; read
[`../../audits/RELEASE_GATE.md`](../../audits/RELEASE_GATE.md), which is the
published release gate and execution backlog.

## What is here

Two kinds of file live in this directory, and the difference matters to you.

PUBLISHED archived documents, which you can open:

- [`MIGRATION_0.6_TO_0.7.md`](MIGRATION_0.6_TO_0.7.md), a migration guide you may
  still need if you are upgrading across the version it names.
- [`HEURISTIC_TO_MATH_TRACKER.md`](HEURISTIC_TO_MATH_TRACKER.md)
- [`INNOVATION_SWEEP.md`](INNOVATION_SWEEP.md)
- [`JULES_PRIMITIVE_MANIFEST.md`](JULES_PRIMITIVE_MANIFEST.md)
- [`MICRO_FLAW_LOG.md`](MICRO_FLAW_LOG.md)
- [`NAGA_CRITICAL_HOLES.md`](NAGA_CRITICAL_HOLES.md)
- [`UX_SWEEP.md`](UX_SWEEP.md)
- [`ROADMAP_APPEND_ONLY_2026-05-22.md`](ROADMAP_APPEND_ONLY_2026-05-22.md)
- [`vision-2026-04-27-essay.md`](vision-2026-04-27-essay.md)

RETAINED BUT NOT PUBLISHED. The documents below exist in the maintainer's working
copy and are excluded from the repository, because `.gitignore` keeps files named
`*PLAN*`, `*STATUS*`, `*AUDIT*`, `*ROADMAP*` and `*BACKLOG*` out of public
history. They are named here rather than linked, so that the record of what was
decided is not hidden from you, and so that no link on this page dies the moment
you clone. If you need one, ask a maintainer; do not expect to find it in the
tree.

Completed execution plans: `COMPILER_E2E_PLAN.md`,
`COMPILER_PRODUCT_BOUNDARY_PLAN.md`, `CRATE_EXTRACTION_PLAN.md`,
`CUDA_BACKEND_EXECUTION_PLAN.md`, `OP_MASTER_PLAN_BUILDING_BLOCKS_AND_QA.md`,
`V7_AGENT_A_PLAN.md`, `V7_RELEASE_PLAN.md`, `VYRE_RELEASE_COMPLETION_PLAN.md`,
`VYRE_WEIR_RELEASE_RELEASE_PLAN_2026-05-05.md`.

Point-in-time audits: `NAGA_LOWERING_AUDIT.md`, `NAGA_LOWERING_STATUS.md`,
`VYRE_CODEBASE_AUDIT.md`.

Historical status snapshots: `POST_LANDING_STATUS.md`.
