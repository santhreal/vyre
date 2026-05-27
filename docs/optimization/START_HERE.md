# Start here for optimization work

Use this file before opening any old plan, audit, or performance note.

## If you are assigning workers

1. Choose lanes from `OWNERSHIP.toml`.
2. Give each worker one lane and the matching `required_commands`.
3. Require the report format in `AGENT_CONTRACT.md`.
4. Do not assign audit-only work when implementation work is available.
5. Keep cross-lane architecture decisions in the main integrator.

## If you are implementing

1. Read `README.md`.
2. Read `ROADMAP.md`.
3. Pick a lane in `OWNERSHIP.toml`.
4. Add or update a claim in `CLAIMS.toml` for broad work.
5. Identify the optimization class in `TAXONOMY.md`.
6. Update `OP_MATRIX.toml` if op/backend coverage changes.
7. Update `BENCH_TARGETS.toml` if benchmark targets or baselines change.
8. Add tests that prove the optimized path.
9. Run the lane commands from `OWNERSHIP.toml`.

## If you found an old plan

Treat it as evidence. Check `LEGACY_DOCS.md` for its status. If it conflicts
with this directory, this directory wins.

## If you found a half-migration

Migration findings flagged in `ROADMAP.md` `S` rows and the half-migration
ledger (`HM*`) are *forward-only*. When a migration is incomplete, finish it
in the destination architecture; do not restore the legacy path. Rollback
needs explicit user approval via Telegram and proof the destination is
obsolete (ROADMAP S20).

## Live control plane

The only live optimization control plane is this directory
(`docs/optimization/`). Every other docs/audit file is evidence-only  -  see
`LEGACY_DOCS.md` for the supersession map. New control-plane content goes
here; never in `docs/`, `audits/`, `vyre-bench/PLAN.md`, or `.internals/`.

## Fast placement guide

| Question | Answer |
|---|---|
| Does the rewrite preserve IR semantics for every backend? | `vyre-foundation/src/optimizer/` |
| Does it emit a hardware-specific instruction or API call? | owning `vyre-driver-*` crate |
| Is it backend-neutral launch, binding, validation, cache, or residency policy? | `vyre-driver/src/` |
| Is it persistent queue/scheduler/IO behavior? | `vyre-runtime/src/megakernel/` |
| Is it benchmark measurement, baselines, or reporting? | `vyre-bench/` and `BENCH_TARGETS.toml` |
| Is it op support or parity status? | `OP_MATRIX.toml` |
| Is it an old roadmap/audit/perf finding? | Convert it into `ROADMAP.md`, then implement in the owning lane. |
