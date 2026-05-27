# Legacy optimization and performance documents

This file explains where the scattered plans and audits fit after the
optimization control plane was canonicalized.

## Canonical files

| Topic | Canonical file |
|---|---|
| Start point | `docs/optimization/START_HERE.md` |
| Layer and ownership rules | `docs/optimization/README.md` |
| Executable roadmap | `docs/optimization/ROADMAP.md` |
| Active claims | `docs/optimization/CLAIMS.toml` |
| Optimization classes | `docs/optimization/TAXONOMY.md` |
| Worker lanes | `docs/optimization/OWNERSHIP.toml` |
| Patch proof contract | `docs/optimization/AGENT_CONTRACT.md` |
| Op/backend status | `docs/optimization/OP_MATRIX.toml` |
| Benchmark targets | `docs/optimization/BENCH_TARGETS.toml` |

## Superseded or evidence-only files

| File | Status | Use it for |
|---|---|---|
| `audits/RELEASE_1000X_PLAN.md` | evidence-only | Historical consumer/Vyre performance ideas. Convert actionable Vyre substrate work into ownership-lane tasks. |
| `PERF_ROADMAP_2026-05-01.md` | evidence-only | Imported into `ROADMAP.md`; use original IDs only as references. |
| `CC_OWNED_BACKLOG_2026-05-01.md` | evidence-only | Pre-control-plane CC sweep index. Active claims now live in `CLAIMS.toml`; lane assignment is in `OWNERSHIP.toml`. Use the file only for historical context on the five structural seeds (SEED-1..6). |
| `audits/VYRE_OPTIMIZER.md` | evidence-only | Historical optimizer findings. Active optimizer work must use `TAXONOMY.md` and `OWNERSHIP.toml`. |
| `audits/VYRE_PERFORMANCE_ARCHITECTURE_INVENTORY_2026-04-28.md` | evidence-only | Raw performance inventory. Convert rows into `OP_MATRIX.toml` or `BENCH_TARGETS.toml`. |
| `docs/DRIVER_UNIFICATION_AUDIT.md` | evidence-only | Prior driver-consolidation audit. Shared-vs-backend placement now follows `README.md` and `OWNERSHIP.toml`. |
| `docs/CUDA_BACKEND_EXECUTION_PLAN.md` | evidence-only | CUDA roadmap context. Active CUDA work uses `driver_cuda` lane. |
| `vyre-bench/PLAN.md` | evidence-only | Historical bench worklist. Targets now live in `BENCH_TARGETS.toml`. |
| `vyre-bench/RELEASE_BRIEF.md` | evidence-only | Historical bench brief. Implementation must follow bench lane and target file. |
| `.internals/**` | maintainer notes | Evidence only unless linked from a canonical file. |

## How to migrate an old finding

1. Identify the lane in `OWNERSHIP.toml`.
2. Identify the optimization class in `TAXONOMY.md`.
3. Add or update the `OP_MATRIX.toml` row if an op/backend pair changes.
4. Add or update the `BENCH_TARGETS.toml` row if performance target changes.
5. Implement code and tests in the owning lane.
6. Leave the old document intact, but add a supersession header if it is a
   likely entry point for future agents.
