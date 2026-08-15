# Legacy optimization and performance documents

This file explains how historical optimization evidence relates to the current
control plane.

## Canonical files

| Topic | Canonical file |
|---|---|
| Active work | private root `BACKLOG.md` |
| Start point | `docs/optimization/START_HERE.md` |
| Layer and ownership rules | `docs/optimization/README.md` |
| Optimization classes | `docs/optimization/TAXONOMY.md` |
| Worker lanes | `docs/optimization/OWNERSHIP.toml` |
| Patch proof contract | `docs/optimization/README.md` |
| Op and backend status | `docs/optimization/OP_MATRIX.toml` |
| Benchmark targets | `docs/optimization/BENCH_TARGETS.toml` |

## Evidence-only files

| File | Use it for |
|---|---|
| `audits/VYRE_OPTIMIZER.md` | Historical optimizer findings. |
| Prior driver-consolidation evidence | Private historical audit record. |
| `.internals/**` | Private evidence and maintainer notes. |

Old plan, roadmap, backlog, handoff, brief, and status files were imported into
the root backlog and deleted. Do not recreate them as evidence archives.

## How to migrate an old finding

1. Identify the lane in `OWNERSHIP.toml`.
2. Verify the finding against the current tree.
3. Add a dedicated four-column row to the root `BACKLOG.md`.
4. Update `OP_MATRIX.toml` when an op or backend contract changes.
5. Update `BENCH_TARGETS.toml` when a performance target changes.
6. Delete any parallel planning document after preserving useful evidence in
   the backlog.
