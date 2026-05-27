# PHASE 7 GRAPH PRIMITIVES AUDIT  -  Closure Status

Date: 2026-04-29

Scope:
- `vyre-primitives/src/graph/*.rs`
- graph consumers in `vyre-libs/src/dataflow` and `vyre-libs/src/security`

## Current Findings

| Status | Claim | Source proof |
|---|---|---|
| fixed | `reachable` was CPU-only. | `reachable::reachable_program` now emits a fused Tier-2.5 Program that seeds the source bitset, repeats CSR traversal, and accumulates the reached set. Unit tests assert emitted buffers and non-empty IR. |
| fixed | `toposort` was CPU-only. | `toposort::toposort_program` now emits a lane-0 Kahn Program with explicit scratch buffers and `[1, 1, 1]` workgroup size. Unit tests assert buffer layout and workgroup shape. |
| fixed | `exploded` used debug-only bounds and unchecked dense-id products. | `exploded::build_cpu_reference` uses checked arithmetic for `blocks_per_proc * facts_per_proc` and `num_procs * slots_per_proc`; invalid dimensions fail loudly. |
| fixed | `adaptive_traverse` described a density chooser without source backing. | `adaptive_traverse` now exposes `should_use_dense`, dense traversal CPU reference, and a checked dense-step Program that rejects shape overflow. |
| fixed | Persistent BFS was absent. | `persistent_bfs` and `persistent_bfs_step` now exist as graph primitives with harness entries and tests. |
| fixed | Graph source comments carried open-work labels. | The graph module and affected primitive headers now describe the current concrete contract without open-work labels. |

## Verification Commands

Run these after graph changes:

```bash
CARGO_BUILD_JOBS=1 cargo test -p vyre-primitives graph
CARGO_BUILD_JOBS=1 cargo test -p vyre-libs dataflow
CARGO_BUILD_JOBS=1 cargo test -p vyre-libs security
```

## Residual Risk

Some graph algorithms are intentionally serial within a single GPU
invocation because their contracts have loop-carried dependencies
(`toposort`, path reconstruction). Those are concrete implementations,
not missing builders; throughput-sensitive callers compose them with
partitioning or batching.
