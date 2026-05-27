# PHASE5 AST WALK + SECURITY PREDICATES  -  Scoped Closure Status

Date: 2026-04-29

This file now tracks only the graph/dataflow/security primitive rows
owned by the current vyre closure pass. Surge compiler registry work,
predicate-map performance, and AST frontend work are outside this pass.

## Current Findings

| Status | Claim | Source proof |
|---|---|---|
| fixed | `sanitized_by` did not subtract sanitizer nodes inside the emitted Program. | `vyre-libs/src/security/sanitized_by.rs` now accepts `sanitizers_in`, fuses `bitset_and_not(frontier_in, sanitizers_in)` before `csr_forward_traverse`, and registers a non-vacuous witness. |
| fixed | `flows_to` and `taint_flow` used an all-edge mask. | `vyre-libs/src/security/flows_to.rs` defines `FLOWS_TO_MASK`; `taint_flow` imports the same mask, restricting traversal to dataflow edge kinds. |
| fixed | Dataflow primitive rows described missing builder bodies. | `vyre-libs/src/dataflow/{ssa,callgraph,slice,range,escape,summary,loop_sum,points_to}.rs` now emit concrete builders or CPU helper surfaces with tests and soundness markers. |
| fixed | IFDS graph traversal was disconnected from the exploded-supergraph implementation. | `ifds::ifds_reach_step_exploded` delegates to `ifds_gpu::ifds_gpu_step`; `ifds_gpu::solve_cpu` uses a queue traversal and sorted encoded output. |
| fixed | Exploded-supergraph CPU reference had avoidable high-order scans. | `vyre-primitives/src/graph/exploded.rs` validates dimensions with checked arithmetic and keeps malformed graph shapes loud. |

## Verification Commands

```bash
CARGO_BUILD_JOBS=1 cargo test -p vyre-libs security
CARGO_BUILD_JOBS=1 cargo test -p vyre-libs dataflow
CARGO_BUILD_JOBS=1 cargo test -p vyre-primitives graph::exploded
```

## Remaining Non-Owned Rows

The earlier broad AST-walk review also covered surgec registry lookup,
predicate lowering, range ordering, and VAST/frontend integration. Those
are not graph/dataflow/security primitive source rows in `vyre`, so this
pass does not claim ownership of them.
