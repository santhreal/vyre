# PHASE 6 DATAFLOW AUDIT  -  Closure Status

Date: 2026-04-29

Scope for this closure pass:
- `vyre-libs/src/dataflow/*.rs`
- `vyre-primitives/src/graph/exploded.rs`
- `vyre-primitives/src/graph/csr_forward_traverse.rs`
- `vyre-primitives/src/fixpoint/bitset_fixpoint.rs`
- `vyre-primitives/src/graph/program_graph.rs`

## Current Findings

| Status | Claim | Source proof |
|---|---|---|
| fixed | Seven dataflow builders were previously described as panic-only bodies. | `ssa`, `callgraph`, `slice`, `range`, `escape`, `summary`, and `loop_sum` now emit concrete `Program` builders, carry unit tests, and expose soundness markers. |
| fixed | `points_to` used a degenerate one-node shape and lacked harness coverage. | `points_to::andersen_points_to` now requires `ProgramGraphShape`, registers `OpEntry` and `ConvergenceContract`, and includes deterministic plus property tests for subset-closure monotonicity. |
| fixed | IFDS had two disconnected implementations. | `ifds::ifds_reach_step_exploded` delegates to `ifds_gpu::ifds_gpu_step`; the standard IFDS step uses a restricted dataflow mask rather than every edge kind. |
| fixed | IFDS CPU solver claimed queue semantics it did not use. | `ifds_gpu::solve_cpu` now uses `VecDeque` queue traversal and returns sorted encoded node ids. |
| fixed | Exploded-node encoding relied on debug-only checks and allowed product overflow. | `exploded::build_cpu_reference` uses checked products, and encoding paths surface invalid dimensions loudly. |
| fixed | `live` fixture was vacuous. | `live` now seeds a non-empty backward frontier and expects one-step propagation to grow it. |
| fixed | Dataflow registration was incomplete. | All ten named dataflow primitives now submit `OpEntry` coverage or a concrete public builder with local tests for CPU-only helper surfaces. |

## Verification Commands

Run these after touching dataflow:

```bash
CARGO_BUILD_JOBS=1 cargo test -p vyre-libs dataflow
CARGO_BUILD_JOBS=1 cargo test -p vyre-primitives graph::exploded
CARGO_BUILD_JOBS=1 cargo test -p vyre-conform-runner --features gpu --test lens_parity
```

## Residual Risk

The current dataflow surface is still a collection of small composable
steps rather than a monolithic analyzer. That is intentional for vyre's
Tier-3 contract: driver loops compose traversal, transfer, sanitizer, and
summary stages while `OpEntry` fixtures verify each step's byte behavior.
