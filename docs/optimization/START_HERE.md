# Start Here for Optimization Work

Applies to Vyre 0.7.2.

Read [`README.md`](README.md) before you change an optimizer or backend. It
defines the two-layer boundary and the proof contract.

## Implementation sequence

1. Find the owning boundary in `OWNERSHIP.toml`.
2. Identify the optimization class in `TAXONOMY.md`.
3. Check the generated pass reference in `PASSES.md`.
4. Keep semantic IR rewrites in `vyre-foundation/src/optimizer/`.
5. Keep target-specific lowering strategy in the owning driver crate.
6. Update `OP_MATRIX.toml` when operation or backend support changes.
7. Update `BENCH_TARGETS.toml` when a target or baseline class changes.
8. Prove semantic equivalence and exercise the real optimized path.
9. Run the required commands for the owning boundary.

## Placement guide

| Question | Owner |
|---|---|
| Does the rewrite preserve IR semantics for every backend? | `vyre-foundation/src/optimizer/` |
| Does it emit a target-specific instruction or API call? | The owning concrete driver crate |
| Is it neutral launch, binding, validation, cache, or residency policy? | `vyre-driver/src/` |
| Is it persistent queue, scheduler, or I/O behavior? | `vyre-runtime/src/megakernel/` |
| Is it benchmark measurement or reporting? | `vyre-bench/` and `BENCH_TARGETS.toml` |
| Is it operation support or parity status? | `OP_MATRIX.toml` |

Historical plans and reports do not override this control plane.
