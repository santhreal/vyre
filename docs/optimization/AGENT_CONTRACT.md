# Optimization agent contract

This contract applies to every native worker, human contributor, or automated
agent doing performance work in Vyre.

## Required before editing

1. Read `docs/optimization/README.md`.
2. Pick one lane from `docs/optimization/OWNERSHIP.toml`.
3. Confirm the task is Layer 1 or Layer 2.
4. Identify the `cargo_full` checks, tests, benchmark evidence, and release evidence artifacts the patch will produce.

Do not start from grep markers such as TODO/FIXME/deferred. Those markers are
not the optimization backlog. Start from hot paths, algorithmic complexity,
data movement, emitted code quality, launch overhead, cache behavior, and
benchmark targets.

## Patch acceptance

A patch is acceptable only when it changes behavior, structure, or measurable
performance. At least one must be true:

- Fewer IR nodes, memory ops, branches, launches, allocations, copies, locks, or
  cache misses for the same workload.
- Better asymptotic complexity for a pass, runtime queue, serializer, or backend path.
- Better backend codegen for the same IR.
- Stronger conformance/parity proof for an optimized path.
- Better benchmark truth, including active-time measurement and SOTA baselines.

Docs can accompany the patch. Docs cannot be the patch unless the explicit task
is documentation architecture.

## Required evidence in the final worker report

Every worker report must list:

- Lane name.
- Files changed.
- Layer placement.
- Behavior/performance change.
- Tests run, with pass/fail result.
- Benchmark or shape evidence, if applicable.
- Matrix/target rows updated, if applicable.
- Release evidence anchors updated, if applicable.
- Any touched code outside the lane and why.

## Prohibited work

- Audit-only output when implementation work is available.
- Removing a marker without implementing the behavior it described.
- Moving backend-specific logic into shared crates.
- Copying an optimizer rewrite into a driver.
- Weakening a test to match current behavior.
- Adding a compatibility shim or re-export instead of moving the implementation
  to the right layer.
- Claiming support for an op/backend pair without tests and an `OP_MATRIX.toml` row.
- Adding a new benchmark with a naive baseline when a SOTA baseline class exists.
- Using raw `cargo` where `cargo_full` is required by the release lane.
- **Rolling back a half-migration** (ROADMAP S20). Migration findings flagged in
  the `S` series and the half-migration ledger (`HM*`) are *forward-only*.
  When a migration is incomplete, finish it; do not restore the legacy path.
  Rollback is only allowed when (a) the destination architecture is proven
  obsolete and (b) the user explicitly approves deletion via Telegram.

## Escalation

Stop and report to the main integrator when:

- Two lanes need the same file.
- A public API break is required.
- A backend limitation appears to require changing the IR contract.
- Correctness requires choosing between deterministic reference semantics and
  hardware-native undefined behavior.

## Release evidence anchors

- `release/evidence/version/version-matrix.json`
- `release/evidence/backends/backend-matrix.json`
- `release/evidence/metadata/feature-matrix.json`
- `release/evidence/metadata/metadata-matrix.json`
- `release/evidence/benchmarks/release-workload-matrix.json`
- `release/evidence/conformance/conformance-matrix.json`
- `release/evidence/tests/test-matrix.json`
- `release/evidence/final/completion-audit.json`
