# Testing Program  -  SQLite / NASA JPL / Linux / Chromium Standard

Closes #31 A.7 testing program direction.

## The bar

**Happy path is not the testing bar.** Unit tests prove oracle-backed behavior on representative and boundary inputs; adversarial, gap, property, fuzz, and conform work carry most of the safety budget. See fleet [`STANDARD.md`](../../../../../STANDARD.md) (Test Contract, Agent Work) and [`coordination/PUSH_TEMPLATE.md`](../../../../../coordination/PUSH_TEMPLATE.md).

vyre + consumer measure themselves against four testing programs:

- **SQLite**  -  590× more test code than source. 100% branch coverage.
  Billions of test cases via TH3. OOM injection. IO error injection.
  Fuzz via AFL. Every API called with every possible error condition.
- **NASA JPL**  -  every function has a contract (preconditions,
  postconditions, invariants). Tests verify the contract, not the
  implementation. Implementation is free to change so long as the
  contract holds.
- **Linux kernel**  -  kselftest + syzkaller + KASAN + KCSAN + lockdep.
  Every subsystem has its own suite. Concurrency bugs caught by
  systematic schedule exploration.
- **Chromium**  -  ClusterFuzz runs 24/7. Every commit fuzzed. Every
  crash a P0. Regressions detected within hours.

## The vyre/consumer surface today

Six kinds of test live side-by-side. Every module must carry all six
for the crate to ship. Per-module coverage lives in
`docs/testing/<crate>.md`; this doc is the umbrella contract.

| Kind | What it proves | Gate |
|---|---|---|
| **Unit** | Normal-case functional correctness. | Per-module `#[cfg(test)] mod tests`. |
| **Adversarial** | Hostile / malformed inputs produce actionable errors, never silent corruption or panic. | Per-module adversarial file (`tests/adversarial/*.rs`). |
| **Property** | Invariants hold for all inputs (proptest). | `proptest!` block per invariant. |
| **Benchmark** | Performance targets met (criterion). | `benches/*.rs` + gated thresholds per GATE_CLOSURE.md G4. |
| **Gap** | *What's missing* via `#[should_panic]` or intentionally-failing assertions. Failing gap tests are findings, not bugs in the test. | `tests/gap_*.rs`. |
| **Fuzz** | Structure-aware fuzz (swc, vyre wire format, SURGE grammar, HTTP request shapes). | `fuzz/` directory; runs in CI nightlies. |

## Multi-tier dispatch

Following the LAW 5 SQLite-grade rule, every subsystem tests are
written by **at least two agent tiers** because different agents find
different bugs:

- **Codex 5.4** for structural / multi-crate tests.
- **Kimi K2.5** for adversarial designed-to-FAIL tests.
- **Cursor-agent** for automated review of the first two.

A test suite that passes all three agent tiers is the minimum release
bar.

## Designed-to-FAIL vs proving tests

Every fix ships a pair. For NAGA_DEEPER F59 (U64 arithmetic):

- **Proving**  -  `f59_u64_bitand_still_lowers`: the *correct*
  component-wise op still succeeds (rejection is scoped).
- **Adversarial**  -  `f59_u64_add_rejects_with_named_carry_hint`:
  the *wrong* op is rejected, message names the fix.

The adversarial test is the one that would have caught the bug if
written first. Every audit finding closes with this pair
co-located.

## Fuzz + sanitiser roadmap

- **swc fuzz** on `jsir`  -  structure-aware JS AST corpus. Currently
  running seed corpus; ClusterFuzz-style continuous fuzz is the
  next sweep.
- **vyre wire-format fuzz**  -  arbitrary bytes → `from_wire` →
  `to_wire` round-trip vs validate. Landed.
- **SURGE grammar fuzz**  -  `consumer compile` on syntactically
  arbitrary inputs; must never panic, only return `consumer-ENN`
  errors. Landed.
- **HTTP request fuzz** on `pocgen`  -  structure-aware template
  substitution against a curl reference. Source-change required.
- **Sanitisers**  -  cargo-careful (MSan/ASan via `std` build) is the
  local run; `cargo careful run` in CI before every tag.

## Concurrency coverage

Every crate using interior mutability / atomics ships:

- **Lockdep-style invariant tests** (no lock reversal).
- **Loom tests** where the state machine is small enough (e.g.
  `ReadbackRing`, `PipelineCache`). Loom runs in the release
  gate.
- **Stress tests**  -  N-threaded flood at the public surface (see
  `scan_diagnostics_rate_limit::flood_does_not_panic_or_deadlock`).

## CI throughput

- Every PR: unit + adversarial + property + small-fuzz seed.
- Nightly: large-fuzz, criterion regressions, loom exhaustive.
- Per-tag: full SQLite-grade matrix including GATE_CLOSURE G1-G5.

## Coverage record

`scripts/coverage.sh` collects `cargo-llvm-cov` per crate.
Baseline for the 0.4.1 release train:

- vyre-foundation: 92% line / 81% branch
- vyre-driver: 88% line / 74% branch
- vyre-driver-wgpu: 76% line / 63% branch (wgpu surface is
  hostile to unit testing; bench + differential testing covers
  what line counts don't).
- consumer: 87% line / 78% branch

Release target: ≥ 95% line / ≥ 85% branch on every vyre core crate;
≥ 90% / ≥ 75% on consumer.

## Release evidence

Release readiness for this document is proven through the platform and dataflow evidence manifest and generated artifacts under `release/evidence/`. Claims here must map to concrete gate output, benchmark output, conformance output, parser corpus output, or documentation proof files before the release requirement can be closed.

Concrete evidence anchors:

- `release/evidence/tests/test-matrix.json`
- `release/evidence/tests/unit-suite.json`
- `release/evidence/tests/adversarial-suite.json`
- `release/evidence/tests/fuzz-suite.json`
