# Testing Program  -  SQLite / NASA JPL / Linux / Chromium Standard

**Status: Superseded.** Use `docs/testing/TESTING.toml` and the generated
guides under [`docs/testing/`](testing/) for current test commands and evidence.

Closes #31 A.7 testing program direction.

## The bar

**Happy path is not the testing bar.** Unit tests prove oracle-backed behavior on
representative and boundary inputs; adversarial, gap, property, fuzz, and conform
work carry most of the safety budget.

This program implements one non-negotiable axiom, stated here in full so that no
reader depends on a document outside this repository: **if all tests pass, there
must be no known micro issue in scope.** Not "no bugs", not "happy path works".
No doc lie, no weak assertion, no missing contract, no gate that checks the wrong
thing, no correct-but-substandard shortcut sitting in the open. Passing tests are
not comfort, they are a claim that no known debt remains at this bar, so if you
can name a flaw while CI is green then the suite is lying and the suite is what
you repair first. Three corollaries are the ones that bite:

- Discovery turns something red. A review or audit that names a flaw adds or
  tightens a test or gate that fails until the flaw is fixed. Never "we will test
  later".
- A test that cannot fail on a bar miss is a suite defect, not coverage. Rewrite
  it or delete it. It does not count.
- Green CI with a known open flaw is invalid. Fix the code, add the failing test,
  or lower the written bar in public. Never silence the signal.

Volume is not excellence: a million weak tests do not satisfy the axiom, and one
falsifiable test per micro class does. Large-scale execution (proptest, fuzz,
conform) serves the axiom rather than replacing it.

Push reporting for multi-crate waves uses an internal maintainer template that is
not part of this repository and has no bearing on an external contribution.
Nothing in this program depends on it.

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

`scripts/check_test_coverage_per_crate.sh` verifies that each crate has an executable test target.
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

Release verification uses the package commands in [`docs/testing/`](testing/),
the conformance matrix, backend integration tests, parser corpus tests, and
benchmark artifacts under `release/evidence/`.
