# Release engineering

**Status: Superseded.** Use [`docs/RELEASE.md`](RELEASE.md) for the active
release procedure.

Closes #34 (A.10 release engineering). Complements `docs/GATE_CLOSURE.md`
(the per-release gate protocol) with the day-to-day shape of
shipping a version.

## Version discipline

- publishable Vyre crates move in **lock-step** for a release train. The selected
  Vyre and Weir package versions come from `release/release-train.toml`, which is
  the single source of truth; `version-matrix` is the evidence gate for drift.
- `consumer` versions independently but **declares a tested Vyre
  minor** in its Cargo.toml. consumer integration is represented in parser
  coherence evidence; it is not the release tag owner for Vyre.
- Tier-3 dialect splits (`vyre-libs-nn`, `vyre-libs-crypto`, …)
  move on the vyre minor line.
- Tier-4 external packs (`vyre-libs-extern`, community authored)
  version independently per pack. The `ExternDialect` registration
  records the pack's minimum vyre minor.

## Publishing order

Each release pushes crates in dep-order so mid-publish breakage does not leave
downstream consumers linking a wedge-version. The canonical order is maintained
by the readiness report and must be checked with:

```sh
cargo_full run --bin xtask -- package-readiness
```

The report runs `cargo package --list` for every publish-order entry. It records
the normalized file count and BLAKE3 digest, requires metadata, both licenses,
Rust source, and a runnable example, and rejects internal instructions, secret
configuration, path traversal, and build output.

Publishing uses `cargo_full publish --dry-run --locked -p <crate>` and then
`cargo_full publish --locked -p <crate>` for each publishable crate after the
release evidence gate is closed.

## Tag format

- Vyre tag: `vyre-v<vyre version>`.
- Weir tag: `weir-v<weir version>`.
- Combined train tag: `vyre-<vyre version>-weir-<weir version>`.
- Every tag name is declared in `release/release-train.toml`; take them from there
  rather than typing them, and cut the release-candidate tags before the final tags.
- Release artifacts live under `release/evidence/` and include conformance,
  backend, benchmark, parser, optimization, metadata, docs, hygiene, and final
  completion-audit JSON.

Release evidence anchors:

- `release/evidence/final/completion-audit.json`
- `release/evidence/final/release-evidence-run.json`
- `release/evidence/final/expected-artifacts.json`
- `release/evidence/conformance/conformance-matrix.json`
- `release/evidence/benchmarks/cuda-release-suite.json`
- `release/evidence/benchmarks/release-workload-matrix.json`
- `release/evidence/benchmarks/cpu-only-100x-proof.json`
- `release/evidence/tests/release-surface-suite-coverage.json`
- `release/evidence/metadata/metadata-matrix.json`
- `docs/DOCS.toml`
- `docs/SUMMARY.md`
- `docs/INDEX.md`

## Release evidence external artifacts

`release-evidence` records externally refreshed benchmark artifacts as
`external-artifacts-only` instead of spawning `release-benchmarks`. The
expected-artifact registry at `release/evidence/final/expected-artifacts.json`
must expose `command_mode`, `artifact_contracts`, `blockers`, and the
`release-benchmarks --backend cuda` contract rows that keep long benchmark runs
outside the structural evidence command. A registry or artifact-status blocker
is a release-gate exit condition.

The release evidence run at `release/evidence/final/release-evidence-run.json`
inspects `release/evidence/benchmarks/cuda-release-suite.json` for
`schema_digest_chain` provenance. That chain must carry `source_digest`,
`command_digest`, and `hardware_digest` values, and the suite must also expose
top-level hardware provenance before benchmark freshness is accepted.

## Changelog protocol

`CHANGELOG.md` follows Keep-a-Changelog, one per crate:

- **Added / Changed / Deprecated / Removed / Fixed / Security** sections.
- Every item cross-references the audit or issue that drove it
  (`CRITIQUE_* Finding N`, `VISION V<n>`, `#<task>`). A reader
  tracing why a line of code moved must be one grep away from the
  source-of-truth rationale.
- Security-impacting changes (gate C1, C2, pocgen `dangerous-exploits`, …)
  go in the **Security** section and copy the `Fix:` hint from the
  fix commit so the changelog is actionable for downstream pinning
  decisions.

## Pre-flight checklist

1. `cargo_full run --bin xtask -- release-evidence`  -  structural evidence batch.
2. `cargo_full run --bin xtask -- launch-state --output release/evidence/final/public-launch-state.json`  -  truthful external-action state.
3. `cargo_full run --bin xtask -- release-completion-audit --output release/evidence/final/completion-audit.json`  -  prompt-to-artifact audit.
4. `cargo_full run --bin xtask -- vyre-release-gate --prepublish`  -  internal readiness gate that permits only the three explicit approval-gated outward actions.
5. `cargo_full test --workspace --release --all-features`  -  full workspace tests.
6. `cargo_full run -p vyre-bench --release -- run --backend cuda --suite release --measured-samples 30 --warmup-samples 3 --enforce-budgets`  -  CUDA release path.
7. `cargo_full run -p vyre-bench --release -- run --backend wgpu --suite release --measured-samples 30 --warmup-samples 3 --enforce-budgets`  -  WGPU fallback path.
8. Confirm `release/evidence/benchmarks/cpu-only-100x-proof.json` proves every required 100x release case (`release.condition_eval.1m`, `release.string_bitmap_scatter.1m`, `release.offset_count_aggregation.1m`, `release.entropy_window.1m`, `release.quantified_condition_loops.1m`, `release.alias_reaching_def.1m`, `release.ifds_witness.1m`, `release.c_ast_traversal.1m`, `release.megakernel_queue.1m`, `release.egraph_saturation.1m`, and `sparse.compaction.count.1m`) with 30+ CUDA and CPU baseline samples.
9. `cargo_full run -p vyre-conform-runner --release --features gpu --bin vyre-conform -- dispatch --backend cuda --ops all`  -  CUDA conformance.
10. `cargo_full run -p vyre-conform-runner --release --features gpu --bin vyre-conform -- dispatch --backend wgpu --ops all`  -  WGPU conformance.
11. `cargo_full publish --dry-run --locked -p <each crate>` in order.
12. After approved publication, repository verification, and push actions, regenerate launch-state and completion-audit evidence and run `cargo_full run --bin xtask -- vyre-release-gate` as the final hard gate.
13. Open the GitHub release with the evidence summary and conformance artifacts attached.

## Post-release

- Published tags stay published even if a patch ships shortly
  after. No retroactive rewriting of history.
- If a security finding appears post-release, the patch cadence is
  48 h from triage to crates.io push, with a CHANGELOG `Security`
  entry naming the CVE + affected versions.

## Open items

- Prepublication cannot begin until `vyre-release-gate --prepublish` accepts
  every internal requirement and reports only the three approval-gated outward
  actions as pending.
- Release cannot close until `release/evidence/final/completion-audit.json`
  has zero blockers and final-mode `vyre-release-gate` accepts every manifest
  requirement.
- A verified downstream artifact must cite the exact evidence files it relied
  on, not only a green CI run.
