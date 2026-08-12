# Release engineering

**Status: Superseded.** Use [`docs/RELEASE.md`](RELEASE.md) for the active
release procedure.

Closes #34 (A.10 release engineering). Complements `docs/GATE_CLOSURE.md`
(the per-release gate protocol) with the day-to-day shape of
shipping a version.

## Version discipline

- Publishable Vyre crates move in **lock-step** for a release train. The selected
  package version comes from `release/release-train.toml`, which is the single
  source of truth; `version-matrix` is the evidence gate for drift.
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
and Rust source, and rejects internal instructions, secret configuration, path
traversal, and build output.

Publishing uses `cargo_full publish --dry-run --locked -p <crate>` and then
`cargo_full publish --locked -p <crate>` for each publishable crate after the
release evidence gate is closed.

## Tag format

- Vyre release-candidate tag: `vyre-v<vyre version>-rc.1`.
- Vyre final tag: `vyre-v<vyre version>`.
- Both tag names are declared in `release/release-train.toml`; take them from
  there rather than typing them, and cut the release-candidate tag before the
  final tag.
- Release artifacts live under `release/evidence/` and include conformance,
  backend, benchmark, parser, optimization, metadata, docs, hygiene, and public
  launch JSON.

Release evidence anchors:

- `release/evidence/final/release-evidence-run.json`
- `release/evidence/final/expected-artifacts.json`
- `release/evidence/conformance/conformance-matrix.json`
- `release/evidence/benchmarks/cuda-release-suite.json`
- `release/evidence/benchmarks/release-workload-matrix.json`
- `release/evidence/benchmarks/cpu-only-100x-proof.json`
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
3. `cargo_full run --bin xtask -- vyre-release-gate --prepublish`  -  internal readiness gate that permits only the three explicit approval-gated outward actions.
4. `cargo_full test --workspace --release --all-features`  -  full workspace tests.
5. `cargo_full run -p vyre-bench --release -- run --backend cuda --suite release --measured-samples 30 --warmup-samples 300 --enforce-budgets`  -  CUDA release path.
6. `cargo_full run -p vyre-bench --release -- run --backend wgpu --suite release --measured-samples 30 --warmup-samples 300 --enforce-budgets`  -  WGPU fallback path.
7. Confirm `release/evidence/benchmarks/cpu-only-100x-proof.json` proves every currently registered 100x release case from `docs/optimization/BENCH_TARGETS.toml` with 30 or more CUDA and CPU baseline samples. The semantic optimizer impact workload is architectural evidence and is not part of the CPU-SOTA throughput family count.
8. `cargo_full run -p vyre-conform-runner --release --features gpu --bin vyre-conform -- dispatch --backend cuda --ops all`  -  CUDA conformance.
9. `cargo_full run -p vyre-conform-runner --release --features gpu --bin vyre-conform -- dispatch --backend wgpu --ops all`  -  WGPU conformance.
10. `cargo_full publish --dry-run --locked -p <each crate>` in order.
11. After approved publication, repository verification, and push actions, regenerate launch-state evidence and run `cargo_full run --bin xtask -- vyre-release-gate` as the final hard gate.
12. Open the GitHub release with the evidence summary and conformance artifacts attached.

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
- Release cannot close until `release/evidence/final/public-launch-state.json`
  reports complete and final-mode `vyre-release-gate` accepts every manifest
  requirement.
- A verified downstream artifact must cite the exact evidence files it relied
  on, not only a green CI run.
