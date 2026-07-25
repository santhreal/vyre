# Vyre release checklist

Run this checklist before any crates.io publish or release tag. Every checked
box must map to concrete evidence under `release/evidence/` or CI output from a
required workflow.

## Evidence gates

- [ ] `cargo_full run --bin xtask -- release-evidence`
- [ ] `cargo_full run --bin xtask -- release-completion-audit --output release/evidence/final/completion-audit.json`
- [ ] `cargo_full run --bin xtask -- vyre-release-gate`
- [ ] `release/evidence/final/completion-audit.json` reports zero blockers.
- [ ] `release/vyre-release-evidence.toml` has every requirement covered by named artifacts.

## CUDA-first / WGPU fallback

- [ ] CUDA release suite evidence exists and selects backend `cuda`.
- [ ] CUDA PTX pattern evidence proves positive ldmatrix-capable target coverage.
- [ ] WGPU fallback suite evidence exists and selects backend `wgpu`.
- [ ] GPU probe failures are loud and actionable; no CPU fallback or no-GPU skip is accepted.
- [ ] `bench-release` reads evidence axes and fails on missing axes rather than printing `unmeasured`.

## Optimization and proof workloads

- [ ] Optimization corpus has at least 1000 generated semantic-preserving cases.
- [ ] Optimization family manifest covers at least 14 required release families with 128 generated cases per required family.
- [ ] A13-A16 analysis fixture evidence covers coalesce, shared-memory promotion, bank conflicts, and vector packing.
- [ ] Alias-aware DSE, STLF, LICM, loop fusion, and loop fission have before/after benchmark evidence.
- [ ] E-graph saturation evidence proves arithmetic, bitwise, and boolean predicate-chain coverage.
- [ ] At least 12 named workload artifacts back the 10+ workload-family requirement.
- [ ] Seven required release workloads prove 100x+ CUDA wins against CPU-SOTA baselines with 30+ GPU and CPU baseline samples.

## Parser and dataflow integration

- [ ] C parser Linux subsystem corpus evidence exists with AST, diagnostics, provenance, fingerprint, and throughput data.
- [ ] Parser coherence evidence covers `vyre-frontend-c`, `tools/vyrec`, dataflow integration, consumer, and grammar generation boundaries.
- [ ] `tools/vyrec` emits actionable `Fix:` diagnostics for user-facing failures.
- [ ] Weir metadata, README, examples, soundness vocabulary, and Vyre integration evidence are coherent at the version the release train declares.
- [ ] Dataflow-engine release evidence is refreshed and linked from the release gate.

## Conformance and CI

- [ ] Root `.github/workflows/conform.yml` is required and generates conformance matrix evidence.
- [ ] Nested `.github/workflows/conform.yml` is required and blocks unsupported op/backend claims.
- [ ] Root `.github/workflows/gpu-parity.yml` probes the real GPU runner and runs CUDA plus WGPU conformance.
- [ ] `.github/CI_REQUIRED.md` names all release-blocking conformance and GPU jobs.
- [ ] Release OP_MATRIX rows have zero `blocked_release` status for `reference`, `cuda`, and `wgpu`.

## Documentation, metadata, and hygiene

- [ ] Active release docs use every version and product-scoped tag token that `release/release-train.toml` declares in `required_release_note_tokens`. `version-matrix` checks this.
- [ ] Active release docs use `cargo_full`; xtask commands use `cargo_full run --bin xtask -- ...`.
- [ ] Crate metadata, features, docs, readmes, licenses, and version policy evidence are coherent.
- [ ] Hygiene evidence has zero findings for stubs, hidden fallbacks, raw cargo workflow commands, heredocs, public docs, and test hygiene.
- [ ] Benchmark hygiene rejects synthetic GPU timing or fake timing formulas.

## Publish and tags

- [ ] Publish order matches the order `cargo_full run --bin xtask -- package-readiness` derives.
- [ ] Every publishable crate dry-runs with `cargo_full publish --dry-run --locked -p <crate>`.
- [ ] Publish each crate with `cargo_full publish --locked -p <crate>` only after evidence gates close.
- [ ] Cut the three release-candidate tags the release train names, in the order it lists them.
- [ ] Cut the three final tags the release train names, after the gate is green and the crates are published.
- [ ] GitHub release notes cite the generated evidence summary and conformance artifacts.

## Rollback

If post-publish a critical bug is found:

1. Yank the affected crate version with `cargo_full yank --vers <version> <crate>`.
2. Fix forward in a patch release.
3. Document the yank reason in the changelog and release notes.
4. Re-run the release evidence gates before publishing the patch.
