# Release hygiene proof

This artifact backs `release-hygiene`.

Evidence sources:

Required generated evidence:

- `release/evidence/hygiene/hygiene-matrix.json`
- `release/evidence/hygiene/no-stubs-scan.json`
- `release/evidence/hygiene/no-hidden-fallback-scan.json`
- `release/evidence/hygiene/resource-bound-scan.json`
- `release/evidence/hygiene/error-surface-scan.json`
- `release/evidence/hygiene/cargo-wrapper-scan.json`
- `release/evidence/hygiene/audit-location-scan.json`
- `release/evidence/hygiene/public-doc-scan.json`
- `release/evidence/hygiene/test-hygiene-scan.json`

Release contract:

- No shipped source stubs, TODO/FIXME markers, placeholder text, `todo!`, or `unimplemented!`.
- `hygiene-matrix.json` must include `finding_summary`, and the summed summary counts must match the raw `findings` length so categorized hygiene evidence cannot drift from findings.
- `hygiene-matrix.json` must include `release_surface_coverage`, and every required release-surface flag must be true for the Vyre workspace, `vyre-driver-cuda`, `vyre-driver-wgpu`, the dataflow workspace, `tools/vyrec`, the downstream analyzer tool, the security grammar generator, release scripts, GitHub workflows, and branch-protection controls.
- `release_surface_coverage.resource_bound_patterns`, `release_surface_coverage.hidden_fallback_patterns`, and `release_surface_coverage.release_tooling_patterns` must list every blocked release pattern family required by the plan, including whole-file reads, sleeps, no-GPU skips, CPU/software fallback language, synthetic GPU timing formulas, raw workspace cargo, malformed `cargo_full xtask`, heredocs, and missing cargo wrappers.
- Public shipped types must have Rustdoc comments.
- Production source must not use unbounded whole-file reads on release paths.
- Backend caches and durability queues that retain runtime artifacts must expose explicit caps or eviction paths in release evidence; CUDA PTX source cache and WGPU disk-cache pending flush queues are release-surface resource-bound contracts.
- No test TODO/FIXME markers, ignored tests, vacuous assertions, or discarded test results.
- No hidden CPU fallback language in production backend paths.
- Audit, findings, and plan reports must live under `.audits/` or the project `audits/` archive; stray root-level reports block release.
- Hygiene scanning covers Vyre, the dataflow workspace, the CUDA/WGPU driver release roots, and distributed parser/tooling source including `tools/vyrec`, analyzer tools under `libs/tools/`, and grammar generators under `libs/shared/` when those roots exist.
- Release CI hygiene covers `.github/workflows/architectural-invariants.yml`, `.github/CI_REQUIRED.md`, `scripts/apply-branch-protection.sh`, and `scripts/architectural_invariants.sh` so required branch-protection contexts cannot drift from implemented workflows.
- Hidden-fallback scanning rejects silent no-GPU skips, `GpuUnavailable` skip/fallback branches, CPU/software fallback wording, and `cfg(not(feature = "gpu"))` GPU test escapes.
- Benchmark hygiene rejects synthetic GPU timing and known fake timing formulas.
- No raw workspace `cargo` commands in release tooling; use `cargo_full`.
- The Santh root has a `cargo_full` wrapper for root workflows; Vyre-local workflows use `libs/performance/matching/vyre/cargo_full` from the Vyre working directory.
- No heredocs in release tooling.
