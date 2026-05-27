# Conformance documentation proof

This artifact backs `conformance-hard-gate`.

Evidence sources:

Required generated evidence:

- `release/evidence/conformance/conformance-matrix.json`
- `release/evidence/conformance/cuda-conformance.json`
- `release/evidence/conformance/wgpu-conformance.json`
- `release/evidence/conformance/reference-conformance.json`
- `release/evidence/conformance/release-gate-log.json`
- root `.github/workflows/conform.yml`
- root `.github/workflows/gpu-parity.yml`
- root `.github/CI_REQUIRED.md`
- root `scripts/apply-branch-protection.sh`
- nested `libs/performance/matching/vyre/.github/workflows/conform.yml`
- nested `libs/performance/matching/vyre/.github/workflows/gpu-parity.yml`

Release contract:

- CUDA, WGPU, and CPU reference conformance artifacts must exist and be non-empty.
- `conformance-matrix.json` and every backend conformance artifact must prove OP_MATRIX-required catalog coverage with zero `missing_catalog_ops`.
- OP_MATRIX release backend rows must not contain `blocked_release` for `reference`, `cuda`, or `wgpu`; `op_matrix_blocked_release_count` must be zero in both global and backend conformance artifacts.
- `conformance-matrix.json` and every backend conformance artifact must expose `release_backend_row_count` covering all required OP_MATRIX ops across `reference`, `cuda`, and `wgpu`, and `missing_release_backend_rows` must be empty.
- CI must block on conformance matrix generation and all-backend release conformance through the active root Santh `conform-release-gate`, active root GPU `gpu-release-gate`, and branch-protection required-job list.
- CI must also block on the root GPU workflow `Vyre/Weir final release gate`, which downloads `vyre-release-conformance-evidence` and `vyre-release-benchmark-evidence`, stages conformance, benchmark, and optimization artifacts, regenerates structural evidence, emits the completion audit at `release/evidence/final/completion-audit.json`, and runs `vyre-release-gate`.
- The final GPU workflow must prove optimization evidence staging through `release/evidence/optimization`, exact completion-audit output wiring, and upload of `vyre-weir-final-release-evidence`.
- Branch-protection evidence must include `.github/CI_REQUIRED.md` and `scripts/apply-branch-protection.sh`; the required-status list and the applier cannot drift into separate hardcoded contracts.
- `conformance-matrix.json` must expose non-empty `required_ci_statuses` and empty `missing_required_ci_statuses`, proving every branch-protection context listed before the scheduled/manual section is backed by an actual workflow job.
- Dynamic matrix jobs must have static fan-in statuses when they are release-required; the crate batch matrix is represented by the static `crate-checks` context.
- Required workflows that contribute branch-protection contexts must not use path filters; required contexts must appear on every pull request and every push to `main`, and `conformance-matrix.json` must report empty `path_filtered_required_workflows`.
- Nested Vyre release guardrails for conformance, GPU parity, CI, benches, fuzz smoke, architectural invariants, feature matrix, core, rewrite proofs, and lego-block audit are part of the path-filter regression scan; a skipped workflow is release-blocking evidence drift, not a benign optimization.
- Required fan-in contexts must fail closed on dependency failures; `crate-checks`, `Conform release gate`, `GPU release gate`, and `Vyre/Weir final release gate` cannot rely on skipped dependency semantics.
- `conformance-matrix.json` must report empty `missing_fail_closed_fanins`.
- The root `conform-release-gate` fan-in job must run the conformance matrix generator itself and upload the generated release artifact; it cannot be an echo-only advisory job.
- Conformance fuzz, loom, and benchmark-compare jobs must fail when their release-critical inputs are absent; they cannot silently skip missing fuzz targets, loom tests, or benchmark suites.
- The nested Vyre workflows must also expose the conformance matrix release blocker job and GPU release gate so repo-local CI cannot drift from the root gate.
- Conformance is a release gate, not advisory documentation.
