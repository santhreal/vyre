# WGPU fallback proof

This artifact backs `wgpu-fallback`.

Evidence sources:

Required generated evidence:

- `release/evidence/backends/backend-matrix.json`
- `release/evidence/benchmarks/wgpu-fallback-suite.json`
- `release/evidence/conformance/conformance-matrix.json`
- `release/evidence/conformance/wgpu-conformance.json`

Release contract:

- WGPU is the portable GPU fallback path, not a CPU fallback.
- `backend-matrix.json` must prove preferred runtime acquisition is GPU-only; if CUDA and WGPU are unusable, release code must surface a GPU configuration error rather than selecting `cpu-ref`/`reference`.
- WGPU must acquire successfully when selected and must expose the required fallback feature markers in `backend-matrix.json`.
- `wgpu-fallback-suite.json` must cover at least 12 required workload families, include one artifact status per family, preserve source family id and requested case id, prove selected backend parity, report zero failed cases, report zero backend mismatches, and include at least 30 `wall_ns` and `baseline_wall_ns` samples per case.
- `wgpu-conformance.json` must prove backend `wgpu`, at least 49 op pairs, distinct op coverage for the required catalog, zero failed pairs, and no blocked-release OP_MATRIX rows.
- Fallback evidence must fail loudly on adapter/device misconfiguration; hidden CPU fallback wording or no-GPU skip behavior is a release hygiene blocker.
