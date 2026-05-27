# Release 1.0  -  gate checklist

P-RELEASE-1. Every box must tick before the 1.0 tag.

## Build + test

- [ ] `cargo build --workspace --release` clean.
- [ ] `cargo test --workspace --release` all green.
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` clean.
- [ ] `cargo fmt --check` clean.
- [ ] CI green on the launch branch (3 consecutive runs).

## Conform

- [ ] `cargo test -p vyre-conform-runner` green on wgpu adapter.
- [ ] `scripts/check_spirv_parity_perf_gate.sh` green for SPIR-V lowering
  parity/performance.
- [ ] `scripts/check_cuda_parity_perf_gate.sh` green on a CUDA-visible GPU
  runner; CUDA probe failure is a configuration failure, not a skip.
  The gate auto-discovers every `*gpu_parity*` integration test, including
  `int4_quantized_gpu_parity` for all six `vyre-libs::quant::int4_*` harness ops.
- [ ] `cargo test -p vyre-conform-runner --features cuda --test release_gate_contracts`
  green so CUDA/SPIR-V floors and nightly gates cannot regress to unchecked.
- [ ] `cargo run -p vyre-conform-runner --features cuda -- dispatch --backend cuda --ops all`
  emits CUDA dispatch certificates against `vyre-reference`.
- [ ] `cargo test -p vyre-conform-runner` green on the photonic
  contract-check adapter (CI-gated; hardware-absent runs the
  contract target).
- [ ] Per-shape sweep `bash libs/tools/consumer/scripts/per_shape_sweep.sh`
  reports precision = 1.0 on every shipped shape.

## Recursion thesis

- [ ] ≥ 80 % of substrate consumers identified in
  `docs/RECURSION_THESIS.md` are inside vyre.
- [ ] No substrate consumer ships in two crates at once
  (deduplication audit clean).

## Publishing

- [ ] `cargo publish --dry-run -p vyre-foundation` clean.
- [ ] `cargo publish --dry-run -p vyre-spec` clean.
- [ ] `cargo publish --dry-run -p vyre-driver` clean.
- [ ] `cargo publish --dry-run -p vyre-driver-wgpu` clean.
- [ ] `cargo publish --dry-run -p vyre-driver-spirv` clean.
- [ ] `cargo publish --dry-run -p vyre-libs` clean.
- [ ] `cargo publish --dry-run -p vyre-runtime` clean.
- [ ] `cargo publish --dry-run -p vyre-primitives` clean.
- [ ] `cargo publish --dry-run -p vyre-intrinsics` clean.
- [ ] `cargo publish --dry-run -p vyre-harness` clean.
- [ ] `cargo publish --dry-run -p vyre` (meta-shim) clean.

## Documentation

- [ ] Every public crate has a `docs/ARCHITECTURE.md`.
- [ ] `VISION.md` recursion-trajectory section reflects current
  state.
- [ ] `BENCHMARKS.md` numbers from the latest standard-corpus run.
- [ ] `RECURSION_THESIS.md` table reflects the latest absorption
  status.

## consumer launch

- [ ] All 13 launch-class rules ship in
  `libs/tools/consumer/rules/launch/remote_overflow/shapes/`.
- [ ] Each shape's `rule.gate.toml` declares `precision_min = 1.00`.
- [ ] `verify_class1_pocs.sh` runs clean on the linux/chromium/oss
  demo outputs.
- [ ] `certify_launch_findings.sh` produces a deterministic sha256.
- [ ] `preflight_smoke_multi_backend.sh` agrees across wgpu/cuda/
  spirv on the smoke fixture.

## Cross-machine reproducibility

- [ ] `cert.toml` regenerates bit-identically on a second host
  using the same corpus snapshots.
- [ ] Conformance certificates round-trip through
  `validate_certificate` for every demo finding.
