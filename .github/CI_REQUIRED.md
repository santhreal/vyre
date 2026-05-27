# Required CI Jobs for Branch Protection

All jobs listed below are **required** to pass before a PR can merge into `main`.
This list is enforced by branch protection rules (see `scripts/apply-branch-protection.sh`).

## From `ci.yml` (run on every PR + push to main)
- `${os} / ${rust-toolchain}` matrix entries for:
  - `./cargo_full fmt --check`
  - `./cargo_full run -p xtask -- abstraction-gate`
  - `./cargo_full clippy --workspace -- -D warnings`
  - `./cargo_full test --workspace`
  - `./cargo_full doc --workspace`

## From `bench.yml` (run on PRs touching benchmarked crates)
- `criterion-regression`

## From `architectural-invariants.yml` (run on every PR)
- `architectural-invariants`
- `law-a-closed-enums`
- `law-b-string-wgsl`
- `law-b-shader-assets`
- `law-c-capability-negotiation`
- `law-d-unsafe-justifications`
- `dialect-coverage`
- `trait-freeze`
- `registry-consistency`
- `no-raw-unwrap`
- `no-hot-path-inventory`
- `no-opspec-tokens`
- `error-codes-cataloged`
- `consistency-contracts`
- `base-monument`
- `abstraction-gate`
- `vyre-lints-raw-ir`
- `vyre-lints-allowlist-drift`
- `op-matrix-coverage`
- `lego-audit`

## From `conform.yml` (run on every PR)
- `conformance matrix release blocker`
- `Operation matrix release gate`
- `conform/* CPU substrate`
- `Conform release gate`

## From `gpu-parity.yml` (run on self-hosted GPU runner)
- `Probe real GPU adapter`
- `WGPU backend contracts`
- `Composition parity on real GPU`
- `Determinism stress on real GPU`
- `Mandatory GPU enforcement`
- `CUDA conformance all ops`
- `WGPU conformance all ops`
- `Release conformance artifacts`
- `Weir CUDA parity`
- `CUDA release benchmark evidence`
- `GPU release gate`

## From `reproducible-build.yml` (nightly schedule)
- `reproducible`  -  nightly gate; not blocking on individual PRs but tracked in cycle reports.

## Scheduled or Manual Deep Gates
- `fuzz.yml`  -  full fuzz lane once active fuzz targets exist.
- `mutation-testing.yml`  -  weekly zero-survivor gate once restored from `workflows-paused`.
