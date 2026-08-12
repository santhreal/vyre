# Required CI Jobs for Branch Protection

All jobs listed below are **required** to pass before a PR can merge into `main`.
This list is enforced by branch protection rules (see `scripts/apply-branch-protection.sh`).

## From `ci.yml` (run on every PR + push to main)
- `CI release gate`

## From `bench.yml` (run on every PR and push to main)
- `criterion-regression`

## From `architectural-invariants.yml` (run on every PR)
- `Architecture release gate`

## From `conform.yml` (run on every PR)
- `Conform release gate`

## From `gpu-parity.yml` (run on self-hosted GPU runner)
- `GPU release gate`

## From `reproducible-build.yml` (nightly schedule)
- `reproducible`  -  nightly gate; not blocking on individual PRs but tracked in cycle reports.

## Scheduled or Manual Deep Gates
- `fuzz.yml`  -  full fuzz lane once active fuzz targets exist.
- `mutation-testing.yml`  -  weekly zero-survivor gate once restored from `workflows-paused`.
