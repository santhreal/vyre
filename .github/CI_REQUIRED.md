# Required CI Jobs for Branch Protection

All jobs listed below are **required** to pass before a PR can merge into `main`.
The `ci-required` gate holds this list to the workflows that define it, and
`scripts/apply-branch-protection.sh` applies it to the branch.

## From `ci.yml` (run on every PR + push to main)
- `CI release gate`

## From `gates.yml` (run on every PR + push to main)
- `Every registered gate`
- `Rules the sweep owns, named by domain`
- `Every feature compiles alone on the MSRV`
- `Every feature compiles alone or records why it cannot`
- `Oracle-matrix sweeps ci.yml cannot reach`
- `Emitted SPIR-V is validated, not just shaped`

## From `bench.yml` (run on every PR and push to main)
- `criterion-regression`

## From `architectural-invariants.yml` (run on every PR)
- `Architecture release gate`

## From `conform.yml` (run on every PR)
- `Conform release gate`

## From `gpu-parity.yml` (run on self-hosted GPU runner)
- `GPU release gate`

## Scheduled or Manual Deep Gates

Not blocking on individual PRs. Tracked in cycle reports.

- `fuzz.yml`  -  full fuzz lane once active fuzz targets exist.
- `mutation-testing.yml`  -  weekly zero-survivor gate once restored from `workflows-paused`.
- `reproducible-build.yml`  -  nightly `reproducible` gate once restored from `workflows-paused`.
