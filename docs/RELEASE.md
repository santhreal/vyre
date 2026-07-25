# vyre release process

This document is the release contract. It states what must be true before vyre
publishes, and it is the only runbook for cutting a release. It names no version
number: the version, the tags, and the required release-note tokens all come from
`release/release-train.toml`, which is the single source of truth. Read them with

```bash
./cargo_full run --bin xtask -- version-matrix
```

Any deviation from this document is a process bug, not a shortcut.

For conflicts between release docs, generated docs, and internal archives, use
[`docs/DOCUMENTATION_GOVERNANCE.md`](DOCUMENTATION_GOVERNANCE.md).

## Backend contract

CUDA-first: CUDA is the release speed path, and WGPU is the portability and
correctness fallback. A WGPU performance miss is not a release blocker as long as
WGPU produces correct outputs; a WGPU correctness miss always is.

The CUDA path must keep `BufferDecl::output_byte_range` semantics intact for both
direct dispatch and cudaGraph replay. Narrow and zero-length readbacks are
performance features, not edge cases: megakernel control-report workloads depend
on them to avoid copying ring, debug, and IO buffers back to the host.

## Version story

Every publishable crate in the workspace publishes on every tagged release, and
every crate in a release group carries that group's version. The gate proves it:

```bash
./cargo_full run --bin xtask -- version-matrix     # zero blockers
./cargo_full run --bin xtask -- package-readiness  # zero blockers
```

Tags are product-scoped, never bare. The train declares three release-candidate
tags and three final tags, and the release-candidate tags are cut first so the
gate can run against them. A bare version tag is ambiguous between the outer
monorepo and the vyre crates and is never used.

Release notes live in this repository at `docs/release/v<version>.md` and must
contain every token the train lists in `required_release_note_tokens`. The
version matrix checks that for you.

## Publish order

The order is derived, not maintained by hand. Every path dependency must resolve
from crates.io before its consumer publishes, so publish in the topological order
the tool prints:

```bash
./cargo_full run --bin xtask -- package-readiness
```

The readiness report carries the publish order and the per-crate blockers. Do not
publish workspace tooling crates (`xtask`, `vyre-bench`, `vyre-conform-*`) unless
their manifests are deliberately changed to publishable.

## Pre-release checklist

1. `./cargo_full check --workspace --all-targets` is clean.
2. `./cargo_full test --workspace` passes.
3. `./cargo_full deny check` is green on licenses, advisories, and sources.
4. `./cargo_full public-api --all-features` matches the `docs/public-api/*.txt` baselines.
5. `./cargo_full semver-checks check-release` passes for every publishable crate.
6. The CUDA release benchmark suite passes its budgets:
   `./cargo_full run --bin xtask -- release-benchmarks --backend cuda`.
7. The WGPU fallback suite passes correctness.
8. `./cargo_full test -p vyre-driver-cuda cuda_honors_` proves the readback semantics,
   and `cuda_graph_honors_output_byte_ranges_like_direct_dispatch` proves them for
   cudaGraph replay.
9. `CHANGELOG.md` has an entry for the new version.
10. `CITATION.cff` version and release date match the tag.
11. The full gate reports zero blockers:
    `./cargo_full run --bin xtask -- vyre-release-gate`.

Benchmark evidence is keyed to a source-tree fingerprint, so any source change
after a benchmark run invalidates that run. Run the benchmarks last, after the
code is final, and re-run them if anything changes.

## Publish

For each crate, in the derived order:

```bash
./cargo_full publish --dry-run --locked -p <crate>
./cargo_full publish --locked -p <crate>
bash scripts/wait-crates-index.sh <crate> <version>
```

Wait for the index between crates. A dry run of a crate with internal
dependencies only succeeds once those dependency versions are already in the
registry, so a single up-front dry run of the whole workspace is not possible
while crates.io still holds the previous release.

## Tag

Cut the release-candidate tags, run the gate, then cut the final tags. Take the
exact tag names from the train rather than typing them:

```bash
./cargo_full run --bin xtask -- version-matrix   # prints the tag story and tag order
git tag <vyre rc tag> && git tag <combined rc tag>
git push origin <vyre rc tag> <combined rc tag>
# gate green, crates published
git tag <vyre final tag> && git tag <combined final tag>
git push origin <vyre final tag> <combined final tag>
gh release create <vyre final tag> --notes-file docs/release/v<version>.md
```

The weir repository cuts its own product tag before the combined release-train
tag counts as complete.

Release notes come from the changelog entries for the new version. Never write
them separately from the changelog.

## Rollback

Yank, never unpublish: crates.io does not permit unpublish after 72 hours, and a
disappearing version breaks every consumer that resolved it.

```bash
./cargo_full yank --vers <version> <crate>
```

Fix forward and publish the next patch version the same day.

## Completion audit checklist

A release is complete only when all of the following hold. Each line maps to a
generated artifact under `release/evidence/`, so none of it is self-reported.

1. The version matrix and package readiness reports have zero blockers.
2. Every requirement in `release/vyre-release-evidence.toml` is closed with its
   evidence artifacts present.
3. The benchmark evidence fingerprint matches the published source tree.
4. Every publishable crate is on crates.io at the train's version.
5. The product-scoped release-candidate and final tags are pushed, in that order.
6. The release notes for the version exist and carry every required token.
7. The README and `CITATION.cff` name the released version.

## Post-release

1. Update the README version references.
2. Update the install documentation.
3. File an issue for every finding surfaced during the release that did not block ship.

## Release evidence

Release readiness is proven by the evidence manifest
`release/vyre-release-evidence.toml` and the generated artifacts under
`release/evidence/`. Every claim in this document maps to gate output, benchmark
output, conformance output, parser corpus output, or a documentation proof file
before its requirement can be closed.
