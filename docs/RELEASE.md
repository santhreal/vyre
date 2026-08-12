# Vyre release process

This runbook applies to the active release train in
`release/release-train.toml`. It is the only operator procedure for cutting a
Vyre release.

The active train currently declares Vyre 0.7.2. The release document check
rejects this runbook when that version moves.

Use the generated [`release checklist`](RELEASE_CHECKLIST.md) while you run this
procedure. Do not edit versions, packages, repositories, tags, or approval
actions in that checklist. Regenerate them from the release train:

```bash
python3 scripts/release_docs.py --write
python3 scripts/release_docs.py --check
```

Any deviation from this procedure is a process bug. For conflicts with
historical or generated documents, use
[`docs/DOCUMENTATION_GOVERNANCE.md`](DOCUMENTATION_GOVERNANCE.md).

## Sources of truth

The release process has five authorities:

1. `release/release-train.toml` defines versions, release groups, package
   membership, repositories, product-scoped tags, release-note tokens, and the
   three approval-gated external actions.
2. `release/changes/unreleased.toml` defines the validated changelog fragments.
   The release document generator writes the `CHANGELOG.md` Unreleased section
   from these fragments.
3. `docs/release/v<version>.md` contains the release notes. The generator owns
   its release-group and tag preamble plus the validated change list. You
   maintain the example-first summary and upgrade guidance.
4. `release/vyre-release-evidence.toml` defines the evidence required by the
   release gate.
5. `release/evidence/package/publish-readiness.json` defines the
   dependency-safe package order after `package-readiness` regenerates it.

The generator fails closed when a package appears in two groups, a group lacks
its repository or version authority, a changelog fragment is missing, a
release-note token is absent, or generated release prose is stale.

## Backend contract

CUDA is the release speed path. WGPU is the portability and correctness
fallback. A WGPU performance miss does not block release when its results remain
correct. A WGPU correctness miss always blocks release.

The CUDA path must preserve `BufferDecl::output_byte_range` for direct dispatch
and cudaGraph replay. Narrow and zero-length readbacks are required behavior.
Megakernel control-report workloads use them to avoid copying ring, debug, and
IO buffers back to the host.

## Prepare the release

Start from a clean `main` branch in the Vyre release repository. Update the
release train and changelog fragments, then regenerate the release documents.
Run these gates before any external action:

```bash
python3 scripts/release_docs.py --write
python3 scripts/release_docs.py --check
scripts/check_docs_index.sh
./cargo_full check --workspace --all-targets
./cargo_full test --workspace
./cargo_full deny check
./cargo_full public-api --all-features
./cargo_full semver-checks check-release
./cargo_full run --bin xtask -- version-matrix \
  --output release/evidence/version/version-matrix.json
./cargo_full run --bin xtask -- package-readiness \
  --output release/evidence/package/publish-readiness.json
```

The version matrix must report zero blockers. Package readiness must report zero
blockers and a package for every member declared by the release groups.

Run the CUDA release benchmarks and WGPU correctness suite after production
source is frozen:

```bash
./cargo_full run --bin xtask -- release-benchmarks --backend cuda
./cargo_full test -p vyre-driver-cuda cuda_honors_
./cargo_full test -p vyre-driver-cuda \
  cuda_graph_honors_output_byte_ranges_like_direct_dispatch
```

Benchmark evidence carries a runtime source-tree fingerprint. Production source
changes invalidate it. Generated evidence, release tooling, tests, and
operator-internal files do not affect that fingerprint.

Generate the pending launch state and run the prepublication gate:

```bash
./cargo_full run --bin xtask -- launch-state \
  --output release/evidence/final/public-launch-state.json
./cargo_full run --bin xtask -- vyre-release-gate --prepublish
scripts/final-launch.sh --preflight
```

The prepublication gate must leave only the three external actions from the
release train blocked pending explicit approval. Preflight performs no publish,
tag, commit, visibility change, or push.

## Approval boundary

Do not perform an external release action until the user explicitly approves
that action. The three action classes are:

1. Publish approved crates in dependency order.
2. Verify the approved public repository visibility.
3. Push the release branches, product-scoped tags, and release record.

The launch script verifies visibility. It never changes repository visibility.
It also refuses dirty repositories, detached heads, non-`main` branches,
unexpected remotes, or pre-existing tags.

After approval, load the train-derived token and run the guarded launcher:

```bash
source scripts/lib/release_train.sh
vyre_load_release_train
VYRE_RELEASE_APPROVED="$VYRE_RELEASE_LAUNCH_APPROVAL_TOKEN" \
  scripts/final-launch.sh
```

## Executed order

The guarded launcher performs the release in this order:

1. Prove the sharded all-backend conformance certificate.
2. Cut and push the Vyre release-candidate tag.
3. Run `vyre-release-gate --prepublish` against the candidate state.
4. Publish packages in the dependency-safe order from package readiness. Each
   package waits for the registry index before its consumers publish.
5. Cut and push the Vyre final tag.
6. Create the public Vyre release record from the generated release-note
   metadata and maintained change summary.
7. Regenerate the public launch state, then run the final release gate.
8. Commit the completion evidence and push the Vyre release branch.

The script takes every version, repository, and tag from
`release/release-train.toml`. It does not use a bare version tag.

## Package contents

The readiness report carries one `cargo package --list` proof for each
publishable crate. Each proof records a BLAKE3 file-list digest and rejects
missing metadata, licenses, source, internal instruction files, secret
configuration, path traversal, and build output.

Workspace tooling crates such as `xtask`, `vyre-bench`, and
`vyre-conform-*` remain unpublished unless their manifests deliberately join a
release group.

## Completion

A release is complete only after the approved actions have actually succeeded
and the final gate reports zero blockers. The final evidence must prove:

1. Every release requirement is closed with its artifacts present.
2. Benchmark source fingerprints match the released source.
3. Every declared package version is available from its registry.
4. The Vyre release-candidate and final tags exist in the public repository and
   follow the train order.
5. Release notes contain every required token.
6. `README.md` and `CITATION.cff` agree with the released version.

Create the public release record from `docs/release/v<version>.md` only after
these checks pass.

## Rollback

Yank an affected package version. Never delete or overwrite a published version
or tag.

```bash
./cargo_full yank --vers <version> <crate>
```

Add a changelog fragment that explains the failure. Fix forward with the next
patch version, regenerate evidence, and use new product-scoped tags.
