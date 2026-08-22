# Release

```sh
./cargo_full run --bin xtask -- version-matrix
./cargo_full run --bin xtask -- gates --subset prepublish
```

`release/release-train.toml` is the manifest. It names the version, the
tags, the packages in the release group, the tokens the release notes must
carry, and the packaging steps that must hold. Nothing about a release is
decided outside it.

## The manifest

`[versions]` carries one version per product. `vyre` is at `0.8.0`.

`[tags]` carries product-scoped tags, `vyre-v0.8.0-rc.1` and
`vyre-v0.8.0`. A bare `v0.8.0` tag is rejected by policy: this repository
is one product among several under one account, and a bare tag is ambiguous
about which.

`[release_groups.vyre]` names the repository, the version key and the 26
packages published together.

`required_release_note_tokens` is the set of strings the release notes must
contain. `required_packaging_steps` is the set of ordering facts that must
hold: versions align before a tag is cut, release-candidate tags precede
final tags, notes cite the product-scoped tags, and artifacts are generated
only after the version matrix reports zero blockers.

Those three keys and `package_verify_passed` are top-level and must sit
above the first table header. In TOML a bare key written after `[tags]`
binds into `[tags]`, which drops it from the parsed manifest and makes
`version-matrix` fail with a missing-field error.

## Gates

`xtask version-matrix` reads the manifest and reports blockers.

`xtask gates --subset prepublish` is what must hold before publishing,
beyond what a `cargo publish --dry-run` catches: `operation-schema`,
`list-ops`, `catalog`, `gate1`, `abstraction-gate`, `cross-target`,
`dep-drift`, `platform-boundary`, `vyre-release-gate`, `lockfile-clean`.

Run one gate by name, or a subset with `--subset`. `xtask gates` with no
subset runs the whole registry. Every gate returns findings; the runner
decides what a finding count means, and `xtask/gate-baselines.toml` pins
the count each gate is allowed. A pin moves down.

## Evidence

Release readiness is proven by generated artifacts under
`release/evidence/`, not by a claim in a document. A claim maps to concrete
gate output, benchmark output, conformance output or a documentation proof
file before its requirement closes. Every filesystem path cited inside
release evidence is checked to resolve, and to be reachable rather than
gitignored, by the `evidence-paths` gate.

## Changelog

An observable change adds one fragment under
`release/changes/unreleased/`, named by its id, as a TOML file with
exactly two keys:

```toml
category = "Fixed"
text = "One sentence naming what changed and what it changed from."
```

The six categories, in the order a section renders them: `Added`,
`Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`. Any other
`category` value, and any third key, is a finding.

`release-docs` regenerates two documents from the fragments and the release
train: `CHANGELOG.md` and `release/evidence/docs/release-notes-body.md`.
Neither is edited by hand, and a fragment is not a summary of the commit that
carried it. Both are generated pages, so a rule that judges authored prose
skips them.
