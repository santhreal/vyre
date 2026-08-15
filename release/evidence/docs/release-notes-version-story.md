# Release notes version story evidence

This artifact backs `release/vyre-release-evidence.toml` requirement `version-story`.

Evidence sources:

- `release/evidence/version/version-matrix.json`
- `release/evidence/version/release-tag-plan.json`
- `release/evidence/docs/release-notes.md`

Required product versions:

- Vyre release: `0.7.2`
- Required version-matrix packages: `vyre@0.7.2`, `vyre-driver-cuda@0.7.2`, and `vyre-driver-wgpu@0.7.2`; `missing_required_release_packages` must be empty.
- Workspace-inherited package versions count only when the matrix resolves them to the concrete release version; unresolved `package.version.workspace = true` entries are blockers, not acceptable evidence.

Required product-scoped tags:

- Vyre RC tag: `vyre-v0.7.2-rc.1`
- Vyre tag: `vyre-v0.7.2`

Before requesting approval for publication or pushes, run
`cargo_full run --bin xtask -- vyre-release-gate`. The default prepublication
mode accepts only the three explicitly approval-gated outward actions as
pending. `--launch-complete` requires those actions to be done and is only
meaningful after the release has shipped.

Required pre-tag gates:

- `cargo_full run --bin xtask -- version-matrix --output release/evidence/version/version-matrix.json`
- `cargo_full run --bin xtask -- vyre-release-gate`
- `scripts/apply-branch-protection.sh main`

Release-note wording contract:

- Release notes must name `Vyre 0.7.2`.
- Release notes must name `vyre-v0.7.2-rc.1`.
- Release notes must name `vyre-v0.7.2`.
- Release notes must not instruct maintainers to create or push a bare `v0.7.2` tag for this release train.
- The version matrix scans the Vyre release-note documents for ambiguous bare tag commands.
