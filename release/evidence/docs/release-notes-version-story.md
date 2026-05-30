# Release notes version story evidence

This artifact backs `release/vyre-release-evidence.toml` requirement `version-story`.

Evidence sources:

- `release/evidence/version/version-matrix.json`
- `release/evidence/version/release-tag-plan.json`
- `release/evidence/docs/release-notes.md`

Required product versions:

- Vyre release: `0.6.1`
- Weir release: `0.1.0`
- Required version-matrix packages: `vyre@0.6.1`, `vyre-driver-cuda@0.6.1`, `vyre-driver-wgpu@0.6.1`, `vyrec@0.1.0`, `vyre-frontend-c@0.6.1`, and `weir@0.1.0`; `missing_required_release_packages` must be empty.
- Workspace-inherited package versions count only when the matrix resolves them to the concrete release version; unresolved `package.version.workspace = true` entries are blockers, not acceptable evidence.

Required product-scoped tags:

- Vyre RC tag: `vyre-v0.6.1-rc.1`
- Weir RC tag: `weir-v0.1.0-rc.1`
- Combined release-train RC tag: `vyre-0.6.1-weir-0.1.0-rc.1`
- Vyre tag: `vyre-v0.6.1`
- Weir tag: `weir-v0.1.0`
- Combined release-train tag: `vyre-0.6.1-weir-0.1.0`

Required pre-tag gates:

- `cargo_full run --bin xtask -- version-matrix --output release/evidence/version/version-matrix.json`
- `cargo_full run --bin xtask -- release-completion-audit --output release/evidence/final/completion-audit.json`
- `cargo_full run --bin xtask -- vyre-release-gate`
- `scripts/apply-branch-protection.sh main`

Release-note wording contract:

- Release notes must name `vyre 0.6.1`.
- Release notes must name `weir 0.1.0`.
- Release notes must preserve the required package story: `vyre`, `vyrec`, and `vyre-frontend-c` ship on the `0.6.1` Vyre train, while `weir` ships as `0.1.0`.
- Release notes must name `vyre-v0.6.1-rc.1`.
- Release notes must name `weir-v0.1.0-rc.1`.
- Release notes must name `vyre-0.6.1-weir-0.1.0-rc.1`.
- Release notes must name `vyre-v0.6.1`.
- Release notes must name `weir-v0.1.0`.
- Release notes must name `vyre-0.6.1-weir-0.1.0`.
- Release notes must not instruct maintainers to create or push a bare `v0.6.1` tag for this release train.
- The version matrix scans release-note documents, the root release plan, Weir README, and `tools/vyrec` README for ambiguous bare tag commands.
