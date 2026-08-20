# Release notes evidence

This artifact backs human-facing release note readiness.

Evidence sources:

Required generated evidence:

- `release/evidence/version/version-matrix.json`
- `release/evidence/version/release-tag-plan.json`
- `release/evidence/final/public-launch-state.json`

Release contract:

- Release notes must use `Vyre 0.8.0`.
- Release notes must state that `vyre`, `vyre-driver-cuda@0.8.0`, and `vyre-driver-wgpu@0.8.0` are present on the `0.8.0` Vyre release train; `missing_required_release_packages` in `version-matrix.json` must be empty before notes are cut.
- Workspace-inherited manifest versions must resolve to the concrete release versions in `version-matrix.json`; an inherited version that cannot be resolved is treated as release drift.
- Release notes must reference RC tag `vyre-v0.8.0-rc.1` before final tag `vyre-v0.8.0`.
- Release notes must not instruct a bare `v0.8.0` tag workflow.
- Release-facing docs must not contain unapproved deferral or capability-disclaimer language.
- Release notes are cut only after the prepublication release gate and `scripts/apply-branch-protection.sh main` pass.
