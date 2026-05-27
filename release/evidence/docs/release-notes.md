# Release notes evidence

This artifact backs human-facing release note readiness.

Evidence sources:

Required generated evidence:

- `release/evidence/version/version-matrix.json`
- `release/evidence/version/release-tag-plan.json`
- `release/evidence/final/completion-audit.json`

Release contract:

- Release notes must use `vyre 0.4.2` and `weir 0.1.0`.
- Release notes must state that `vyre`, `vyre-driver-cuda@0.4.2`, `vyre-driver-wgpu@0.4.2`, `vyrec`, and `vyre-frontend-c` are present on the `0.4.2` Vyre release train, and `weir` is present at `0.1.0`; `missing_required_release_packages` in `version-matrix.json` must be empty before notes are cut.
- Workspace-inherited manifest versions must resolve to the concrete release versions in `version-matrix.json`; an inherited version that cannot be resolved is treated as release drift.
- Release notes must reference RC tags `vyre-v0.4.2-rc.1`, `weir-v0.1.0-rc.1`, and `vyre-0.4.2-weir-0.1.0-rc.1` before final tags `vyre-v0.4.2`, `weir-v0.1.0`, and `vyre-0.4.2-weir-0.1.0`.
- Release notes must not instruct a bare `v0.4.2` tag workflow.
- Release-facing docs must not contain unapproved deferral or capability-disclaimer language.
- Release notes are cut only after the completion audit, release gate, and `scripts/apply-branch-protection.sh main` pass.
