# Testing `xtask`

Run the default crate suite from the workspace root:

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p xtask
```

Generate evidence and enforce repository, release, documentation, and architecture contracts.

The crate lives at `xtask`. The `release-tooling` owner maintains its
`tooling` testing contract.

## Commands

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p xtask
```

```console
CARGO_BUILD_JOBS=1 ./cargo_full run --bin xtask -- vyre-release-gate --prepublish
```

## Feature sets

This crate declares no Cargo features.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `bin` | `audit_rule_contracts` | `xtask/src/bin/audit_rule_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p xtask --bin audit_rule_contracts` |
| `bin` | `lint_shape_tests` | `xtask/src/bin/lint_shape_tests.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p xtask --bin lint_shape_tests` |
| `bin` | `public_api_check` | `xtask/src/bin/public_api_check.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p xtask --bin public_api_check` |
| `bin` | `scaffold_rule` | `xtask/src/bin/scaffold_rule.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p xtask --bin scaffold_rule` |
| `bin` | `xtask` | `xtask/src/main.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p xtask --bin xtask` |
| `test` | `architecture_docs` | `xtask/tests/architecture_docs.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p xtask --test architecture_docs` |
| `test` | `canonical_first_workgroup_guard` | `xtask/tests/canonical_first_workgroup_guard.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p xtask --test canonical_first_workgroup_guard` |
| `test` | `cli_docs` | `xtask/tests/cli_docs.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p xtask --test cli_docs` |
| `test` | `crate_ownership_registry` | `xtask/tests/crate_ownership_registry.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p xtask --test crate_ownership_registry` |
| `test` | `crate_readmes` | `xtask/tests/crate_readmes.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p xtask --test crate_readmes` |
| `test` | `docs_references` | `xtask/tests/docs_references.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p xtask --test docs_references` |
| `test` | `operation_schema` | `xtask/tests/operation_schema.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p xtask --test operation_schema` |
| `test` | `public_api_snapshot_inventory` | `xtask/tests/public_api_snapshot_inventory.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p xtask --test public_api_snapshot_inventory` |
| `test` | `relation_import_certificates` | `xtask/tests/relation_import_certificates.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p xtask --test relation_import_certificates` |
| `test` | `release_docs` | `xtask/tests/release_docs.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p xtask --test release_docs` |
| `test` | `root_readme` | `xtask/tests/root_readme.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p xtask --test root_readme` |
| `test` | `testing_guides` | `xtask/tests/testing_guides.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p xtask --test testing_guides` |

## Test classes

- Command and policy behavior
- Evidence schema and regeneration contracts
- Failure diagnostics and repository boundaries

## Hardware requirements

Most policy tests are host-only. Commands that generate backend evidence inherit the hardware contract of the selected backend and must fail on an unavailable requested device.

## Evidence outputs

- `release/evidence/`
- Generated documentation and matrices named by each command
- Command status and exact semantic blocker diagnostics

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
