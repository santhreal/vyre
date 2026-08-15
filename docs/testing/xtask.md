# Testing `xtask`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p xtask
```

Own the subcommand registry and every gate that judges the tree from source text, manifests, workflows, and recorded evidence, linking no vyre crate.

The crate lives at `xtask`. The `release-tooling` owner maintains its
`tooling` testing contract.

## Commands

```console
./cargo_full test -p xtask
```

```console
./cargo_full run --bin xtask -- vyre-release-gate
```

## Feature sets

This crate declares no Cargo features.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `bin` | `audit_rule_contracts` | `xtask/src/bin/audit_rule_contracts.rs` | None | `./cargo_full test -p xtask --bin audit_rule_contracts` |
| `bin` | `scaffold_rule` | `xtask/src/bin/scaffold_rule.rs` | None | `./cargo_full test -p xtask --bin scaffold_rule` |
| `bin` | `xtask` | `xtask/src/main.rs` | None | `./cargo_full test -p xtask --bin xtask` |
| `lib` | `xtask` | `xtask/src/lib.rs` | None | `./cargo_full test -p xtask` |
| `test` | `docs_references` | `xtask/tests/docs_references.rs` | None | `./cargo_full test -p xtask --test docs_references` |
| `test` | `release_docs` | `xtask/tests/release_docs.rs` | None | `./cargo_full test -p xtask --test release_docs` |

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
