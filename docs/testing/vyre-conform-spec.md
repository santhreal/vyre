# Testing `vyre-conform-spec`

Run the default crate suite from the workspace root:

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-conform-spec
```

Define conformance case, result, and certificate schemas against the public facade.

The crate lives at `conform/vyre-conform-spec`. The `conformance` owner maintains its
`conformance` testing contract.

## Commands

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-conform-spec
```

## Feature sets

This crate declares no Cargo features.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `lib` | `vyre_conform_spec` | `conform/vyre-conform-spec/src/lib.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-conform-spec` |
| `test` | `schema_contract` | `conform/vyre-conform-spec/tests/schema_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-conform-spec --test schema_contract` |
| `test` | `witness_contract` | `conform/vyre-conform-spec/tests/witness_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-conform-spec --test witness_contract` |

## Test classes

- Case and certificate schema contracts
- Generator determinism
- Cross-backend semantic conformance

## Hardware requirements

No accelerator is required for the default suite.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
