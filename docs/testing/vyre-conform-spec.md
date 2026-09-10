# Testing `vyre-conform-spec`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-conform-spec
```

Define conformance case, result, and certificate schemas against the public facade.

The crate lives at `conform/vyre-conform-spec` and owns the `conformance-schema` seam in the `conformance` layer.

## Commands

```console
./cargo_full test -p vyre-conform-spec
```

## Feature sets

This crate declares no Cargo features.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `lib` | `vyre_conform_spec` | `conform/vyre-conform-spec/src/lib.rs` | None | `./cargo_full test -p vyre-conform-spec` |
| `test` | `all_tests` | `conform/vyre-conform-spec/tests/all_tests.rs` | None | `./cargo_full test -p vyre-conform-spec --test all_tests` |

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
