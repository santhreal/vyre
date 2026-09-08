# Testing `vyre-megakernel`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-megakernel
```

Construct foundation-owned selected schedules through bounded whole-ProgramGraph search, and own immutable Artifact identity and authenticated TargetPayload construction. Does not own logical semantics, schedule schemas, physical-kernel lowering, admission, execution, or lifecycle policy.

The crate lives at `vyre-megakernel`. The `megakernel-compiler` owner maintains its
`compiler-boundary` testing contract.

## Commands

```console
./cargo_full test -p vyre-megakernel
```

## Feature sets

This crate declares no Cargo features.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `lib` | `vyre_megakernel` | `vyre-megakernel/src/lib.rs` | None | `./cargo_full test -p vyre-megakernel` |
| `test` | `all_tests` | `vyre-megakernel/tests/all_tests.rs` | None | `./cargo_full test -p vyre-megakernel --test all_tests` |

## Test classes

- Megakernel artifact compilation contracts
- Static and persistent plan validation
- Invalid program and boundary rejection

## Hardware requirements

No accelerator is required for the default suite.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
