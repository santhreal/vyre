# Testing `vyre-lower`

Run the default crate suite from the workspace root:

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-lower
```

Consume verified semantic programs and own the single backend-neutral lowering boundary and pre-emission transforms.

The crate lives at `vyre-lower`. The `lowering` owner maintains its
`lowering` testing contract.

## Commands

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-lower
```

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-lower --all-features
```

## Feature sets

- Default feature members: None
- Available manifest features: `default`, `test-fixtures`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `lib` | `vyre_lower` | `vyre-lower/src/lib.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-lower` |
| `test` | `analysis_fixture_corpuses` | `vyre-lower/tests/analysis_fixture_corpuses.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-lower --test analysis_fixture_corpuses` |
| `test` | `rewrite_layer_contract` | `vyre-lower/tests/rewrite_layer_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-lower --test rewrite_layer_contract` |
| `test` | `target_capabilities` | `vyre-lower/tests/target_capabilities.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-lower --test target_capabilities` |
| `test` | `verify_result_id_uniqueness` | `vyre-lower/tests/verify_result_id_uniqueness.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-lower --test verify_result_id_uniqueness` |

## Test classes

- Backend-neutral lowering transforms
- Pre-emission invariants
- Invalid IR and target rejection

## Hardware requirements

No accelerator is required for the default suite.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
