# Testing `vyre-test-support`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-test-support
```

Provide shared deterministic fixtures and assertions for workspace tests.

The crate lives at `vyre-test-support` and owns the `test-support` seam in the `test-tooling` layer.

## Commands

```console
./cargo_full test -p vyre-test-support
```

```console
./cargo_full test -p vyre-test-support --all-features
```

```console
./cargo_full test -p vyre-test-support --features ir-fixtures,parity-oracles
```

## Feature sets

- Default feature members: None
- Available manifest features: `default`, `driver-artifact-contracts`, `driver-contracts`, `golden-corpus`, `host-input-abi`, `ir-fixtures`, `parity-oracles`, `semantic-parity`, `semantic-requests`, `spec-strategies`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `lib` | `vyre_test_support` | `vyre-test-support/src/lib.rs` | None | `./cargo_full test -p vyre-test-support` |
| `test` | `all_tests` | `vyre-test-support/tests/all_tests.rs` | None | `./cargo_full test -p vyre-test-support --test all_tests` |
| `test` | `all_tests_ir_fixtures` | `vyre-test-support/tests/all_tests_ir_fixtures.rs` | `ir-fixtures`, `parity-oracles` | `./cargo_full test -p vyre-test-support --test all_tests_ir_fixtures --features ir-fixtures,parity-oracles` |
| `test` | `workspace_root_follows_the_working_directory` | `vyre-test-support/tests/workspace_root_follows_the_working_directory.rs` | None | `./cargo_full test -p vyre-test-support --test workspace_root_follows_the_working_directory` |

## Test classes

- Fixture determinism
- Typed output boundaries and numerical comparison
- Failure-preserving bounded replay minimization
- Mutation detection with valid positive controls, exact diagnostic identities, output differences, and executed race findings
- Failure propagation and diagnostic contracts

## Hardware requirements

No accelerator is required for the default suite.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

An invalid or incorrect positive control, an unchanged mutant, an unrelated validation error, incomplete race exploration, or an execution error fails the mutation proof. Mutation fixtures do not certify every optimizer branch or the full device interleaving space.
