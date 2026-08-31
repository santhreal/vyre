# Testing `vyre-test-support`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-test-support
```

Provide shared deterministic fixtures and assertions for workspace tests.

The crate lives at `vyre-test-support`. The `test-support` owner maintains its
`test-tooling` testing contract.

## Commands

```console
./cargo_full test -p vyre-test-support
```

```console
./cargo_full test -p vyre-test-support --all-features
```

## Feature sets

- Default feature members: None
- Available manifest features: `default`, `ir-fixtures`, `semantic-requests`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `lib` | `vyre_test_support` | `vyre-test-support/src/lib.rs` | None | `./cargo_full test -p vyre-test-support` |
| `test` | `adversarial_and_mutation_contracts` | `vyre-test-support/tests/adversarial_and_mutation_contracts.rs` | `ir-fixtures` | `./cargo_full test -p vyre-test-support --test adversarial_and_mutation_contracts` |
| `test` | `binop_parity_tables` | `vyre-test-support/tests/binop_parity_tables.rs` | `ir-fixtures` | `./cargo_full test -p vyre-test-support --test binop_parity_tables` |
| `test` | `cast_parity_tables` | `vyre-test-support/tests/cast_parity_tables.rs` | `ir-fixtures` | `./cargo_full test -p vyre-test-support --test cast_parity_tables` |
| `test` | `differential_matrix_contracts` | `vyre-test-support/tests/differential_matrix_contracts.rs` | `ir-fixtures` | `./cargo_full test -p vyre-test-support --test differential_matrix_contracts` |
| `test` | `expr_variant_coverage` | `vyre-test-support/tests/expr_variant_coverage.rs` | `ir-fixtures` | `./cargo_full test -p vyre-test-support --test expr_variant_coverage` |
| `test` | `extension_variant_coverage` | `vyre-test-support/tests/extension_variant_coverage.rs` | `ir-fixtures` | `./cargo_full test -p vyre-test-support --test extension_variant_coverage` |
| `test` | `memory_order_coverage` | `vyre-test-support/tests/memory_order_coverage.rs` | `ir-fixtures` | `./cargo_full test -p vyre-test-support --test memory_order_coverage` |
| `test` | `registry_nets_fire` | `vyre-test-support/tests/registry_nets_fire.rs` | `ir-fixtures` | `./cargo_full test -p vyre-test-support --test registry_nets_fire` |
| `test` | `workspace_root_follows_the_working_directory` | `vyre-test-support/tests/workspace_root_follows_the_working_directory.rs` | None | `./cargo_full test -p vyre-test-support --test workspace_root_follows_the_working_directory` |

## Test classes

- Fixture determinism
- Harness execution and comparison behavior
- Failure propagation and diagnostic contracts

## Hardware requirements

No accelerator is required for the default suite.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
