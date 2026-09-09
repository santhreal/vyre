# Testing `vyre-libs-security`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-libs-security
```

Own security taint analysis compositions, predicate evaluators, and label resolver IR builders. Does not own rule engine AST evaluation, backend lowering, or runtime execution.

The crate lives at `vyre-libs-security`. The `product-libraries` owner maintains its
`libraries` testing contract.

## Commands

```console
./cargo_full test -p vyre-libs-security
```

```console
./cargo_full test -p vyre-libs-security --all-features
```

## Feature sets

- Default feature members: `security`
- Available manifest features: `default`, `label`, `predicate`, `security`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `lib` | `vyre_libs_security` | `vyre-libs-security/src/lib.rs` | None | `./cargo_full test -p vyre-libs-security` |
| `test` | `all_tests` | `vyre-libs-security/tests/all_tests.rs` | None | `./cargo_full test -p vyre-libs-security --test all_tests` |

## Test classes

- Product-library exact behavior
- Primitive-to-library composition
- Reference and backend parity

## Hardware requirements

No accelerator is required for the default suite.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
