# Testing `structure-gate`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p structure-gate
```

Enforce the crate roster, one operation identity per semantic operation, and one home per concept. Depends on no vyre crate so it keeps running while the workspace does not compile.

The crate lives at `structure-gate`. The `release-tooling` owner maintains its
`standalone-tooling` testing contract.

## Commands

```console
./cargo_full test -p structure-gate
```

## Feature sets

This crate declares no Cargo features.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `bin` | `structure-gate` | `structure-gate/src/main.rs` | None | `./cargo_full test -p structure-gate --bin structure-gate` |
| `lib` | `structure_gate` | `structure-gate/src/lib.rs` | None | `./cargo_full test -p structure-gate` |
| `test` | `checkout_provenance` | `structure-gate/tests/checkout_provenance.rs` | None | `./cargo_full test -p structure-gate --test checkout_provenance` |
| `test` | `crate_structure_contracts` | `structure-gate/tests/crate_structure_contracts.rs` | None | `./cargo_full test -p structure-gate --test crate_structure_contracts` |
| `test` | `device_only_routing` | `structure-gate/tests/device_only_routing.rs` | None | `./cargo_full test -p structure-gate --test device_only_routing` |
| `test` | `materializer_admission` | `structure-gate/tests/materializer_admission.rs` | None | `./cargo_full test -p structure-gate --test materializer_admission` |
| `test` | `node_child_descent_owner` | `structure-gate/tests/node_child_descent_owner.rs` | None | `./cargo_full test -p structure-gate --test node_child_descent_owner` |

## Test classes

- Repository contract scans
- Checkout resolution and path boundaries
- Failure diagnostics that name the correction

## Hardware requirements

No accelerator is required for the default suite.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
