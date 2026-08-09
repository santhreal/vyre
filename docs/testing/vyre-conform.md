# Testing `vyre-conform`

Run the default crate suite from the workspace root:

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-conform
```

Execute production artifacts against independent reference semantics, minimize counterexamples, check algebraic laws, and issue versioned certificates and replay records through one library and thin CLI.

The crate lives at `conform/vyre-conform`. The `conformance` owner maintains its
`conformance` testing contract.

## Commands

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-conform
```

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-conform --all-features
```

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-conform --all-features -- --ignored --nocapture
```

## Feature sets

- Default feature members: `gpu`
- Available manifest features: `default`, `gpu`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `bin` | `vyre-conform` | `conform/vyre-conform/src/main.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-conform --bin vyre-conform` |
| `example` | `vyre_conform_runner_release_surface` | `conform/vyre-conform/examples/vyre_conform_runner_release_surface.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-conform --example vyre_conform_runner_release_surface` |
| `lib` | `vyre_conform` | `conform/vyre-conform/src/lib.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-conform` |
| `test` | `_compute_pins` | `conform/vyre-conform/tests/_compute_pins.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-conform --test _compute_pins` |
| `test` | `cert_artifact` | `conform/vyre-conform/tests/cert_artifact.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-conform --test cert_artifact` |
| `test` | `cert_regression_pin` | `conform/vyre-conform/tests/cert_regression_pin.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-conform --test cert_regression_pin` |
| `test` | `composition_discipline` | `conform/vyre-conform/tests/composition_discipline.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-conform --test composition_discipline` |
| `test` | `countless_readwrite_output_parity` | `conform/vyre-conform/tests/countless_readwrite_output_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-conform --test countless_readwrite_output_parity` |
| `test` | `dispatch_grid_contracts` | `conform/vyre-conform/tests/dispatch_grid_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-conform --test dispatch_grid_contracts` |
| `test` | `fp_parity_ul_policy_contracts` | `conform/vyre-conform/tests/fp_parity_ul_policy_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-conform --test fp_parity_ul_policy_contracts` |
| `test` | `gap_cert_artifact` | `conform/vyre-conform/tests/gap_cert_artifact.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-conform --test gap_cert_artifact` |
| `test` | `invariants` | `conform/vyre-conform/tests/invariants.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-conform --test invariants` |
| `test` | `lens_parity` | `conform/vyre-conform/tests/lens_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-conform --test lens_parity` |
| `test` | `minimizer_contract` | `conform/vyre-conform/tests/minimizer_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-conform --test minimizer_contract` |
| `test` | `op_matrix_truth` | `conform/vyre-conform/tests/op_matrix_truth.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-conform --test op_matrix_truth` |
| `test` | `parity_matrix` | `conform/vyre-conform/tests/parity_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-conform --test parity_matrix` |
| `test` | `production_route` | `conform/vyre-conform/tests/production_route.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-conform --test production_route` |
| `test` | `release_gate_contracts` | `conform/vyre-conform/tests/release_gate_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-conform --test release_gate_contracts` |
| `test` | `schema_compatibility` | `conform/vyre-conform/tests/schema_compatibility.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-conform --test schema_compatibility` |
| `test` | `ulp_audit` | `conform/vyre-conform/tests/ulp_audit.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-conform --test ulp_audit` |

## Test classes

- Case and certificate schema contracts
- Generator determinism
- Cross-backend semantic conformance

## Hardware requirements

Cross-backend certificates require every selected physical backend. Missing selected hardware is a failed conformance run.

## Evidence outputs

- `release/evidence/conformance/release-all-backends-certificate.json`
- Command status and exact cross-backend results

## Skips and failures

The default command omits tests marked `#[ignore]`. The ignored command is the explicit physical-backend run and cannot silently skip a selected backend.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
