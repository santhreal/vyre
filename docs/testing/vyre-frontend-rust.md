# Testing `vyre-frontend-rust`

Run the default crate suite from the workspace root:

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-frontend-rust
```

Lower the supported Rust frontend subset into typed Vyre programs and execute it through selected backends.

The crate lives at `vyre-frontend-rust`. The `rust-frontend` owner maintains its
`frontend` testing contract.

## Commands

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-frontend-rust
```

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-frontend-rust --all-features
```

## Feature sets

This crate declares no Cargo features.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `lib` | `vyre_frontend_rust` | `vyre-frontend-rust/src/lib.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-frontend-rust` |
| `test` | `adversarial_parse_depth` | `vyre-frontend-rust/tests/adversarial_parse_depth.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-frontend-rust --test adversarial_parse_depth` |
| `test` | `borrow` | `vyre-frontend-rust/tests/borrow.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-frontend-rust --test borrow` |
| `test` | `borrowck_engine` | `vyre-frontend-rust/tests/borrowck_engine.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-frontend-rust --test borrowck_engine` |
| `test` | `conflict` | `vyre-frontend-rust/tests/conflict.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-frontend-rust --test conflict` |
| `test` | `differential_fuzz` | `vyre-frontend-rust/tests/differential_fuzz.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-frontend-rust --test differential_fuzz` |
| `test` | `escape` | `vyre-frontend-rust/tests/escape.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-frontend-rust --test escape` |
| `test` | `lexer_ir_reference` | `vyre-frontend-rust/tests/lexer_ir_reference.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-frontend-rust --test lexer_ir_reference` |
| `test` | `lexer_oracle` | `vyre-frontend-rust/tests/lexer_oracle.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-frontend-rust --test lexer_oracle` |
| `test` | `lower_exec` | `vyre-frontend-rust/tests/lower_exec.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-frontend-rust --test lower_exec` |
| `test` | `proptest_robustness` | `vyre-frontend-rust/tests/proptest_robustness.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-frontend-rust --test proptest_robustness` |
| `test` | `resolve` | `vyre-frontend-rust/tests/resolve.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-frontend-rust --test resolve` |
| `test` | `rust_borrow_fact_delta_registry` | `vyre-frontend-rust/tests/rust_borrow_fact_delta_registry.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-frontend-rust --test rust_borrow_fact_delta_registry` |
| `test` | `rustc_differential` | `vyre-frontend-rust/tests/rustc_differential.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-frontend-rust --test rustc_differential` |
| `test` | `smoke` | `vyre-frontend-rust/tests/smoke.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-frontend-rust --test smoke` |
| `test` | `source_to_ir_contract` | `vyre-frontend-rust/tests/source_to_ir_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-frontend-rust --test source_to_ir_contract` |
| `test` | `typeck` | `vyre-frontend-rust/tests/typeck.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-frontend-rust --test typeck` |

## Test classes

- Parser and lowering behavior
- Source diagnostics and unsupported-language rejection
- Generated program execution parity

## Hardware requirements

Host parser and lowering tests require no accelerator. Feature-selected backend execution requires its physical device and reports acquisition failure.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
