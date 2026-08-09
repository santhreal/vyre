# Testing `vyre-frontend-c`

Run the default crate suite from the workspace root:

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-frontend-c
```

Parse C input and lower supported language constructs into typed Vyre programs.

The crate lives at `vyre-frontend-c`. The `c-frontend` owner maintains its
`frontend` testing contract.

## Commands

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-frontend-c
```

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-frontend-c --all-features
```

## Feature sets

This crate declares no Cargo features.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `lib` | `vyre_frontend_c` | `vyre-frontend-c/src/lib.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-frontend-c` |
| `test` | `source_to_ir_contract` | `vyre-frontend-c/tests/source_to_ir_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-frontend-c --test source_to_ir_contract` |

## Test classes

- Parser and lowering behavior
- Source diagnostics and unsupported-language rejection
- Generated program execution parity

## Hardware requirements

Parser and lowering tests are host-capable. End-to-end backend execution requires the explicitly selected device and fails if that backend cannot initialize.

## Evidence outputs

- `release/evidence/parser/`
- Command status, exact diagnostics, and generated-program parity assertions

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
