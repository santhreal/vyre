# Diagnostics

```text
V009  memory  Atomic on non-writable buffer
      Fix: Declare the buffer with `BufferAccess::ReadWrite`.
```

Every diagnostic carries a stable code, the invariant it enforces, and a
corrective action. A diagnostic without a corrective action is a defect in
the rule, not a style question.

## Validation codes

`docs/generated/error-codes.toml` is the whole validation rule set: 98
rules at schema version 1, rendered from
`vyre-foundation/src/validate/catalog.rs`. Each row carries `code`,
`phase`, `invariant` and `corrective_action`.

The eight phases and how many rules each owns:

| Phase | Rules |
|---|---|
| type | 24 |
| node | 21 |
| memory | 20 |
| expression | 18 |
| program | 9 |
| limits | 3 |
| composition | 2 |
| capability | 1 |

The catalog in source is the authority and the file is rendered from it.
`vyre-foundation/tests/validator_error_docs.rs` fails on divergence and
reports one finding per divergent code, so the two cannot disagree.

## Backend errors

`docs/generated/driver-error-codes.toml` carries the nine `BackendError`
variants with their stable numeric ids:

| Id | Variant |
|---|---|
| 1001 | `DeviceOutOfMemory` |
| 1002 | `UnsupportedFeature` |
| 1003 | `PoisonedLock` |
| 1004 | `KernelCompileFailed` |
| 1005 | `DispatchFailed` |
| 1006 | `InvalidProgram` |
| 1007 | `CooperativeResidencyExceeded` |
| 1008 | `DeviceLost` |
| 1999 | `Unknown` |

`1999` is the terminal id, not a bucket to grow: a new failure mode gets
its own id. `vyre-driver/tests/error_code_catalog.rs` compares the
committed file against `vyre-driver/src/backend/error_catalog.rs`.

The same file carries the compile-time diagnostic set.
`W-OP-DEPRECATED` fires when a resolved operation is marked deprecated in
the migration registry, and carries the operation location plus the
migration note as its suggested fix.

## Reading an error

Error text states the corrective action inline after `Fix:`. Two examples
from live code paths:

```text
backend `cuda` is not linked into this binary. Fix: link the concrete
driver crate that registers this backend or choose one of the registered
backend ids.
```

```text
IR wire-format offset overflow. Fix: provide a valid VIR0 Program blob.
```

An error that names no corrective action is reportable.
