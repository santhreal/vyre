# vyre Error Codes

Applies to Vyre 0.7.2.

This document is the canonical registry of stable error and warning codes
surfaced through the foundation-owned diagnostic protocol. Codes are
append-only within a schema version. A new code must define its stage,
corrective action, and retry class before release.

Every message emitted through [`Diagnostic`](../vyre-foundation/src/lib.rs)
carries one of these codes. Tooling (LSP clients, CI annotators, editor
extensions) keys rules off the code, not the prose; prose is free to drift
across versions as long as the code stays stable.

## Code families

| Family | Severity | Source                                      |
|--------|----------|---------------------------------------------|
| `E-*`  | error    | General errors surfaced via `Diagnostic`    |
| `W-*`  | warning  | Deprecations, soft-failed invariants        |
| `V###` | error    | Program-validation rules (`validate_program`) |
| `B-*`  | error    | Backend dispatch / capability errors        |
| `C-*`  | error    | Conformance verdict failures                |

## Validation codes (`V###`)

Validation emits structured fields: code, validation phase, typed location,
cause, corrective action, and retry class. Human-readable rendering retains
the code and corrective action, but callers key behavior on the structured
fields rather than rendered prefixes.

All validation codes use diagnostic stage `validate` and retry class
`never`.

The registry is [`generated/error-codes.toml`](generated/error-codes.toml),
rendered from the rule table in
[`vyre-foundation/src/validate/catalog.rs`](../vyre-foundation/src/validate/catalog.rs).
A rule's phase, invariant and corrective action come from there, or at run
time from `ValidationCode::phase`, `ValidationCode::invariant` and
`ValidationCode::corrective_action`.

Codes `V024`, `V026`, `V037`-`V040`, `V048`-`V050`, `V062`, `V069`,
`V071`-`V082`, `V113`, `V117`, and codes above `V130` are reserved slots.
Allocate through the rule table before emitting a new diagnostic.

## General errors (`E-*`)

| Code | Description | Fix template |
|------|-------------|--------------|
| `E-IR-001` | Decode of unknown Opaque extension id (wire format) | Link the crate that registers the extension id, then re-decode. |
| `E-IR-002` | Buffer zero-count with non-empty shape payload | Reject the non-canonical Program bytes. |
| `E-IR-003` | Diagnostic catalog carries a code not listed in `docs/error-codes.md` | Add the code to the registry before shipping. |

## Warnings (`W-*`)

| Code | Description | Fix template |
|------|-------------|--------------|
| `W-DEPREC-001` | Deprecated op id in use | Migrate to the replacement op listed in the deprecation registry. |

## Backend codes (`B-*`)

| Code | Description | Fix template |
|------|-------------|--------------|
| `B-CAP-001` | Backend does not support this op's capability class | Pick a backend that supports this op's capabilities, or use a different op. |
| `B-CAP-002` | Backend factory refused to construct (no GPU adapter, missing driver) | Fix the adapter issue per the error's `Fix:` prose, or skip this backend. |
| `B-CAP-003` | Unsupported feature, for example a dispatch request on an emission-only target | Use a backend whose `supports_dispatch` returns true. |

## Backend ErrorCode stable ids

| Variant | code | description |
|---------|------|-------------|
| `DeviceOutOfMemory` | 1001 | Backend device reported insufficient memory during allocation, staging, or dispatch. |
| `UnsupportedFeature` | 1002 | The selected backend lacks a feature required by the program or dispatch policy. |
| `PoisonedLock` | 1003 | A synchronization lock was poisoned after a panic while held. |
| `KernelCompileFailed` | 1004 | Kernel source compilation or validation failed for WGSL, SPIR-V, PTX, Metal IR, or another backend source format. |
| `DispatchFailed` | 1005 | Queue submission, command execution, readback, or dispatch completion failed. |
| `InvalidProgram` | 1006 | The submitted program violates backend constraints or the portable program contract. |
| `Unknown` | 1999 | Legacy or unclassified backend failure produced without a more specific machine-readable code. |

## Pipeline codes (`P-*`, `vyre-runtime::PipelineError`)

| Code | Variant | Description | Fix template |
|------|---------|-------------|--------------|
| `P-URING-001` | `IoUringSyscall { syscall, errno, fix }` | A raw `io_uring_setup` / `mmap` / `io_uring_enter` / `io_uring_register` syscall returned an errno. | Per-variant `fix:` string names the remediation; typical causes are kernel too old, missing CAP_SYS_ADMIN for SQPOLL on <5.13, or exhausted `max_map_count`. |
| `P-URING-002` | `QueueFull { queue, fix }` | The submission or completion queue rejected a request because it is full, out of bounds, or a slot is still in flight. | Drain completions with `AsyncUringStream::poll` or `UringCompletionPump::poll`, then retry. For backpressure-triggered rejections on `publish_slot`, wait for the kernel to advance `control[DONE_COUNT]`. |
| `P-URING-003` | `NotLinux` | `io_uring` or `futex_waitv` was requested on a non-Linux host. | Run on Linux 5.16+ or use artifact submission without an io_uring completion pump. |
| `P-URING-004` | `NvmePassthroughDisabled` | `submit_nvme_passthrough` was called without the `uring-cmd-nvme` feature. | Add `features = ["uring-cmd-nvme"]` to `vyre-runtime` in your `Cargo.toml`; requires Linux 6.0+. |
| `P-BACKEND-001` | `Backend(msg)` | A backend error bubbled up from `Megakernel::bootstrap` or `Megakernel::dispatch`. | Inspect the wrapped message; usually a validation error on the IR or an OOM during pipeline creation. |

## Conformance codes (`C-*`)

| Code | Description | Fix template |
|------|-------------|--------------|
| `C-LAW-001` | Backend output disagreed with reference on witnessed input | Fix the backend lowering or rewrite the op to honor the declared `AlgebraicLaw`. |
| `C-DET-001` | Backend produced non-deterministic output across seeds | Remove the non-deterministic code path; conform bans silent nondeterminism. |

## Adding a new code

1. Pick the next unused integer in the appropriate family.
2. Add a row to this document with the code, description, and `Fix:` template.
3. Emit the code via `Diagnostic` with the matching `code` field.
4. CI verifies every code emitted in source appears in this registry
   (see `scripts/check_error_codes_cataloged.sh`).

## Policy

- **Append-only.** Never reuse a retired code. Retiring a code leaves a
  row behind with a `Retired: v<X.Y.Z>` note.
- **Code is the stable key.** Prose may drift.
- **`Fix:` is mandatory.** Every variant carries actionable remediation.
- **No stringly-typed errors.** Every error-path surfaces a structured
  code; the prose is a formatting detail.


## Uncataloged Legacy / Auto-migrated Codes
| Code | Description | Fix template |
|------|-------------|--------------|
| `E-CSR` | Migrated code | See diagnostic output. |
| `E-DATAFLOW` | Migrated code | See diagnostic output. |
| `E-DECODE` | Migrated code | See diagnostic output. |
| `E-DECODE-CONFIG` | Migrated code | See diagnostic output. |
| `E-DECOMPRESS` | Migrated code | See diagnostic output. |
| `E-DFA` | Migrated code | See diagnostic output. |
| `E-GPU` | Migrated code | See diagnostic output. |
| `E-INLINE-ARG-COUNT` | Migrated code | See diagnostic output. |
| `E-INLINE-CYCLE` | Migrated code | See diagnostic output. |
| `E-INLINE-NON-INLINABLE` | Migrated code | See diagnostic output. |
| `E-INLINE-NO-OUTPUT` | Migrated code | See diagnostic output. |
| `E-INLINE-OUTPUT-COUNT` | Migrated code | See diagnostic output. |
| `E-INLINE-UNKNOWN-OP` | Migrated code | See diagnostic output. |
| `E-INTERP` | Migrated code | See diagnostic output. |
| `E-LOWERING` | Migrated code | See diagnostic output. |
| `E-PREFIX` | Migrated code | See diagnostic output. |
| `E-RULE-EVAL` | Migrated code | See diagnostic output. |
| `E-SERIALIZATION` | Migrated code | See diagnostic output. |
| `E-TEST` | Migrated code | See diagnostic output. |
| `E-TOML-PARSE` | Migrated code | See diagnostic output. |
| `E-UNKNOWN` | Migrated code | See diagnostic output. |
| `E-WIRE-UNKNOWN-DIALECT` | Migrated code | See diagnostic output. |
| `E-WIRE-UNKNOWN-OP` | Migrated code | See diagnostic output. |
| `E-WIRE-VALIDATION` | Migrated code | See diagnostic output. |
| `E-WIRE-VERSION` | Migrated code | See diagnostic output. |
| `E-X` | Migrated code | See diagnostic output. |
| `W-DEPRECATED` | Migrated code | See diagnostic output. |
| `W-OP-DEPRECATED` | Migrated code | See diagnostic output. |
| `W-TOML-BAD-OP-ID` | Migrated code | See diagnostic output. |
| `W-TOML-UNREADABLE` | Migrated code | See diagnostic output. |
| `E-TOML-BAD-CATEGORY` | Migrated code | See diagnostic output. |
| `E-TOML-BAD-OP-ID` | Migrated code | See diagnostic output. |
| `E-TOML-DIALECT-DIR-ENTRY` | Migrated code | See diagnostic output. |
| `E-TOML-DIALECT-DIR-MISSING` | Migrated code | See diagnostic output. |
| `E-TOML-DIALECT-DIR-UNREADABLE` | Migrated code | See diagnostic output. |
| `E-TOML-DUPLICATE-OP` | Migrated code | See diagnostic output. |
| `E-TOML-EMPTY-DIALECT` | Migrated code | See diagnostic output. |
| `E-TOML-EMPTY-DIALECT-PATH` | Migrated code | See diagnostic output. |
| `E-TOML-EMPTY-VERSION` | Migrated code | See diagnostic output. |
| `E-TOML-MANIFEST-REJECTED` | Migrated code | See diagnostic output. |
| `E-TOML-UNREADABLE` | Migrated code | See diagnostic output. |
