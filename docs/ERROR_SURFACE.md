# Error Surface Contract

Last verified: 2026-08-04

Applies to Vyre 0.7.2.

Vyre publishes stable diagnostic codes in
[`docs/error-codes.md`](error-codes.md). Use the code as the machine-readable
key. Treat the accompanying prose as the operator-facing explanation.

## Stable code families

| Family | Surface | Typical owner |
|---|---|---|
| `V###` | `Program` validation | `vyre-foundation` |
| `E-*` | General IR and execution diagnostics | foundation / shared |
| `W-*` | Warnings and deprecations | shared |
| `B-*` | Backend capability and dispatch failures | `vyre-driver*` |
| `P-*` | Runtime pipeline failures | `vyre-runtime` |
| `C-*` | Conformance failures | `conform/*` |
| megakernel `DiagnosticCode` | Artifact compile / envelope auth | `vyre-megakernel` |

The registry reserves retired or unused values instead of reassigning them.
Adding a code requires a matching registry row with a description and a
concrete `Fix:` action.

## Operator workflow

1. Capture the full diagnostic string, including the stable code and `Fix:`.
2. Match on the code in automation. Do not parse prose to identify the error.
3. Apply the fix action. Prefer the smallest change that restores a valid
   program, backend request, or artifact.
4. Re-run the same command. A different code means a different boundary failed.

### Examples

| Code | Meaning | First fix |
| --- | --- | --- |
| `V052` | Call references an undeclared buffer | Add the buffer to `Program::buffers` |
| Backend unavailable (`B-*`) | Requested device/path missing | Fix driver/device config; do not expect silent CPU fallback |
| Envelope admission failure | Cache/package bytes are not an authentic target payload | Rebuild AOT package or discard corrupt cache entry |
| Conformance mismatch (`C-*`) | Backend disagreed with reference oracle | Inspect op matrix support and fixture bounds |

Search the source for the code string when you need the exact emission site.

## Layer-specific rules

- **Validation** fails before lowering. Invalid programs never reach a backend.
- **Backend** failures name the selected backend. They do not rewrite the request
  to another backend.
- **Runtime** protocol and admission failures are fail closed. Malformed slots,
  tenants, and envelopes reject before publication or dispatch.
- **Artifact compile** (`vyre-megakernel`) rejects invalid graphs, incomplete
  external facts, invalid search bounds, resource overflow, and envelope
  authentication failures before bytes ship.

## Source ownership

Each crate owns its error variants. The shared public code registry remains
[`docs/error-codes.md`](error-codes.md). Backend errors also expose the stable
numeric identifiers listed in that registry. Megakernel compile diagnostics use
`vyre_megakernel::DiagnosticCode` and still carry actionable `Fix:` text.

## Verification

- `scripts/check_error_codes_cataloged.sh` rejects emitted codes missing from
  the registry.
- `scripts/check_expect_has_fix.sh` rejects expectation messages without an
  actionable fix.
- Validation, backend, runtime admission, and megakernel artifact suites assert
  exact codes on representative failure paths.
