# Testing `vyre-libs-hash`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-libs-hash
```

Own hash and checksum compositions including FNV-1a, CRC-32, Adler-32, and BLAKE3 IR builders. Does not own security taint logic, backend lowering, or runtime execution.

The crate lives at `vyre-libs-hash` and owns the `libs-hash` seam in the `libraries` layer.

## Commands

```console
./cargo_full test -p vyre-libs-hash
```

```console
./cargo_full test -p vyre-libs-hash --all-features
```

## Feature sets

- Default feature members: `hash`
- Available manifest features: `crypto-blake3`, `default`, `hash`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `lib` | `vyre_libs_hash` | `vyre-libs-hash/src/lib.rs` | None | `./cargo_full test -p vyre-libs-hash` |
| `test` | `all_tests` | `vyre-libs-hash/tests/all_tests.rs` | None | `./cargo_full test -p vyre-libs-hash --test all_tests` |

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
