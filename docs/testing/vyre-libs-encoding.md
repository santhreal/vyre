# Testing `vyre-libs-encoding`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-libs-encoding
```

Own compiler-internal bitset, provenance, matroid, and fingerprint encoding compositions. Does not own consumer decoding algorithms, backend lowering, or runtime execution.

The crate lives at `vyre-libs-encoding`. The `product-libraries` owner maintains its
`libraries` testing contract.

## Commands

```console
./cargo_full test -p vyre-libs-encoding
```

```console
./cargo_full test -p vyre-libs-encoding --all-features
```

## Feature sets

- Default feature members: `encoding`
- Available manifest features: `default`, `encoding`, `nn-paging`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `lib` | `vyre_libs_encoding` | `vyre-libs-encoding/src/lib.rs` | None | `./cargo_full test -p vyre-libs-encoding` |
| `test` | `all_tests` | `vyre-libs-encoding/tests/all_tests.rs` | None | `./cargo_full test -p vyre-libs-encoding --test all_tests` |

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
