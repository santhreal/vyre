# Testing `vyre-libs-device`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-libs-device
```

Own compiler-internal device boundary contracts, memory ownership models, and resident graph layout compositions. Does not own concrete backend implementations or driver dispatch.

The crate lives at `vyre-libs-device`. The `product-libraries` owner maintains its
`libraries` testing contract.

## Commands

```console
./cargo_full test -p vyre-libs-device
```

```console
./cargo_full test -p vyre-libs-device --all-features
```

## Feature sets

- Default feature members: `device`
- Available manifest features: `default`, `device`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `lib` | `vyre_libs_device` | `vyre-libs-device/src/lib.rs` | None | `./cargo_full test -p vyre-libs-device` |
| `test` | `all_tests` | `vyre-libs-device/tests/all_tests.rs` | None | `./cargo_full test -p vyre-libs-device --test all_tests` |

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
