# Testing `vyre-libs-vfs`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-libs-vfs
```

Own virtual filesystem DMA asynchronous block load and asset resolution compositions. Does not own host file IO or runtime storage drivers.

The crate lives at `vyre-libs-vfs` and owns the `libs-vfs` seam in the `libraries` layer.

## Commands

```console
./cargo_full test -p vyre-libs-vfs
```

```console
./cargo_full test -p vyre-libs-vfs --all-features
```

## Feature sets

- Default feature members: `vfs`
- Available manifest features: `default`, `vfs`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `lib` | `vyre_libs_vfs` | `vyre-libs-vfs/src/lib.rs` | None | `./cargo_full test -p vyre-libs-vfs` |
| `test` | `all_tests` | `vyre-libs-vfs/tests/all_tests.rs` | None | `./cargo_full test -p vyre-libs-vfs --test all_tests` |

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
