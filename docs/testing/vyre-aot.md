# Testing `vyre-aot`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-aot
```

Package the same megakernel artifact class ahead of time. Not a second compile path. No workspace crate currently depends on this one.

The crate lives at `vyre-aot`. The `aot-artifacts` owner maintains its
`packaging` testing contract.

## Commands

```console
./cargo_full test -p vyre-aot
```

```console
./cargo_full test -p vyre-aot --all-features
```

## Feature sets

- Default feature members: None
- Available manifest features: `default`, `ptx`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `example` | `vyre_aot_release_surface` | `vyre-aot/examples/vyre_aot_release_surface.rs` | None | `./cargo_full test -p vyre-aot --example vyre_aot_release_surface` |
| `lib` | `vyre_aot` | `vyre-aot/src/lib.rs` | None | `./cargo_full test -p vyre-aot` |
| `test` | `all_tests` | `vyre-aot/tests/all_tests.rs` | None | `./cargo_full test -p vyre-aot --test all_tests` |

## Test classes

- Artifact planning and serialization
- Package compatibility
- Invalid artifact and boundary rejection

## Hardware requirements

No accelerator is required for the default suite.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
