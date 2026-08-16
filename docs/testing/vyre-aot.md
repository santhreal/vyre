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
| `test` | `artifact_contracts` | `vyre-aot/tests/artifact_contracts.rs` | None | `./cargo_full test -p vyre-aot --test artifact_contracts` |
| `test` | `bundle_contracts` | `vyre-aot/tests/bundle_contracts.rs` | None | `./cargo_full test -p vyre-aot --test bundle_contracts` |
| `test` | `cache_contracts` | `vyre-aot/tests/cache_contracts.rs` | None | `./cargo_full test -p vyre-aot --test cache_contracts` |
| `test` | `canonical_package` | `vyre-aot/tests/canonical_package.rs` | None | `./cargo_full test -p vyre-aot --test canonical_package` |
| `test` | `compile_smoke` | `vyre-aot/tests/compile_smoke.rs` | None | `./cargo_full test -p vyre-aot --test compile_smoke` |
| `test` | `generated_artifact_manifest_matrix` | `vyre-aot/tests/generated_artifact_manifest_matrix.rs` | None | `./cargo_full test -p vyre-aot --test generated_artifact_manifest_matrix` |
| `test` | `generated_loader_contracts` | `vyre-aot/tests/generated_loader_contracts.rs` | None | `./cargo_full test -p vyre-aot --test generated_loader_contracts` |
| `test` | `launcher_contracts` | `vyre-aot/tests/launcher_contracts.rs` | None | `./cargo_full test -p vyre-aot --test launcher_contracts` |
| `test` | `manifest_round_trip` | `vyre-aot/tests/manifest_round_trip.rs` | None | `./cargo_full test -p vyre-aot --test manifest_round_trip` |

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
