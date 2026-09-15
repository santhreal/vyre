# Testing `vyre-registry-link`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-registry-link
```

Own every inventory registry link anchor, report which sources a build links, and assert that each linked source reached the registry it submits into.

The crate lives at `vyre-registry-link` and owns the `registry-link` seam in the `registry-link` layer.

## Commands

```console
./cargo_full test -p vyre-registry-link
```

```console
./cargo_full test -p vyre-registry-link --all-features
```

## Feature sets

- Default feature members: `operations`, `cuda`, `metal`, `spirv`, `wgpu`
- Available manifest features: `cuda`, `default`, `device-tests`, `metal`, `operations`, `spirv`, `wgpu`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `lib` | `vyre_registry_link` | `vyre-registry-link/src/lib.rs` | None | `./cargo_full test -p vyre-registry-link` |
| `test` | `all_tests` | `vyre-registry-link/tests/all_tests.rs` | None | `./cargo_full test -p vyre-registry-link --test all_tests` |
| `test` | `all_tests_device` | `vyre-registry-link/tests/all_tests_device.rs` | `device-tests` | `./cargo_full test -p vyre-registry-link --test all_tests_device` |
| `test` | `all_tests_operations` | `vyre-registry-link/tests/all_tests_operations.rs` | `operations` | `./cargo_full test -p vyre-registry-link --test all_tests_operations` |

## Test classes

- Link-anchor and registration-source contracts
- Per-source floors derived from the tree
- Registry closure against a partial link
- Numeric-mode decisions closed over the registered backends

## Hardware requirements

No accelerator is required for the default suite.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
