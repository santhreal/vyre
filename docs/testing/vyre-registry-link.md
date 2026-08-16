# Testing `vyre-registry-link`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-registry-link
```

Own every inventory registry link anchor, report which sources a build links, and assert that each linked source reached the registry it submits into.

The crate lives at `vyre-registry-link`. The `registry-link` owner maintains its
`registry-link` testing contract.

## Commands

```console
./cargo_full test -p vyre-registry-link
```

```console
./cargo_full test -p vyre-registry-link --all-features
```

## Feature sets

- Default feature members: `operations`, `cuda`, `metal`, `reference`, `spirv`, `wgpu`
- Available manifest features: `cuda`, `default`, `metal`, `operations`, `reference`, `spirv`, `wgpu`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `lib` | `vyre_registry_link` | `vyre-registry-link/src/lib.rs` | None | `./cargo_full test -p vyre-registry-link` |
| `test` | `registry_link_rules` | `vyre-registry-link/tests/registry_link_rules.rs` | None | `./cargo_full test -p vyre-registry-link --test registry_link_rules` |
| `test` | `registry_link_rules` | `vyre-registry-link/tests/registry_link_rules.rs` | `operations` | `./cargo_full test -p vyre-registry-link --test registry_link_rules` |

## Test classes

- Link-anchor and registration-source contracts
- Per-source floors derived from the tree
- Registry closure against a partial link

## Hardware requirements

No accelerator is required for the default suite.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
