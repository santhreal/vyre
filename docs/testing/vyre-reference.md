# Testing `vyre-reference`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-reference
```

The only crate permitted to compute on the CPU: the pure-Rust IR oracle. Not a backend and not a fallback.

The crate lives at `vyre-reference` and owns the `reference-semantics` seam in the `semantics` layer.

## Commands

```console
./cargo_full test -p vyre-reference
```

```console
./cargo_full test -p vyre-reference --all-features
```

## Feature sets

- Default feature members: `subgroup-ops`
- Available manifest features: `default`, `subgroup-ops`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `example` | `vyre_reference_release_surface` | `vyre-reference/examples/vyre_reference_release_surface.rs` | None | `./cargo_full test -p vyre-reference --example vyre_reference_release_surface` |
| `lib` | `vyre_reference` | `vyre-reference/src/lib.rs` | None | `./cargo_full test -p vyre-reference` |
| `test` | `all_tests` | `vyre-reference/tests/all_tests.rs` | None | `./cargo_full test -p vyre-reference --test all_tests` |

## Test classes

- Reference execution semantics
- Exact oracle and witness contracts
- Adversarial and property parity tests

## Hardware requirements

No accelerator is required for the default suite.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
