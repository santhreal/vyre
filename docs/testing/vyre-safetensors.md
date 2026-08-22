# Testing `vyre-safetensors`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-safetensors
```

Validate safetensors metadata, shard indexes, compiler requirements, trusted shard digests, and immutable checkpoint identities without owning runtime residency.

The crate lives at `vyre-safetensors`. The `safetensors-adapter` owner maintains its
`runtime` testing contract.

## Commands

```console
./cargo_full test -p vyre-safetensors
```

## Feature sets

This crate declares no Cargo features.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `lib` | `vyre_safetensors` | `vyre-safetensors/src/lib.rs` | None | `./cargo_full test -p vyre-safetensors` |
| `test` | `ingestion` | `vyre-safetensors/tests/ingestion.rs` | None | `./cargo_full test -p vyre-safetensors --test ingestion` |

## Test classes

- Execution planning and cache contracts
- Persistent runtime state transitions
- IO, telemetry, and failure semantics

## Hardware requirements

No accelerator is required for the default suite.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

Malformed framing, oversized metadata, invalid tensor ranges, unsafe shard paths, incomplete mappings, requirement drift, or digest mismatch returns a typed SafetensorError before runtime admission.
