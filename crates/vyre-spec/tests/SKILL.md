# tests/SKILL.md, vyre-spec

One test file per contract. A file is named for the contract it proves, and
the directory has no catch-all target. `docs/testing/TESTING.toml` holds the
workspace-level default command, hardware expectation and failure behavior
for every package.

## Purpose

`vyre-spec` is the frozen data contract. Its manifest declares no
dependencies at all, so any conformance runner can use it as the stable
contract for byte-identity proofs. Every type here is versioned wire
surface, and change discipline is the whole point.

## Critical invariants

- Every extensible enum carries an `Opaque` variant holding an extension id.
  A future variant must never break a downstream match.
- Every tag allocation is stable and recorded at the top of the file that
  owns the enum. `BinOp::Add` is `0x01` and `BinOp::Opaque` is `0x80`
  forever. Renumbering is a wire-format break and requires a major bump.
- Every `const fn` returns deterministic values. `DataType::min_bytes` and
  `DataType::size_bytes` are inputs to catalog generation.
- Zero runtime state. No `static`, no `OnceLock`, no lazy init.

## Adversarial surface

- `wire_tag_surface` makes the tag manifest executable. The frozen builtin
  tag for each core enum is public API, so a codec reads it instead of
  trusting a comment, and the builtin space excludes every extension id.
- `frozen_discriminants` and `extension_id_contracts` pin the allocation and
  reject an extension id that collides with the builtin range.
- `data_type_packed_size_adversarial` and `data_type_layout_matrix` drive the
  layout functions across the boundary values of every field.
- `sweep_wire_roundtrip_oracle_matrix` builds hostile programs over the
  frozen tag surface and pins byte-identical round-trip idempotence. It
  reaches encode and decode through the `vyre-foundation` dev-dependency,
  because what it proves is that the tag table is total under a codec, not
  how the codec is written.
- `token_ids_have_one_owner` proves no id is minted in two places.

## Cross-crate contracts

- `DataType`, `BinOp`, `UnOp`, `AtomicOp` and `RuleCondition` are consumed by
  `vyre-foundation`, `vyre-driver` and every dialect crate.
- `OpSignature`, `OpMetadata` and `IntrinsicDescriptor` are consumed by the
  conform runner and the catalog generators under `conform/`.
- `BackendId` and `Backend` are consumed by `vyre-driver` and the concrete
  driver crates.

## Bench targets

The crate declares no bench target. Its surface is `const fn` evaluation and
small value comparisons, and there is no runtime path to measure.

## Fuzz targets

The crate declares no fuzz target. It is data only, with no byte-in and
byte-out API. Fuzzing happens at `vyre-foundation::serial::wire`, where the
spec's tag table meets the decoder.

## What NOT to test here

- Decoder behavior on a malformed byte stream. The codec and its error
  surface belong to `vyre-foundation::serial::wire`.
- Backend dispatch semantics. Those live in the concrete driver crates and
  in `vyre-reference`.
- Op lowering. That lives in the emitter crates.

## Running

```bash
./cargo_full test -p vyre-spec
```
