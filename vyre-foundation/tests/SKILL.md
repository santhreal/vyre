# tests/SKILL.md, vyre-foundation

One test file per contract. A file is named for the contract it proves, and
the directory has no catch-all target. `docs/testing/TESTING.toml` holds the
workspace-level default command, hardware expectation and failure behavior
for every package.

## Purpose

`vyre-foundation` owns the typed IR, the validator, the optimizer framework
and the serialization formats. Every program lives here, every decoder starts
here, and every optimizer pass runs here. It depends on `vyre-spec` and
`vyre-macros` and on nothing else in the workspace.

## Critical invariants

- Wire round-trip. `from_wire(to_wire(p)) == p` for every program `p` that
  passes `validate`. Encoder and decoder stay in lockstep through every
  `#[non_exhaustive]` variant addition.
- Unknown tags never panic. Every byte in every position of a decoder input
  either decodes cleanly or returns an actionable `Fix:`-prefixed error.
  Random bytes through the decoder never abort the process.
- Opaque round-trip preservation. Unknown extension bytes on the `0x80`
  opaque path are preserved byte-identically even when the decoder does not
  link the extension crate.
- Optimizer preserves semantics. `eval(p) == eval(optimize(p))` for every
  valid `p`.
- A traversal reaches every variant or refuses to compile.
  `transform::visit::node_shape` and `transform::visit::child_bodies` match
  every `Node` variant with no catch-all arm, so adding a variant fails to
  compile until the author records whether it nests bodies, owns operands, or
  is opaque. `tests/node_variant_traversal_closure.rs` owns that class and
  states why a match-count proxy replaced nothing.

## Adversarial surface

- Corrupted and truncated payloads. `wire_adversarial`,
  `wire_decode_corruption` and `serial_envelope_corruption` cover a valid
  magic over a truncated body, a length prefix pointing past the end of
  input, and a corrupted envelope frame.
- Hostile inputs generated rather than hand-picked.
  `wire_generated_hostile_inputs` enumerates the shapes, so a new tag is
  covered without anyone remembering to add a case.
- Resource exhaustion. `wire_decode_oom_guard`, `validation_depth_limits` and
  `resource_exhaustion_adversarial` prove a hostile size or nesting depth is
  refused with a diagnostic instead of consuming the host.
- Extension payloads. `extension_adversarial`, `opaque_payload_endian` and
  `opaque_wire_round_trip` prove an unlinked extension survives a round trip
  byte for byte, including its endianness.
- Version skew. `wire_version_mismatch` proves a payload from another format
  version is refused rather than misread.
- Type and ordering boundaries. `type_boundary_adversarial`,
  `memory_ordering_adversarial` and `region_chain_adversarial` cover the
  edges of the type lattice, the memory model and region nesting.

## Cross-crate contracts

- `Program`, `Expr`, `Node`, `BufferDecl`, `MemoryKind` and `BufferAccess` are
  consumed by every backend and by conform.
- `ExprVisitor` and `NodeVisitor` are consumed by the lowering crates and the
  reference interpreter.
- `OperationRegistry` and `SemanticOperation` are consumed by
  `vyre-driver::backend::registry`, by every primitive domain and by every
  backend.
- `TargetOperationFacet` and `TargetId` are built from target-owned
  registrations and consumed by `vyre-reference` and conform.

## Bench targets

The crate declares one bench, `optimizer_pipeline`. It measures the pass
pipeline end to end over the program corpus, which is where a regression in
per-pass allocation or in pass ordering shows up.

## Fuzz targets

`vyre-foundation/fuzz` declares three targets: `decoder`, `program_wire` and
`registry_toml`. The fuzz package is excluded from the workspace and its
README owns the commands.

## What NOT to test here

- Concrete backend dispatch. That belongs to the owning backend crate.
- Backend-specific lowering. That belongs to the owning emitter crate.
- Primitive builder semantics. Those belong to `vyre-primitives`.

## Running

```bash
./cargo_full test -p vyre-foundation
./cargo_full test -p vyre-foundation --all-features
./cargo_full bench -p vyre-foundation --bench optimizer_pipeline
```
