# vyre-emit-ptx

Layer `emitter`. Owner `primary-binary-emitter`.

## Owns

Consume verified lowering products and emit the primary binary backend text
artifact.

The chapter is [crate boundaries](docs/architecture/crates.md#vyre-emit-naga-vyre-emit-ptx-vyre-emit-spirv-vyre-emit-metal).

## Must never contain

Lowering decisions, and any second copy of a translation another emitter
already owns. A fork here is two implementations of one language that drift
apart in the direction of whichever backend was debugged last.

## What crosses its edges

Out of this crate, into:

- `vyre-foundation` over the `foundation-ir` seam, private: typed IR, graph,
  diagnostics, validation, and semantic optimization contracts. Built when:
  always.
- `vyre-lower` over the `lowering` seam, public: verified backend-neutral
  representation lowering. Built when: always.

Into this crate, from:

- `vyre-bench` over the `primary-binary-emitter` seam, private.
- `vyre-driver-cuda` over the `primary-binary-emitter` seam, private.

## Direction that may not reverse

`vyre-foundation`, `vyre-lower` must never depend on `vyre-emit-ptx`. The edge
is one way: a cycle back into this crate makes the two crates one crate that
cannot be built, reviewed or published apart.

## Invariants

- Every edge above is declared in `docs/CRATE_OWNERSHIP.toml`. An edge in
  `Cargo.toml` that the registry does not carry, and a registry row no manifest
  declares, both fail.
- The crate declares 1 feature beyond `default`: `nvrtc`. It builds alone.
- The public surface is recorded in `docs/public-api/vyre-emit-ptx.txt`. An
  item added, removed or moved changes that file in the same change.

## Enforcing gates

`layering`, `dep-drift`, `workspace-membership`, `file-size`,
`public-api-snapshot`, `public-api-paths`, `feature-isolation`,
`feature-matrix`, `platform-boundary`, `neutral-crates`.
