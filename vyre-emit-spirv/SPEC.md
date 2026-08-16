# vyre-emit-spirv

Layer `emitter`. Owner `spirv-emitter`.

## Owns

Consume verified lowering products and emit SPIR-V artifacts through the shared
writer.

The chapter is [crate boundaries](docs/architecture/crates.md#vyre-emit-naga-vyre-emit-ptx-vyre-emit-spirv-vyre-emit-metal).

## Must never contain

Lowering decisions, and any second copy of a translation another emitter
already owns. A fork here is two implementations of one language that drift
apart in the direction of whichever backend was debugged last.

## What crosses its edges

Out of this crate, into:

- `vyre-emit-naga` over the `primary-text-emitter` seam, public: primary text
  and related binary emission. Built when: always.
- `vyre-lower` over the `lowering` seam, public: verified backend-neutral
  representation lowering. Built when: always.

Into this crate, from:

- `vyre-driver-spirv` over the `spirv-emitter` seam, private.

## Direction that may not reverse

`vyre-emit-naga`, `vyre-lower` must never depend on `vyre-emit-spirv`. The edge
is one way: a cycle back into this crate makes the two crates one crate that
cannot be built, reviewed or published apart.

## Invariants

- Every edge above is declared in `docs/CRATE_OWNERSHIP.toml`. An edge in
  `Cargo.toml` that the registry does not carry, and a registry row no manifest
  declares, both fail.
- The crate declares no feature beyond `default`, so every build of it is the
  same build.
- The public surface is recorded in `docs/public-api/vyre-emit-spirv.txt`. An
  item added, removed or moved changes that file in the same change.

## Enforcing gates

`layering`, `dep-drift`, `workspace-membership`, `file-size`,
`public-api-snapshot`, `public-api-paths`, `platform-boundary`,
`neutral-crates`.
