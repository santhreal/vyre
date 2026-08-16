# vyre-lower

Layer `lowering`. Owner `lowering`.

## Owns

Consume verified semantic programs and own the single backend-neutral lowering
boundary and pre-emission transforms.

The chapter is [crate boundaries](docs/architecture/crates.md#vyre-lower).

## Must never contain

Any dialect. The moment one leaks in, every emitter downstream inherits a
decision that was theirs to make, and the neutral descriptor stops being a
shared contract.

## What crosses its edges

Out of this crate, into:

- `vyre-foundation` over the `foundation-ir` seam, public: typed IR, graph,
  diagnostics, validation, and semantic optimization contracts. Built when:
  always.

Into this crate, from:

- `vyre-bench` over the `lowering` seam, private.
- `vyre-debug` over the `lowering` seam, public.
- `vyre-driver-cuda` over the `lowering` seam, private.
- `vyre-driver-metal` over the `lowering` seam, private.
- `vyre-driver-spirv` over the `lowering` seam, private.
- `vyre-driver-wgpu` over the `lowering` seam, private.
- `vyre-emit-metal` over the `lowering` seam, public.
- `vyre-emit-naga` over the `lowering` seam, public.
- `vyre-emit-ptx` over the `lowering` seam, public.
- `vyre-emit-spirv` over the `lowering` seam, public.
- `vyre-megakernel` over the `lowering` seam, private.

## Direction that may not reverse

`vyre-foundation` must never depend on `vyre-lower`. The edge is one way: a
cycle back into this crate makes the two crates one crate that cannot be built,
reviewed or published apart.

## Invariants

- Every edge above is declared in `docs/CRATE_OWNERSHIP.toml`. An edge in
  `Cargo.toml` that the registry does not carry, and a registry row no manifest
  declares, both fail.
- The crate declares 1 feature beyond `default`: `test-fixtures`. It builds
  alone.
- The public surface is recorded in `docs/public-api/vyre-lower.txt`. An item
  added, removed or moved changes that file in the same change.

## Enforcing gates

`layering`, `dep-drift`, `workspace-membership`, `file-size`,
`public-api-snapshot`, `public-api-paths`, `feature-isolation`,
`feature-matrix`.
