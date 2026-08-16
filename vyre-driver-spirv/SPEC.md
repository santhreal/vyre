# vyre-driver-spirv

Layer `concrete-backend`. Owner `spirv-driver`.

## Owns

Own SPIR-V target compilation, immutable module-bundle emission, Vulkan
materialization and dispatch integration, and backend evidence.

The chapter is [crate boundaries](docs/architecture/crates.md#vyre-driver-cuda-vyre-driver-wgpu-vyre-driver-spirv-vyre-driver-metal-vyre-driver-reference).

## Must never contain

Anything another backend would also need. That belongs in `vyre-driver`, or it
becomes five copies that drift.

## What crosses its edges

Out of this crate, into:

- `vyre-driver` over the `backend-contract` seam, public: backend-neutral
  target, materialization, submission, and completion contracts. Built when:
  always.
- `vyre-emit-spirv` over the `spirv-emitter` seam, private: SPIR-V emission.
  Built when: always.
- `vyre-foundation` over the `foundation-ir` seam, public: typed IR, graph,
  diagnostics, validation, and semantic optimization contracts. Built when:
  always.
- `vyre-lower` over the `lowering` seam, private: verified backend-neutral
  representation lowering. Built when: always.
- `vyre-megakernel` over the `megakernel-compiler` seam, private: whole-graph
  compilation and immutable artifact contracts. Built when: always.
- `vyre-spec` over the `specification` seam, private: stable cross-engine
  schemas and operation definitions. Built when: always.

Into this crate, from:

- `vyre-registry-link` over the `spirv-driver` seam, private.

## Direction that may not reverse

`vyre-driver`, `vyre-emit-spirv`, `vyre-foundation`, `vyre-lower`,
`vyre-megakernel`, `vyre-spec` must never depend on `vyre-driver-spirv`. The
edge is one way: a cycle back into this crate makes the two crates one crate
that cannot be built, reviewed or published apart.

## Invariants

- Every edge above is declared in `docs/CRATE_OWNERSHIP.toml`. An edge in
  `Cargo.toml` that the registry does not carry, and a registry row no manifest
  declares, both fail.
- The crate declares 1 feature beyond `default`: `spirv-val`. It builds alone.
- The public surface is recorded in `docs/public-api/vyre-driver-spirv.txt`. An
  item added, removed or moved changes that file in the same change.

## Enforcing gates

`layering`, `dep-drift`, `workspace-membership`, `file-size`,
`public-api-snapshot`, `public-api-paths`, `feature-isolation`,
`feature-matrix`, `platform-boundary`, `neutral-crates`.
