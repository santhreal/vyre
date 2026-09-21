# vyre-driver-metal

Layer `concrete-backend`. Owner `metal-driver`.

## Owns

Own pure MSL target compilation, native Apple device acquisition,
materialization, dispatch, and backend evidence.

The chapter is [crate boundaries](../docs/architecture/crates.md#vyre-driver-cuda-vyre-driver-wgpu-vyre-driver-spirv-vyre-driver-metal-vyre-driver-reference).

## Must never contain

Anything another backend would also need. That belongs in `vyre-driver`, or it
becomes five copies that drift.

## What crosses its edges

Out of this crate, into:

- `vyre-driver` over the `backend-contract` seam, public: backend-neutral
  target, materialization, submission, and completion contracts. Built when:
  always.
- `vyre-emit-metal` over the `metal-emitter` seam, private: native Apple source
  emission. Built when: always.
- `vyre-foundation` over the `foundation-ir` seam, private: typed IR, graph,
  diagnostics, validation, and semantic optimization contracts. Built when:
  always.
- `vyre-lower` over the `lowering` seam, private: verified backend-neutral
  representation lowering. Built when: always.
- `vyre-megakernel` over the `megakernel-compiler` seam, private: whole-graph
  compilation and immutable artifact contracts. Built when: always.

Into this crate, from:

- `vyre-registry-link` over the `metal-driver` seam, private.

## Direction that may not reverse

`vyre-driver`, `vyre-emit-metal`, `vyre-foundation`, `vyre-lower`,
`vyre-megakernel` must never depend on `vyre-driver-metal`. The edge is one
way: a cycle back into this crate makes the two crates one crate that cannot be
built, reviewed or published apart.

## Invariants

- Every edge above is declared in `docs/CRATE_OWNERSHIP.toml`. An edge in
  `Cargo.toml` that the registry does not carry, and a registry row no manifest
  declares, both fail.
- The crate declares no feature beyond `default`, so every build of it is the
  same build.
- The public surface is recorded in `docs/public-api/vyre-driver-metal.txt`. An
  item added, removed or moved changes that file in the same change.

## Enforcing gates

`layering`, `dep-drift`, `workspace-membership`, `file-size`,
`public-api-snapshot`, `public-api-paths`, `platform-boundary`,
`neutral-crates`.
