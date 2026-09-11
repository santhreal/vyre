# vyre-driver-wgpu

Layer `concrete-backend`. Owner `portable-driver`.

## Owns

Own pure WGSL target compilation, portable GPU acquisition, materialization,
dispatch, graph execution, and backend evidence.

The chapter is [crate boundaries](../docs/architecture/crates.md#vyre-driver-cuda-vyre-driver-wgpu-vyre-driver-spirv-vyre-driver-metal-vyre-driver-reference).

## Must never contain

Anything another backend would also need. That belongs in `vyre-driver`, or it
becomes five copies that drift.

## What crosses its edges

Out of this crate, into:

- `vyre-libs` over the `product-libraries` seam, private: composition trees the
  portable adapters plan against. Built when: always.
- `vyre-driver` over the `backend-contract` seam, public: backend-neutral
  target, materialization, submission, and completion contracts. Built when:
  always.
- `vyre-emit-naga` over the `primary-text-emitter` seam, private: primary text
  and related binary emission. Built when: always.
- `vyre-foundation` over the `foundation-ir` seam, public: typed IR, graph,
  diagnostics, validation, and semantic optimization contracts. Built when:
  always.
- `vyre-lower` over the `lowering` seam, private: verified backend-neutral
  representation lowering. Built when: always.
- `vyre-megakernel` over the `megakernel-compiler` seam, private: whole-graph
  compilation and immutable artifact contracts. Built when: always.
- `vyre-pass-engine` over the `pass-engine` seam, public: optimizer pass
  execution as dispatched Vyre Programs. Built when: always.
- `vyre-spec` over the `specification` seam, public: stable cross-engine
  schemas and operation definitions. Built when: always.

Into this crate, from:

- `vyre` over the `portable-driver` seam, private.
- `vyre-bench` over the `portable-driver` seam, private.
- `vyre-conform` over the `portable-driver` seam, private.
- `vyre-registry-link` over the `portable-driver` seam, private.

## Direction that may not reverse

`vyre-libs`, `vyre-driver`, `vyre-emit-naga`, `vyre-foundation`, `vyre-lower`,
`vyre-megakernel`, `vyre-pass-engine`, `vyre-spec` must never depend on
`vyre-driver-wgpu`. The edge is one way: a cycle back into this crate makes the
two crates one crate that cannot be built, reviewed or published apart.

## Invariants

- Every edge above is declared in `docs/CRATE_OWNERSHIP.toml`. An edge in
  `Cargo.toml` that the registry does not carry, and a registry row no manifest
  declares, both fail.
- The crate declares 8 features beyond `default`: `pattern-dfa`,
  `pattern-nfa`, `pattern-substring`, `math-linalg`, `math-scan`,
  `nn-attention`, `parity-testing`, `wgpu`. Each builds alone.
- The public surface is recorded in `docs/public-api/vyre-driver-wgpu.txt`. An
  item added, removed or moved changes that file in the same change.

## Enforcing gates

`layering`, `dep-drift`, `workspace-membership`, `file-size`,
`public-api-snapshot`, `public-api-paths`, `feature-isolation`,
`feature-matrix`, `platform-boundary`, `neutral-crates`.
