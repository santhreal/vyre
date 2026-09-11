# vyre-driver

Layer `backend-neutral`. Owner `backend-contract`.

## Owns

Define backend-neutral device, target compiler registration, artifact
materialization, binding, submission, completion, capability, dispatch, and
evidence contracts.

The chapter is [crate boundaries](../docs/architecture/crates.md#vyre-driver).

## Must never contain

A driver name, a dialect string, or a backend-specific error message. This
crate is what makes the backends interchangeable, and a concrete detail
admitted here is a detail every backend must then pretend to have.

## What crosses its edges

Out of this crate, into:

- `vyre-foundation` over the `foundation-ir` seam, public: typed IR, graph,
  diagnostics, validation, and semantic optimization contracts. Built when:
  always.
- `vyre-megakernel` over the `megakernel-compiler` seam, public: whole-graph
  compilation and immutable artifact contracts. Built when: always.
- `vyre-libs` over the `product-libraries` seam, private: composition library
  the driver adapters plan against. Built when: always.
- `vyre-spec` over the `specification` seam, public: stable cross-engine
  schemas and operation definitions. Built when: always.

Into this crate, from:

- `vyre` over the `backend-contract` seam, private.
- `vyre-aot` over the `backend-contract` seam, public.
- `vyre-bench` over the `backend-contract` seam, private.
- `vyre-conform` over the `backend-contract` seam, private.
- `vyre-driver-cuda` over the `backend-contract` seam, public.
- `vyre-driver-metal` over the `backend-contract` seam, public.
- `vyre-driver-reference` over the `backend-contract` seam, public.
- `vyre-driver-spirv` over the `backend-contract` seam, public.
- `vyre-driver-wgpu` over the `backend-contract` seam, public.
- `vyre-registry-link` over the `backend-contract` seam, private.
- `vyre-runtime` over the `backend-contract` seam, public.
- `xtask-evidence` over the `backend-contract` seam, private.
- `xtask-registry` over the `backend-contract` seam, private.

## Direction that may not reverse

`vyre-foundation`, `vyre-megakernel`, `vyre-libs`, `vyre-spec` must never
depend on `vyre-driver`. The edge is one way: a cycle back into this crate
makes the two crates one crate that cannot be built, reviewed or published
apart.

## Invariants

- Every edge above is declared in `docs/CRATE_OWNERSHIP.toml`. An edge in
  `Cargo.toml` that the registry does not carry, and a registry row no manifest
  declares, both fail.
- The crate declares 2 features beyond `default`: `libs-compositions`,
  `test-fixtures`. Each builds alone.
- The public surface is recorded in `docs/public-api/vyre-driver.txt`. An item
  added, removed or moved changes that file in the same change.

## Enforcing gates

`layering`, `dep-drift`, `workspace-membership`, `file-size`,
`public-api-snapshot`, `public-api-paths`, `feature-isolation`,
`feature-matrix`, `platform-boundary`, `neutral-crates`.
