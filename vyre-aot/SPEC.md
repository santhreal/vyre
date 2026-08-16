# vyre-aot

Layer `packaging`. Owner `aot-artifacts`.

## Owns

Package the same megakernel artifact class ahead of time. Not a second compile
path. No workspace crate currently depends on this one.

The chapter is [crate boundaries](docs/architecture/crates.md#vyre-aot).

## Must never contain

A second compile path. It produces the same artifact class as every other
route, or the bundle is a different compiler with the same name.

## What crosses its edges

Out of this crate, into:

- `vyre-driver` over the `backend-contract` seam, public: backend-neutral
  target, materialization, submission, and completion contracts. Built when:
  always.
- `vyre-foundation` over the `foundation-ir` seam, public: typed IR, graph,
  diagnostics, validation, and semantic optimization contracts. Built when:
  always.
- `vyre-megakernel` over the `megakernel-compiler` seam, public: whole-graph
  compilation and immutable artifact contracts. Built when: always.
- `vyre-primitives` over the `primitive-library` seam, private: reusable
  semantic Program builders. Built when: always.
- `vyre-spec` over the `specification` seam, private: stable cross-engine
  schemas and operation definitions. Built when: always.

No workspace member depends on this crate.

## Direction that may not reverse

`vyre-driver`, `vyre-foundation`, `vyre-megakernel`, `vyre-primitives`,
`vyre-spec` must never depend on `vyre-aot`. The edge is one way: a cycle back
into this crate makes the two crates one crate that cannot be built, reviewed
or published apart.

## Invariants

- Every edge above is declared in `docs/CRATE_OWNERSHIP.toml`. An edge in
  `Cargo.toml` that the registry does not carry, and a registry row no manifest
  declares, both fail.
- The crate declares 1 feature beyond `default`: `ptx`. It builds alone.
- The public surface is recorded in `docs/public-api/vyre-aot.txt`. An item
  added, removed or moved changes that file in the same change.

## Enforcing gates

`layering`, `dep-drift`, `workspace-membership`, `file-size`,
`public-api-snapshot`, `public-api-paths`, `feature-isolation`,
`feature-matrix`.
