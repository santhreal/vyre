# vyre-debug

Layer `tooling`. Owner `debugging`.

## Owns

Inspect, explain, and diagnose typed programs, lowering, and product-library
composition.

The chapter is [crate boundaries](../docs/architecture/crates.md#vyre-debug).

## Must never contain

Anything a user's program depends on. This crate reads what other crates
produced.

## What crosses its edges

Out of this crate, into:

- `vyre` over the `public-facade` seam, private: public lifecycle facade. Built
  when: always.
- `vyre-emit-naga` over the `primary-text-emitter` seam, public: primary text
  and related binary emission. Built when: always.
- `vyre-foundation` over the `foundation-ir` seam, public: typed IR, graph,
  diagnostics, validation, and semantic optimization contracts. Built when:
  always.
- `vyre-libs` over the `product-libraries` seam, private: product operation
  builders. Built when: always.
- `vyre-lower` over the `lowering` seam, public: verified backend-neutral
  representation lowering. Built when: always.

No workspace member depends on this crate.

## Direction that may not reverse

`vyre`, `vyre-emit-naga`, `vyre-foundation`, `vyre-libs`, and `vyre-lower`
must never depend on `vyre-debug`. The edge is one way: a
cycle back into this crate makes the two crates one crate that cannot be built,
reviewed or published apart.

## Invariants

- Every edge above is declared in `docs/CRATE_OWNERSHIP.toml`. An edge in
  `Cargo.toml` that the registry does not carry, and a registry row no manifest
  declares, both fail.
- The crate declares no feature beyond `default`, so every build of it is the
  same build.
- The public surface is recorded in `docs/public-api/vyre-debug.txt`. An item
  added, removed or moved changes that file in the same change.

## Enforcing gates

`layering`, `dep-drift`, `workspace-membership`, `file-size`,
`public-api-snapshot`, `public-api-paths`.
