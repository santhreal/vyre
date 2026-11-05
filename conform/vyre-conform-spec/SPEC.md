# vyre-conform-spec

Layer `conformance`. Owner `conformance`.

## Owns

Define conformance case, result, and certificate schemas against the public
facade.

The chapter is [crate boundaries](../../docs/architecture/crates.md#vyre-conform-and-vyre-conform-spec).

## Must never contain

A pass a backend can satisfy without producing the reference's bytes. A soft
pass is worse than no suite.

## What crosses its edges

Out of this crate, into:

- `vyre-spec` over the `specification` seam, private: stable cross-engine
  schemas and operation definitions. Built when: always.

Into this crate, from:

- `vyre-conform` over the `conformance` seam, private.

## Direction that may not reverse

`vyre-spec` must never depend on `vyre-conform-spec`. The edge is one way: a
cycle back into this crate makes the two crates one crate that cannot be built,
reviewed or published apart.

## Invariants

- Every edge above is declared in `docs/CRATE_OWNERSHIP.toml`. An edge in
  `Cargo.toml` that the registry does not carry, and a registry row no manifest
  declares, both fail.
- The crate declares no feature beyond `default`, so every build of it is the
  same build.

## Enforcing gates

`layering`, `dep-drift`, `workspace-membership`, `file-size`.
