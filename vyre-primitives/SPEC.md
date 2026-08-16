# vyre-primitives

Layer `primitives`. Owner `primitive-library`.

## Owns

Own marker types and uncomposable hardware intrinsics. A composition belongs in
vyre-libs, not here.

The chapter is [crate boundaries](docs/architecture/crates.md#vyre-primitives).

## Must never contain

Compositions. Admission is by whether the operation can be composed, never by
how many callers it has.

## What crosses its edges

Out of this crate, into:

- `vyre-foundation` over the `foundation-ir` seam, private: typed IR, graph,
  diagnostics, validation, and semantic optimization contracts. Built when:
  always.
- `vyre-spec` over the `specification` seam, private: stable cross-engine
  schemas and operation definitions. Built when: always.

Into this crate, from:

- `vyre-aot` over the `primitive-library` seam, private.
- `vyre-bench` over the `primitive-library` seam, private.
- `vyre-conform` over the `primitive-library` seam, private.
- `vyre-debug` over the `primitive-library` seam, private.
- `vyre-libs` over the `primitive-library` seam, public.
- `vyre-pass-engine` over the `primitive-library` seam, public.
- `vyre-reference` over the `primitive-library` seam, public.
- `vyre-registry-link` over the `primitive-library` seam, private.
- `vyre-runtime` over the `primitive-library` seam, public.
- `xtask-registry` over the `primitive-library` seam, private.

## Direction that may not reverse

`vyre-foundation`, `vyre-spec` must never depend on `vyre-primitives`. The edge
is one way: a cycle back into this crate makes the two crates one crate that
cannot be built, reviewed or published apart.

## Invariants

- Every edge above is declared in `docs/CRATE_OWNERSHIP.toml`. An edge in
  `Cargo.toml` that the registry does not carry, and a registry row no manifest
  declares, both fail.
- The crate declares 5 features beyond `default`: `cpu-parity`, `gpu`,
  `hardware`, `inventory-registry`, `vyre-foundation`. Each builds alone.
- The public surface is recorded in `docs/public-api/vyre-primitives.txt`. An
  item added, removed or moved changes that file in the same change.

## Enforcing gates

`layering`, `dep-drift`, `workspace-membership`, `file-size`,
`public-api-snapshot`, `public-api-paths`, `feature-isolation`,
`feature-matrix`.
