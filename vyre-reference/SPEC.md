# vyre-reference

Layer `semantics`. Owner `reference-semantics`.

## Owns

The only crate permitted to compute on the CPU: the pure-Rust IR oracle. Not a
backend and not a fallback.

The chapter is [crate boundaries](docs/architecture/crates.md#vyre-reference).

## Must never contain

Performance work, and any role other than oracle. It is not a backend and not a
fallback. It exists so a backend's answer can be proved identical to a
definition, so speed here buys nothing and complexity costs the definition's
credibility.

## What crosses its edges

Out of this crate, into:

- `vyre-foundation` over the `foundation-ir` seam, public: typed IR, graph,
  diagnostics, validation, and semantic optimization contracts. Built when:
  always.
- `vyre-libs` over the `product-libraries` seam, public: host-side helpers the
  oracle shares with the composition it checks: the FNV-1a state functions and
  the DFA compiler. Built when: always.
- `vyre-primitives` over the `primitive-library` seam, public: the wire format,
  the marker types, and guarded IR construction. Built when: always.
- `vyre-spec` over the `specification` seam, public: stable cross-engine
  schemas and operation definitions. Built when: always.

Into this crate, from:

- `vyre-bench` over the `reference-semantics` seam, private.
- `vyre-conform` over the `reference-semantics` seam, private.
- `vyre-driver-reference` over the `reference-semantics` seam, private.
- `xtask-registry` over the `reference-semantics` seam, private.

## Direction that may not reverse

`vyre-foundation`, `vyre-libs`, `vyre-primitives`, `vyre-spec` must never
depend on `vyre-reference`. The edge is one way: a cycle back into this crate
makes the two crates one crate that cannot be built, reviewed or published
apart.

## Invariants

- Every edge above is declared in `docs/CRATE_OWNERSHIP.toml`. An edge in
  `Cargo.toml` that the registry does not carry, and a registry row no manifest
  declares, both fail.
- The crate declares 1 feature beyond `default`: `subgroup-ops`. It builds
  alone.
- The public surface is recorded in `docs/public-api/vyre-reference.txt`. An
  item added, removed or moved changes that file in the same change.

## Enforcing gates

`layering`, `dep-drift`, `workspace-membership`, `file-size`,
`public-api-snapshot`, `public-api-paths`, `feature-isolation`,
`feature-matrix`.
