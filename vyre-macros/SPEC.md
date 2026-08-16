# vyre-macros

Layer `foundation`. Owner `registration-macros`.

## Owns

Provide compile-time registration and declaration macros without depending on
runtime crates.

The chapter is [crate boundaries](../docs/architecture/crates.md#vyre-macros).

## Must never contain

Logic. A macro that decides something moves the decision out of source a reader
can grep and into an expansion they cannot.

## What crosses its edges

This crate depends on no other workspace member.

Into this crate, from:

- `vyre-foundation` over the `registration-macros` seam, private.

## Direction that may not reverse

`vyre-macros` takes no workspace dependency. Adding one reverses the direction
every crate above it relies on.

## Invariants

- Every edge above is declared in `docs/CRATE_OWNERSHIP.toml`. An edge in
  `Cargo.toml` that the registry does not carry, and a registry row no manifest
  declares, both fail.
- The crate declares no feature beyond `default`, so every build of it is the
  same build.
- The public surface is recorded in `docs/public-api/vyre-macros.txt`. An item
  added, removed or moved changes that file in the same change.

## Enforcing gates

`layering`, `dep-drift`, `workspace-membership`, `file-size`,
`public-api-snapshot`, `public-api-paths`.
