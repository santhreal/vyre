# vyre-lints

Layer `tooling`. Owner `lint-policy`.

## Owns

Enforce source-level project policies without depending on runtime crates.

The chapter is [crate boundaries](../docs/architecture/crates.md#vyre-lints).

## Must never contain

(none recorded)

## What crosses its edges

This crate depends on no other workspace member.

No workspace member depends on this crate.

## Direction that may not reverse

`vyre-lints` takes no workspace dependency. Adding one reverses the direction
every crate above it relies on.

## Invariants

- Every edge above is declared in `docs/CRATE_OWNERSHIP.toml`. An edge in
  `Cargo.toml` that the registry does not carry, and a registry row no manifest
  declares, both fail.
- The crate declares no feature beyond `default`, so every build of it is the
  same build.
- The public surface is recorded in `docs/public-api/vyre-lints.txt`. An item
  added, removed or moved changes that file in the same change.

## Enforcing gates

`layering`, `dep-drift`, `workspace-membership`, `file-size`,
`public-api-snapshot`, `public-api-paths`.
