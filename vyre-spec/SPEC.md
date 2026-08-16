# vyre-spec

Layer `foundation`. Owner `specification`.

## Owns

Own stable schemas, operation definitions, and compatibility contracts without
runtime dependencies.

The chapter is [crate boundaries](docs/architecture/crates.md#vyre-spec).

## Must never contain

Anything that executes, allocates or decides. A behavioural change in a crate
this widely depended on is a change to every crate at once.

## What crosses its edges

This crate depends on no other workspace member.

Into this crate, from:

- `vyre` over the `specification` seam, private.
- `vyre-aot` over the `specification` seam, private.
- `vyre-bench` over the `specification` seam, private.
- `vyre-conform` over the `specification` seam, private.
- `vyre-conform-spec` over the `specification` seam, private.
- `vyre-driver` over the `specification` seam, public.
- `vyre-driver-cuda` over the `specification` seam, private.
- `vyre-driver-spirv` over the `specification` seam, private.
- `vyre-driver-wgpu` over the `specification` seam, public.
- `vyre-foundation` over the `specification` seam, public.
- `vyre-libs` over the `specification` seam, public.
- `vyre-primitives` over the `specification` seam, private.
- `vyre-reference` over the `specification` seam, public.
- `vyre-test-support` over the `specification` seam, private.
- `xtask-registry` over the `specification` seam, private.

## Direction that may not reverse

`vyre-spec` takes no workspace dependency. Adding one reverses the direction
every crate above it relies on.

## Invariants

- Every edge above is declared in `docs/CRATE_OWNERSHIP.toml`. An edge in
  `Cargo.toml` that the registry does not carry, and a registry row no manifest
  declares, both fail.
- The crate declares no feature beyond `default`, so every build of it is the
  same build.
- The public surface is recorded in `docs/public-api/vyre-spec.txt`. An item
  added, removed or moved changes that file in the same change.

## Enforcing gates

`layering`, `dep-drift`, `workspace-membership`, `file-size`,
`public-api-snapshot`, `public-api-paths`.
