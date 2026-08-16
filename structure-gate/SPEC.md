# structure-gate

Layer `standalone-tooling`. Owner `release-tooling`.

## Owns

Enforce the crate roster, one operation identity per semantic operation, one
home per concept, and one place per module. Depends on no vyre crate so it
keeps running while the workspace does not compile.

The chapter is [crate boundaries](docs/architecture/crates.md#structure-gate).

## Must never contain

A second source scanner. A bad masker desynchronizes a brace matcher on the
first raw string it meets, and a contract built on a bad masker reports
confident nonsense.

## What crosses its edges

This crate depends on no other workspace member.

Into this crate, from:

- `vyre-test-support` over the `release-tooling` seam, private.
- `xtask` over the `release-tooling` seam, private.

## Direction that may not reverse

`structure-gate` takes no workspace dependency. Adding one reverses the
direction every crate above it relies on.

## Invariants

- Every edge above is declared in `docs/CRATE_OWNERSHIP.toml`. An edge in
  `Cargo.toml` that the registry does not carry, and a registry row no manifest
  declares, both fail.
- The crate declares no feature beyond `default`, so every build of it is the
  same build.

## Enforcing gates

`layering`, `dep-drift`, `workspace-membership`, `file-size`.
