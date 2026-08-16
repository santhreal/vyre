# xtask

Layer `tooling`. Owner `release-tooling`.

## Owns

Own the subcommand registry and every gate that judges the tree from source
text, manifests, workflows, and recorded evidence, linking no vyre crate.

The chapter is [crate boundaries](docs/architecture/crates.md#xtask-xtask-registry-xtask-evidence).

## Must never contain

An exemption. A gate that is allowed to be red is not a gate.

## What crosses its edges

Out of this crate, into:

- `structure-gate` over the `release-tooling` seam, private: resolve the
  checkout a gate reports on from the working directory at run time. Built
  when: always.

Into this crate, from:

- `vyre-bench` over the `release-tooling` seam, private.
- `xtask-evidence` over the `release-tooling` seam, private.
- `xtask-registry` over the `release-tooling` seam, private.

## Direction that may not reverse

`structure-gate` must never depend on `xtask`. The edge is one way: a cycle
back into this crate makes the two crates one crate that cannot be built,
reviewed or published apart.

## Invariants

- Every edge above is declared in `docs/CRATE_OWNERSHIP.toml`. An edge in
  `Cargo.toml` that the registry does not carry, and a registry row no manifest
  declares, both fail.
- The crate declares no feature beyond `default`, so every build of it is the
  same build.

## Enforcing gates

`layering`, `dep-drift`, `workspace-membership`, `file-size`.
