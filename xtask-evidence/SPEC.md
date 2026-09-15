# xtask-evidence

Layer `tooling`. Owner `release-tooling`.

## Owns

Own the xtask subcommands that decide whether a recorded benchmark or release
measurement still describes this tree.

The chapter is [crate boundaries](../docs/architecture/crates.md#xtask-xtask-registry-xtask-evidence).

## Must never contain

An exemption. A gate that is allowed to be red is not a gate.

## What crosses its edges

Out of this crate, into:

- `xtask` over the `release-tooling` seam, private: subcommand registry,
  bounded readers, and release manifests. Built when: always.
- `vyre-bench` over the `benchmarks` seam, private: benchmark workloads and
  evidence. Built when: always.
- `vyre-foundation` over the `foundation-ir` seam, private: the release
  optimization family list the pass-family manifest is checked against. Built
  when: always.
- `vyre-driver` over the `backend-contract` seam, private: backend-neutral
  target, materialization, submission, and completion contracts. Built when:
  always.
- `vyre-registry-link` over the `registry-link` seam, private: linked inventory
  registry sources and the per-source floor. Built when: always.

No workspace member depends on this crate.

## Direction that may not reverse

`xtask`, `vyre-bench`, `vyre-foundation`, `vyre-driver`, `vyre-registry-link`
must never depend on `xtask-evidence`. The edge is one way: a cycle back into
this crate makes the two crates one crate that cannot be built, reviewed or
published apart.

## Invariants

- Every edge above is declared in `docs/CRATE_OWNERSHIP.toml`. An edge in
  `Cargo.toml` that the registry does not carry, and a registry row no manifest
  declares, both fail.
- The crate declares no feature beyond `default`, so every build of it is the
  same build.

## Enforcing gates

`layering`, `dep-drift`, `workspace-membership`, `file-size`.
